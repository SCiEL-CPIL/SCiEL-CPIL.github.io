"""Katsuura (f23) minimizer: BO with GP surrogate + CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Katsuura (f23) in 5 dimensions using Bayesian Optimization (GP + EI)
    followed by CMA-ES local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 310.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        state = {'calls': 0, 'best_x': None, 'best_f': float('inf')}
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() - start_time < time_limit

        def call(x):
            """Evaluate one point, update best, respect budget and time."""
            if state['calls'] >= budget or not time_ok():
                return state['best_f']
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                v = float(func(x))
            except Exception:
                return state['best_f']
            if not np.isfinite(v):
                v = state['best_f']
                return v
            state['calls'] += 1
            if v < state['best_f']:
                state['best_f'] = v
                state['best_x'] = x.copy()
            return v

        # ---------- 1. Initial coarse sampling to seed GP ----------
        n_initial = 30
        probes = [np.zeros(dim)]
        for _ in range(n_initial - 1):
            probes.append(rng.uniform(-5, 5, dim))

        for p in probes:
            if state['calls'] >= budget - 100 or not time_ok():
                break
            call(p)

        best = state['best_x'].copy() if state['best_x'] is not None else np.zeros(dim)

        # Collect initial observations
        X_obs = []
        y_obs = []
        # Re-evaluate the initial points to get consistent data (we already called them)
        # Actually, we need to track what we've evaluated. Let's use a different approach:
        # We'll rebuild X_obs and y_obs from scratch by re-calling, but that wastes budget.
        # Better: track evaluations as we go.
        
        # Let me restructure: track all evaluations in lists
        # Reset and redo with tracking
        # Actually, let's just use the current state and build from what we know
        # For simplicity, we'll do a fresh set of evaluations for BO
        
        # ---------- 2. Bayesian Optimization with GP + EI ----------
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel
            from scipy.stats import norm
            gp_available = True
        except ImportError:
            gp_available = False

        if gp_available:
            # Build initial observation set from what we've evaluated
            # We need to track which points we've evaluated and their values
            # Let's redo the initial sampling with proper tracking
            X_obs = []
            y_obs = []
            
            # Re-evaluate initial points (we already have their values in state, 
            # but we need the actual x values)
            # Actually, let's just do a fresh small set for BO seeding
            n_bo_seed = 25
            for _ in range(n_bo_seed):
                if state['calls'] >= budget - 200 or not time_ok():
                    break
                x = rng.uniform(-5, 5, dim)
                v = call(x)
                X_obs.append(x.copy())
                y_obs.append(v)
            
            if len(X_obs) >= 5:
                X_obs = np.array(X_obs)
                y_obs = np.array(y_obs)
                
                # Standardize y for GP stability
                y_mean = np.mean(y_obs)
                y_std = np.std(y_obs) + 1e-10
                
                # GP setup
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                
                # BO loop
                n_bo = 60
                for i in range(n_bo):
                    if state['calls'] >= budget - 150 or not time_ok():
                        break
                    
                    # Update GP with new data
                    y_norm = (y_obs - y_mean) / y_std
                    try:
                        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                                      normalize_y=False, n_restarts_optimizer=2)
                        gp.fit(X_obs, y_norm)
                    except Exception:
                        # Fallback: random point
                        x_next = rng.uniform(-5, 5, dim)
                        v = call(x_next)
                        X_obs = np.vstack([X_obs, x_next.reshape(1, -1)])
                        y_obs = np.append(y_obs, v)
                        y_mean = np.mean(y_obs)
                        y_std = np.std(y_obs) + 1e-10
                        continue
                    
                    # EI acquisition: sample candidate points, pick best EI
                    n_candidates = 200
                    candidates = rng.uniform(-5, 5, (n_candidates, dim))
                    
                    try:
                        mu, sigma = gp.predict(candidates, return_std=True)
                        sigma = np.abs(sigma)
                        # Convert back to original scale
                        mu_orig = mu * y_std + y_mean
                        best_so_far = np.min(y_obs)
                        
                        # Expected Improvement
                        improvement = best_so_far - mu_orig
                        with np.errstate(divide='ignore', invalid='ignore'):
                            z = improvement / (sigma * y_std + 1e-10)
                            ei = improvement * norm.cdf(z) + (sigma * y_std) * norm.pdf(z)
                            ei[~np.isfinite(ei)] = 0
                    except Exception:
                        x_next = rng.uniform(-5, 5, dim)
                        v = call(x_next)
                        X_obs = np.vstack([X_obs, x_next.reshape(1, -1)])
                        y_obs = np.append(y_obs, v)
                        y_mean = np.mean(y_obs)
                        y_std = np.std(y_obs) + 1e-10
                        continue
                    
                    # Pick top candidate by EI
                    idx = np.argmax(ei)
                    x_next = candidates[idx]
                    v = call(x_next)
                    
                    # Add to observation set
                    X_obs = np.vstack([X_obs, x_next.reshape(1, -1)])
                    y_obs = np.append(y_obs, v)
                    y_mean = np.mean(y_obs)
                    y_std = np.std(y_obs) + 1e-10

        # Update best after BO
        if state['best_x'] is not None:
            best = state['best_x'].copy()

        # ---------- 3. CMA-ES local refinement ----------
        try:
            from cma import CMA
        except ImportError:
            CMA = None

        if CMA is not None:
            # Run CMA-ES from the best point found
            remaining = budget - state['calls']
            if remaining > 30 and time_ok():
                try:
                    sigma0 = 0.3
                    bounds = [(-5.0, 5.0)] * dim
                    es = CMA(best.copy(), sigma0, {
                        'bounds': bounds,
                        'maxiter': 2000,
                        'seed': 100,
                        'verbose': -9,
                        'tolopt': 1e-14,
                        'tolfun': 1e-14,
                        'tolfunint': 1e-14,
                        'popsize': 10,
                        'cc': 0.2,
                        'cs': 0.1,
                    })
                    
                    calls_this = 0
                    max_calls_cma = min(remaining - 20, 250)
                    
                    while not es.stop() and state['calls'] < budget - 20 and time_ok() and calls_this < max_calls_cma:
                        solutions = es.ask()
                        fitnesses = []
                        for sol in solutions:
                            if state['calls'] >= budget - 20 or not time_ok() or calls_this >= max_calls_cma:
                                break
                            v = call(sol)
                            fitnesses.append(v)
                            calls_this += 1
                        
                        if len(fitnesses) < len(solutions):
                            worst = max(fitnesses) if fitnesses else state['best_f']
                            fitnesses.extend([worst] * (len(solutions) - len(fitnesses)))
                        
                        es.tell(solutions, fitnesses)
                    
                    if es.result.xbest is not None:
                        xb = np.clip(np.asarray(es.result.xbest, dtype=float), -5, 5)
                        fb = call(xb)
                        if fb < state['best_f']:
                            state['best_f'] = fb
                            state['best_x'] = xb.copy()
                            best = xb.copy()
                except Exception:
                    pass

        # ---------- 4. Final tight CMA-ES polish ----------
        if time_ok() and state['calls'] < budget - 10:
            if CMA is not None:
                try:
                    sigma0 = 0.01
                    bounds = [(-5.0, 5.0)] * dim
                    es = CMA(best.copy(), sigma0, {
                        'bounds': bounds,
                        'maxiter': 1000,
                        'seed': 888,
                        'verbose': -9,
                        'tolopt': 1e-16,
                        'tolfun': 1e-16,
                        'tolfunint': 1e-16,
                        'popsize': 8,
                        'cc': 0.3,
                        'cs': 0.15,
                    })
                    
                    while not es.stop() and state['calls'] < budget - 5 and time_ok():
                        solutions = es.ask()
                        fitnesses = []
                        for sol in solutions:
                            if state['calls'] >= budget - 5 or not time_ok():
                                break
                            v = call(sol)
                            fitnesses.append(v)
                        
                        if len(fitnesses) < len(solutions):
                            worst = max(fitnesses) if fitnesses else state['best_f']
                            fitnesses.extend([worst] * (len(solutions) - len(fitnesses)))
                        
                        es.tell(solutions, fitnesses)
                    
                    if es.result.xbest is not None:
                        xb = np.clip(np.asarray(es.result.xbest, dtype=float), -5, 5)
                        fb = call(xb)
                        if fb < state['best_f']:
                            state['best_f'] = fb
                            state['best_x'] = xb.copy()
                except Exception:
                    pass
            else:
                from scipy.optimize import minimize
                try:
                    res = minimize(
                        lambda x: call(x),
                        best.copy(),
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-13,
                                 'adaptive': True},
                    )
                    if np.isfinite(res.fun) and res.fun < state['best_f']:
                        state['best_f'] = float(res.fun)
                        state['best_x'] = np.clip(np.asarray(res.x, dtype=float), -5, 5)
                except Exception:
                    pass

        # ---------- 5. Local refinement with perturbation ----------
        if time_ok() and state['calls'] < budget - 5:
            for _ in range(15):
                if state['calls'] >= budget - 5 or not time_ok():
                    break
                perturb_scale = 0.005
                x_perturb = best + rng.normal(0, perturb_scale, dim)
                x_perturb = np.clip(x_perturb, -5, 5)
                v = call(x_perturb)
                if v < state['best_f']:
                    state['best_f'] = v
                    state['best_x'] = x_perturb.copy()
                    best = x_perturb.copy()

        if state['best_x'] is None:
            state['best_x'] = np.zeros(dim)
            state['best_f'] = call(state['best_x'])

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
