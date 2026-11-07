"""Direct-structure optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses the known structure of the function: f(x) = (Griewank(x) - 1) * (Rosenbrock(x) + 450).
Since Rosenbrock is large and positive, minimizing f is dominated by minimizing Rosenbrock,
which has a known global minimum at x* = 1 in each dimension. We exploit this structure
directly with a CMA-ES + Nelder-Mead hybrid. The hidden shift/rotation means the true
minimizer is somewhere in [-4,4]^5, but the landscape is still shaped by the
Griewank-Rosenbrock interaction.

Strategy:
1. Focused multi-start around the Rosenbrock minimum region (x=1)
2. CMA-ES for robust local optimization handling correlated dimensions
3. Nelder-Mead for fast convergence in smooth basins
4. L-BFGS-B final polish for high precision
"""
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        dim = self.dim
        budget = self.budget
        
        # Evaluate a candidate and track best
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def eval_point(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Initialize with diverse points informed by the structure
        # The Rosenbrock part wants x=1, Griewank wants x=0
        # The product structure means the minimum is a compromise
        rng = np.random.default_rng(42)
        
        # Structured initial points
        init_points = []
        # Point near the known Rosenbrock minimum
        init_points.append(np.ones(dim))
        # Point at origin (Griewank minimum)
        init_points.append(np.zeros(dim))
        # Points in between
        for t in [0.3, 0.5, 0.7, 0.9]:
            init_points.append(t * np.ones(dim))
        # Random points in a focused region
        for _ in range(8):
            init_points.append(rng.uniform(-2, 3, dim))
        # A few fully random
        for _ in range(4):
            init_points.append(rng.uniform(-5, 5, dim))
        
        # Evaluate initial points
        for p in init_points:
            if call_count >= budget or time.time() >= deadline:
                break
            eval_point(p)
        
        # If we've used too many calls, return best
        if call_count >= budget or time.time() >= deadline:
            return best_x
        
        # CMA-ES optimization from multiple starting points
        # Focus on the best few starting points
        
        def cmaes(x0, max_evals):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() >= deadline:
                return
            
            # CMA-ES parameters
            n = dim
            lambda_ = 4 + int(3 * np.log(n))  # population size
            sigma = 0.5  # initial step size
            C = np.eye(n)  # covariance matrix
            ps = np.zeros(n)  # evolution path
            cs = 1.0 / (n + 4)  # path length adaptation
            c1 = 2 / ((n + 1)**2 + 10)  # rank-one update
            cmu = min(0.5 - c1, 2 / ((n + 2)**2 - 2))  # rank-mu update
            mu = lambda_ // 2  # number of recombination weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)  # effective population size
            
            # Initial population
            pop = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                x = x0 + sigma * (C @ z)
                pop.append((x, 0.0))
            
            # Evaluate initial population
            for i in range(lambda_):
                if call_count >= budget or time.time() >= deadline:
                    return
                pop[i] = (pop[i][0], eval_point(pop[i][0]))
            
            # Sort by fitness
            pop.sort(key=lambda x: x[1])
            
            # CMA-ES iterations
            for it in range(max_evals // lambda_ + 1):
                if call_count >= budget or time.time() >= deadline:
                    return
                
                # Mean update
                m = np.sum([weights[i] * pop[i][0] for i in range(mu)], axis=0)
                
                # Update evolution path
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C @ (m - x0))
                hsig = np.sqrt((1 - (1 - cs)**20) / (1 - (1 - cs)**2))
                h = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**it) < (1.4 + 2/(n+1)) * hsig else 0.0
                
                # Update covariance
                if h:
                    C = (1 - c1 - cmu) * C + c1 * np.outer(m - x0, m - x0)
                else:
                    C = (1 - c1 - cmu) * C
                for i in range(mu):
                    C += cmu * weights[i] * np.outer(pop[i][0] - x0, pop[i][0] - x0)
                
                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Step size update
                chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
                sigma *= np.exp(cs / (n + 3) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Generate new population
                new_pop = []
                for i in range(lambda_):
                    if call_count >= budget or time.time() >= deadline:
                        return
                    z = rng.standard_normal(n)
                    x = m + sigma * (C @ z)
                    f = eval_point(x)
                    new_pop.append((x, f))
                
                # Sort and update
                new_pop.sort(key=lambda x: x[1])
                pop = new_pop
                x0 = m.copy()
        
        # Run CMA-ES from best start first
        if call_count < budget and time.time() < deadline:
            cmaes(best_x.copy(), 100)
        
        # Run CMA-ES from other starts if we have budget
        nm_starts = []
        # Add a few nearby perturbations
        for _ in range(2):
            nm_starts.append(np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5))
        
        for start in nm_starts:
            if call_count >= budget or time.time() >= deadline:
                break
            cmaes(start, 50)
        
        # Nelder-Mead optimization from multiple starting points
        
        def bounded_nm(x0, max_evals):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() >= deadline:
                return
            
            # Create a penalty-augmented function for bounds
            def penalized(x):
                if call_count >= budget or time.time() >= deadline:
                    return np.inf
                x = np.clip(x, -5, 5)
                f = float(func(x))
                call_count += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                # Add penalty for being out of bounds (shouldn't happen due to clip)
                return f
            
            try:
                res = minimize(
                    penalized,
                    np.clip(x0, -5, 5),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 0.01, 'fatol': 1e-6},
                )
                if res.fun < best_f:
                    eval_point(res.x)
            except Exception:
                pass
        
        # Run NM from best start first
        if call_count < budget and time.time() < deadline:
            bounded_nm(best_x.copy(), 50)
        
        # Run NM from other starts if we have budget
        for start in nm_starts:
            if call_count >= budget or time.time() >= deadline:
                break
            bounded_nm(start, 40)
        
        # If we still have budget, do a fine-grained coordinate search
        if call_count < budget and time.time() < deadline:
            # Coordinate descent with adaptive step size
            x = best_x.copy()
            f = best_f
            step = 0.5
            no_improve = 0
            
            for coord in range(dim * 10):  # Multiple passes
                if call_count >= budget or time.time() >= deadline:
                    break
                c = coord % dim
                # Try positive and negative steps
                for direction in [1, -1]:
                    if call_count >= budget or time.time() >= deadline:
                        break
                    x_trial = x.copy()
                    x_trial[c] += direction * step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = float(func(x_trial))
                    call_count += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()
                    if f_trial < f:
                        x = x_trial.copy()
                        f = f_trial
                        no_improve = 0
                        step = min(step * 1.5, 2.0)
                    else:
                        no_improve += 1
                
                if no_improve > 4:
                    step *= 0.7
                    no_improve = 0
                    if step < 0.05:
                        break
        
        # Final check
        if call_count < budget and time.time() < deadline:
            # Try a few random perturbations around best
            for _ in range(5):
                if call_count >= budget or time.time() >= deadline:
                    break
                x_try = np.clip(best_x + 0.3 * rng.standard_normal(dim), -5, 5)
                eval_point(x_try)
        
        # High-precision polish pass with L-BFGS-B
        if call_count < budget and time.time() < deadline:
            def polish_func(x):
                if call_count >= budget or time.time() >= deadline:
                    return best_f
                f = float(func(x))
                call_count += 1
                if f < best_f:
                    best_f = f
                    best_x = np.clip(x, -5, 5).copy()
                return f
            
            try:
                res = minimize(
                    polish_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                if res.fun < best_f:
                    eval_point(res.x)
            except Exception:
                pass
        
        return best_x
