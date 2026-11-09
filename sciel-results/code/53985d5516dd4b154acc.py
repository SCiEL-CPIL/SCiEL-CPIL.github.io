import numpy as np
import time
import math

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        
        # Phase 1: Initial random sampling
        n_init = min(30, self.budget // 5)
        rng = np.random.default_rng(0)
        
        for i in range(n_init):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES for global convergence
        if HAS_CMA:
            try:
                x0 = best_x.tolist()
                sigma0 = 0.5
                popsize = 4 + int(3 * math.log(self.dim))
                if popsize < 4:
                    popsize = 4
                
                remaining_budget = self.budget - eval_count
                # Use 70% of remaining for CMA, leave 30% for BO + local polish
                cma_budget = int(remaining_budget * 0.7)
                maxiter = max(1, (cma_budget - 1) // popsize)
                
                options = {
                    'verbose': -9,
                    'seed': 1,
                    'bounds': [-5.0] * self.dim + [5.0] * self.dim,
                    'CMA_boundaryhandler': 'reflect',
                    'popsize': popsize,
                    'maxiter': maxiter,
                    'stopitersize': 1e-10,
                    'stopconditionnumber': 1e12,
                    'stopfgap': 1e-12,
                }
                
                es = cma.CMA(x0, sigma0, options)
                
                while not es.stop() and eval_count < self.budget:
                    if time.time() > deadline - 10:
                        break
                    
                    xs = es.ask()
                    fs = []
                    for x in xs:
                        if eval_count >= self.budget:
                            break
                        if time.time() > deadline - 10:
                            break
                        fx = func(x)
                        eval_count += 1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        fs.append(fx)
                    
                    if len(fs) == len(xs):
                        es.tell(xs, fs)
                    else:
                        break
                        
            except Exception:
                pass

        # Phase 3: Bayesian Optimization for refinement
        if HAS_SKLEARN and eval_count < self.budget and time.time() < deadline - 10:
            try:
                # Collect all evaluated points so far (we need to store them)
                # Since we didn't store them in the previous phases, we'll start fresh
                # with the best point and do a few more evaluations to build the GP model
                
                # Use a small set of points around the best solution
                X_data = [best_x.copy()]
                y_data = [best_f]
                
                # Add a few more points around the best to build the model
                n_bo_init = min(20, (self.budget - eval_count) // 4)
                for i in range(n_bo_init):
                    if eval_count >= self.budget or time.time() > deadline - 10:
                        break
                    # Sample around best_x
                    x = best_x + 0.5 * rng.standard_normal(self.dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    eval_count += 1
                    X_data.append(x.copy())
                    y_data.append(fx)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                
                X_data = np.array(X_data)
                y_data = np.array(y_data)
                
                # Set up GP
                length_scale = 0.5
                kernel = Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(noise_level=1e-5)
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-10)
                gp.fit(X_data, y_data)
                
                # Acquire function: Expected Improvement
                def expected_improvement(X, y_best):
                    mu, sigma = gp.predict(X, return_std=True)
                    sigma = np.maximum(sigma, 1e-10)
                    ei = (y_best - mu) * norm_cdf((y_best - mu) / sigma) + sigma * norm_pdf((y_best - mu) / sigma)
                    return ei
                
                # BO loop
                bo_budget = min(100, self.budget - eval_count)
                for _ in range(bo_budget):
                    if eval_count >= self.budget or time.time() > deadline - 10:
                        break
                    
                    # Sample candidate points
                    n_candidates = 200
                    X_cand = rng.uniform(-5, 5, (n_candidates, self.dim))
                    
                    # Compute EI
                    y_best = min(y_data)
                    ei = expected_improvement(X_cand, y_best)
                    
                    # Select best candidate
                    idx = np.argmax(ei)
                    x_new = X_cand[idx]
                    
                    # Evaluate
                    fx_new = func(x_new)
                    eval_count += 1
                    X_data = np.vstack([X_data, x_new])
                    y_data = np.append(y_data, fx_new)
                    
                    if fx_new < best_f:
                        best_f = fx_new
                        best_x = x_new.copy()
                    
                    # Refit GP
                    gp.fit(X_data, y_data)
                    
            except Exception:
                pass

        # Phase 4: Local Polish: (1+1)-ES with adaptive step size
        remaining = self.budget - eval_count
        if remaining < 0:
            remaining = 0
            
        if remaining > 0 and time.time() < deadline - 5:
            rng_local = np.random.default_rng(42)
            x = best_x.copy()
            fx = best_f
            
            # Start with a moderate sigma
            sigma = 0.05
            no_improve = 0
            limit = 15
            
            while remaining > 0 and time.time() < deadline - 5:
                if no_improve > limit:
                    sigma = 0.1
                    no_improve = 0
                    
                y = np.clip(x + sigma * rng_local.standard_normal(self.dim), -5, 5)
                fy = func(y)
                remaining -= 1
                eval_count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.95
                    no_improve += 1
                    if sigma < 1e-5:
                        break
                        
        return best_x, best_f


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
