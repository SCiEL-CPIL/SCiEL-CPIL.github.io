import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # 1. Initial random sampling to build the initial model
        n_init = 20
        if budget < n_init:
            n_init = budget
            
        X_init = np.random.uniform(-5, 5, size=(n_init, dim))
        Y_init = np.array([func(x) for x in X_init])
        
        best_x = X_init[np.argmin(Y_init)]
        best_f = np.min(Y_init)
        
        # Check if we have enough budget for BO
        if budget - n_init < 10:
            # Not enough budget for BO, just do a local polish on the best point
            if time.time() < deadline:
                try:
                    res = minimize(func, best_x, method='Nelder-Mead', 
                                   options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-12})
                    if res.fun < best_f:
                        best_x = res.x
                        best_f = res.fun
                except Exception:
                    pass
            return best_x, best_f

        # 2. Bayesian Optimization Loop
        # Kernel: Matern is good for smooth functions like Sphere
        # Length scales are initialized to 1.0
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)
        
        X = X_init
        Y = Y_init
        
        calls_made = n_init
        no_improve_count = 0
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Fit the GP
            try:
                gp.fit(X, Y)
            except Exception:
                # If GP fitting fails (e.g., singular matrix), fall back to local search
                break
                
            # Acquire Function: Expected Improvement (EI)
            # We minimize, so we look for low values
            y_min = np.min(Y)
            
            def acquisition_function(x_flat):
                x = x_flat.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                sigma = np.maximum(sigma, 1e-10)
                improvement = y_min - mu - 0.01 # small exploration term
                z = improvement / sigma
                ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei # We maximize EI, so minimize -EI
            
            # Optimize the acquisition function using multiple starts
            best_x_acq = None
            best_ei = -np.inf
            n_starts = 10
            
            # Start from current best and some random points
            starts = [best_x]
            starts += list(np.random.uniform(-5, 5, size=(n_starts-1, dim)))
            
            for x0 in starts:
                try:
                    res = minimize(acquisition_function, x0, method='L-BFGS-B', 
                                   bounds=[(-5, 5)] * dim, options={'maxiter': 50})
                    if res.fun < best_ei:
                        best_ei = res.fun
                        best_x_acq = res.x
                except Exception:
                    continue
            
            if best_x_acq is None:
                # Fallback to random if acquisition optimization fails
                best_x_acq = np.random.uniform(-5, 5, dim)
            
            # Evaluate the new point
            new_f = func(best_x_acq)
            calls_made += 1
            
            # Update best
            if new_f < best_f:
                best_f = new_f
                best_x = best_x_acq
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update GP data
            X = np.vstack([X, best_x_acq])
            Y = np.append(Y, new_f)
            
            # If we have a very good result, switch to local polishing
            if best_f < 1e-4 or no_improve_count > 15:
                # Use remaining budget for local refinement
                if time.time() < deadline:
                    try:
                        res = minimize(func, best_x, method='Powell', 
                                       options={'maxfev': budget - calls_made, 'xtol': 1e-12, 'ftol': 1e-12})
                        if res.fun < best_f:
                            best_x = res.x
                            best_f = res.fun
                    except Exception:
                        pass
                break

        return best_x, best_f
