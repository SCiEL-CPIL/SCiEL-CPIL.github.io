"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
import random

class Optimizer:
    """Two-phase Surrogate-Assisted Optimizer for Rosenbrock (f8)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        def time_left():
            return deadline - time.time()
            
        def elapsed():
            return time.time() - start_time

        n = self.dim
        budget = self.budget
        
        # Phase 1: Surrogate-Assisted Exploration (First ~50 calls)
        # Goal: Find a point close to the valley floor to start local optimization.
        # Rosenbrock has a narrow valley. Random search or coarse grid might miss it.
        # A GP surrogate can model the landscape curvature and identify promising basins.
        
        X_train = []
        y_train = []
        
        # Initial random sampling
        rng = np.random.default_rng(42)
        
        # We will use a simple iterative GP acquisition strategy
        # Since implementing a full GP from scratch with hyperparameter optimization
        # might be slow or unstable in 5D with only 50 points, we use a simplified 
        # approach: 
        # 1. Sample a few random points.
        # 2. Fit a simple quadratic model or use a lightweight GP.
        # Given the constraint of "standard library, NumPy, SciPy", let's use 
        # a simple weighted local search or a very basic GP.
        
        # Actually, a robust alternative to a full GP with 50 points in 5D is 
        # a multi-start local search with a smart initialization.
        # However, the directive specifically asks for a GP surrogate strategy.
        
        # Let's implement a basic GP using RBF kernel.
        # 5D, ~50 points. Matrix inversion is 50x50, which is fast.
        
        def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
            # Compute squared distances
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return variance * np.exp(-sq_dists / (2 * length_scale**2))

        # Phase 1: Active Learning with GP
        # We use 50 evaluations for this phase.
        gp_budget = 50
        
        # Initial design: Space-filling (Sobol or Latin Hypercube) for better coverage
        # Simple LHS
        def latin_hypercube(n_points, dim, seed=42):
            rng_lh = np.random.default_rng(seed)
            points = np.zeros((n_points, dim))
            for d in range(dim):
                perm = rng_lh.permutation(n_points)
                points[:, d] = (perm + rng_lh.uniform(size=n_points)) / n_points
            # Scale to [-5, 5]
            points = points * 10.0 - 5.0
            return points

        # Use 10 initial points for GP, then 40 for acquisition
        n_init = 10
        X_init = latin_hypercube(n_init, n)
        
        # Evaluate initial points
        for x in X_init:
            if elapsed() > deadline - 10: break
            f_val = func(x)
            X_train.append(x)
            y_train.append(f_val)
            
        # If we ran out of time, return best so far
        if not X_train:
             # Fallback
             x0 = np.zeros(n)
             f0 = func(x0)
             return x0, f0

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # GP Hyperparameters: fixed length_scale and variance for simplicity and speed
        # We could optimize these, but with 50 points, fixed reasonable values often work
        # Rosenbrock valley width is roughly 0.1-1.0 depending on location.
        length_scale = 1.0
        noise_var = 1e-8
        signal_var = 1.0
        
        # Pre-compute kernel matrix for training data
        K_train = rbf_kernel(X_train, X_train, length_scale, signal_var) + noise_var * np.eye(len(X_train))
        # Add jitter for numerical stability
        K_train += 1e-6 * np.eye(len(X_train))
        
        try:
            L_train = np.linalg.cholesky(K_train)
            alpha = np.linalg.solve(L_train.T, np.linalg.solve(L_train, y_train))
        except np.linalg.LinAlgError:
            # Fallback if GP fails
            best_idx = np.argmin(y_train)
            x_start = X_train[best_idx]
            f_start = y_train[best_idx]
            # Jump to Phase 2 immediately
            pass
        else:
            # Acquisition: Expected Improvement (EI) or Posterior Mean?
            # Directive says: "exploit its posterior mean to identify the most promising basin center"
            # Posterior mean minimization is a valid strategy (PI).
            # We will sample a grid of candidate points and pick the one with the lowest GP mean.
            
            # Generate candidate points: A fine grid around the current best and some random points
            # Since we only have 40 calls left in Phase 1, we can't evaluate a huge grid.
            # Instead, we can optimize the GP mean directly using a few local optimizations of the surrogate.
            
            # Let's use a multi-start local optimization on the GP mean.
            # Start from current best and a few random points.
            
            def gp_mean(x_pred):
                x_pred = x_pred.reshape(1, -1)
                K_s = rbf_kernel(x_pred, X_train, length_scale, signal_var)
                mu = K_s @ alpha
                return mu[0]
                
            # We will use scipy.optimize.minimize on gp_mean
            # This does not consume function evaluations of the true func!
            
            candidates = [X_train[np.argmin(y_train)]]
            # Add a few random points
            for _ in range(5):
                candidates.append(rng.uniform(-4, 4, size=n))
                
            best_gp_x = None
            best_gp_val = float('inf')
            
            for c in candidates:
                try:
                    res = minimize(gp_mean, c, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
                    if res.fun < best_gp_val:
                        best_gp_val = res.fun
                        best_gp_x = res.x
                except Exception:
                    continue
            
            # Evaluate the best GP candidate
            if best_gp_x is not None:
                x_eval = np.clip(best_gp_x, -5, 5)
                f_eval = func(x_eval)
                X_train = np.vstack([X_train, x_eval])
                y_train = np.append(y_train, f_eval)
                
                # Update GP
                K_train = rbf_kernel(X_train, X_train, length_scale, signal_var) + noise_var * np.eye(len(X_train))
                K_train += 1e-6 * np.eye(len(X_train))
                try:
                    L_train = np.linalg.cholesky(K_train)
                    alpha = np.linalg.solve(L_train.T, np.linalg.solve(L_train, y_train))
                except:
                    pass

            # Repeat this process for the remaining Phase 1 budget?
            # The directive says "use the first 50 evaluations".
            # We used 10 + 1 = 11. We have ~39 left.
            # Let's do a few more rounds of "Optimize GP Mean -> Evaluate -> Update GP"
            
            rounds = 5
            for _ in range(rounds):
                if elapsed() > deadline - 100: break # Save time for Phase 2
                
                # Re-optimize GP mean from current best and some randoms
                current_best_idx = np.argmin(y_train)
                candidates = [X_train[current_best_idx]]
                for _ in range(3):
                    candidates.append(rng.uniform(-4, 4, size=n))
                
                best_gp_x = None
                best_gp_val = float('inf')
                
                for c in candidates:
                    try:
                        res = minimize(gp_mean, c, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-5})
                        if res.fun < best_gp_val:
                            best_gp_val = res.fun
                            best_gp_x = res.x
                    except:
                        continue
                        
                if best_gp_x is not None:
                    x_eval = np.clip(best_gp_x, -5, 5)
                    f_eval = func(x_eval)
                    X_train = np.vstack([X_train, x_eval])
                    y_train = np.append(y_train, f_eval)
                    
                    # Update GP
                    K_train = rbf_kernel(X_train, X_train, length_scale, signal_var) + noise_var * np.eye(len(X_train))
                    K_train += 1e-6 * np.eye(len(X_train))
                    try:
                        L_train = np.linalg.cholesky(K_train)
                        alpha = np.linalg.solve(L_train.T, np.linalg.solve(L_train, y_train))
                    except:
                        pass

        # End Phase 1
        # Identify the best point found so far as the starting point for Phase 2
        best_idx = np.argmin(y_train)
        x_start = X_train[best_idx]
        f_start = y_train[best_idx]
        
        # Phase 2: Local Refinement with L-BFGS-B
        # The GP identified a basin center. Now use a gradient-based method (finite diff)
        # to exploit the local structure efficiently.
        
        # Define the objective for L-BFGS-B
        def objective(x):
            if elapsed() > deadline - 5:
                return float('inf')
            x = np.clip(x, -5, 5)
            return func(x)
            
        # Run L-BFGS-B
        # Bounds
        bounds = [(-5, 5)] * n
        
        # We can run multiple restarts if budget allows, but typically one good start is enough
        # given the GP pre-selection.
        
        best_x_final = x_start.copy()
        best_f_final = f_start
        
        # Try L-BFGS-B from the GP-selected start
        try:
            res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_f_final:
                best_f_final = res.fun
                best_x_final = res.x
        except Exception:
            pass
            
        # If we have significant time left, try a few more local optimizations from slightly perturbed points
        # to ensure we didn't miss a better local minimum nearby.
        if elapsed() < deadline - 60:
            # Perturb the best point
            for _ in range(5):
                if elapsed() > deadline - 10: break
                perturbation = rng.normal(0, 0.5, size=n)
                x_trial = np.clip(best_x_final + perturbation, -5, 5)
                try:
                    res = minimize(objective, x_trial, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 500, 'ftol': 1e-10})
                    if res.fun < best_f_final:
                        best_f_final = res.fun
                        best_x_final = res.x
                except:
                    continue
                    
        return best_x_final, best_f_final
# EVOLVE-BLOCK-END
