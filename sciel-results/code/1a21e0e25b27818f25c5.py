"""
Optimizer for Bent Cigar (f12) using Multi-Fidelity Surrogate-Assisted Search.
Combines Sobol global sampling with RBF interpolation and L-BFGS-B local refinement.
"""
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
import warnings

# Suppress warnings from RBF interpolation if the system is ill-conditioned
# This can happen if points are very close, but usually RBF handles it with epsilon.
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        calls_used = 0
        best_x = None
        best_fx = np.inf
        
        # Objective wrapper to count calls and track best
        def wrapped_obj(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")
            
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(-1)
            
            calls_used += 1
            
            # Evaluate
            fx = func(x)
            
            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        # Use a Sobol sequence for low-discrepancy sampling.
        # We need enough points to build a reasonable surrogate.
        # 50-100 points is usually good for 5D. Let's use 64.
        n_global = 64
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sobol.random(n_global)
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        # Evaluate all global samples
        X_obs = []
        Y_obs = []
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                fx = wrapped_obj(x)
                X_obs.append(x)
                Y_obs.append(fx)
            except Exception:
                break
        
        # If we didn't find any point (shouldn't happen), return center
        if not X_obs:
            x0 = np.zeros(dim)
            try:
                wrapped_obj(x0)
            except Exception:
                return best_x, best_fx
            X_obs = [x0]
            Y_obs = [best_fx]

        X_obs = np.array(X_obs)
        Y_obs = np.array(Y_obs)
        
        x0 = X_obs[np.argmin(Y_obs)].copy()
        
        # --- Phase 2: Surrogate-Assisted Refinement ---
        # We iteratively:
        # 1. Fit an RBF interpolant to the observed data.
        # 2. Use L-BFGS-B to minimize the surrogate.
        # 3. Evaluate the real function at the surrogate's minimum.
        # 4. Add the new point to the dataset.
        # 5. Repeat until budget is exhausted or convergence.
        
        # RBF parameters
        # 'multiquadric' is a common choice for smooth interpolation.
        # epsilon controls the shape. A larger epsilon makes it more local.
        # We start with a moderate value and might adjust if needed.
        rbf_eps = 1.0 
        
        # L-BFGS-B parameters
        # maxiter is the number of iterations for the surrogate optimization.
        # Since the surrogate is cheap, we can afford more iterations.
        lbgfs_maxiter = 50
        
        # Number of refinement iterations
        # Each iteration involves one real function call (the surrogate min).
        # We have ~500 - 64 = 436 calls left.
        # Let's allow up to 20 refinement steps.
        max_refine_steps = 20
        
        for step in range(max_refine_steps):
            if time.time() >= deadline or calls_used >= budget:
                break
            
            # Fit RBF to current observations
            # We need at least dim+1 points to fit. We have >= 64.
            try:
                # Use 'multiquadric' kernel
                rbf = RBFInterpolator(X_obs, Y_obs, kernel='multiquadric', epsilon=rbf_eps)
            except Exception:
                # If RBF fails (e.g., singular matrix), fall back to simple coordinate descent
                # or just break. Given the constraints, breaking is safer than crashing.
                # But let's try to increase epsilon if it fails.
                try:
                    rbf = RBFInterpolator(X_obs, Y_obs, kernel='multiquadric', epsilon=rbf_eps * 2)
                    rbf_eps *= 2
                except Exception:
                    break
            
            # Define the surrogate objective function for L-BFGS-B
            def surrogate_obj(x):
                try:
                    return rbf(x)
                except Exception:
                    return 1e10 # Large penalty if interpolation fails (shouldn't happen)
            
            # Define the surrogate gradient
            # RBFInterpolator in scipy does not provide a direct gradient method.
            # However, we can use finite differences for the gradient of the surrogate,
            # OR we can just use L-BFGS-B with numerical gradients.
            # Since the surrogate is cheap, numerical gradients are fine.
            # L-BFGS-B will estimate gradients.
            
            # Start L-BFGS-B from the current best point
            # We can also try starting from the global best of the surrogate
            # Let's just use the current best_x as the start point.
            # Actually, it's better to start from the point that minimizes the surrogate
            # within a small neighborhood, or just the current best.
            # Let's use the current best_x.
            
            x_start = x0.copy()
            
            # Bounds
            bounds = [(-5.0, 5.0)] * dim
            
            # Optimize the surrogate
            try:
                res = minimize(
                    surrogate_obj, 
                    x_start, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': lbgfs_maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                x_new = res.x
            except Exception:
                # If L-BFGS-B fails, break
                break
            
            # Evaluate the real function at the new candidate
            try:
                fx_new = wrapped_obj(x_new)
            except Exception:
                break
            
            # Update observations
            X_obs = np.vstack([X_obs, x_new])
            Y_obs = np.append(Y_obs, fx_new)
            
            # Update best
            if fx_new < best_fx:
                best_fx = fx_new
                best_x = x_new.copy()
                x0 = x_new.copy() # Move start point for next iteration
            else:
                # If no improvement, we might be stuck.
                # We can try to increase the epsilon of the RBF to make it more local
                # or just continue.
                # Let's adjust epsilon slightly to explore more locally.
                rbf_eps *= 1.5
                if rbf_eps > 10.0:
                    rbf_eps = 1.0 # Reset if it gets too large
            
            # Check for convergence
            if res.success and res.fun <= best_fx + 1e-8:
                # The surrogate minimum is close to the best real value found
                # We might have converged.
                # We can break early if we are confident, but let's continue a bit
                # to ensure we aren't in a local minimum of the surrogate.
                pass

        return best_x, best_fx
