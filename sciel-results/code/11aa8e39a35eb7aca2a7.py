import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import solve

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        best_x = None
        best_f = float('inf')
        eval_count = 0
        
        # History for surrogate
        X_hist = []
        F_hist = []
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        def halton_sequence(count, dim):
            # Generate Halton sequence in [0, 1]^dim
            # Then scale to [-5, 5]^dim
            points = np.zeros((count, dim))
            for i in range(count):
                for j in range(dim):
                    # Prime numbers for dimensions
                    # 2, 3, 5, 7, 11
                    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                    p = primes[j]
                    f = 0.0
                    inv_p = 1.0 / p
                    i_val = i + 1
                    while i_val > 0:
                        i_val, remainder = divmod(i_val, p)
                        f += inv_p * remainder
                        inv_p /= p
                    points[i, j] = f
            # Scale to [-5, 5]
            points = points * 10.0 - 5.0
            return points

        # Phase 1: Halton Initialization
        n_init = min(50, budget)
        init_points = halton_sequence(n_init, n)
        
        for x in init_points:
            if eval_count >= budget or time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            update_best(x, f)
            X_hist.append(x.copy())
            F_hist.append(f)
            
        if eval_count >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: RBF Surrogate Guided Search
        # We will use the best K points to build the surrogate to avoid noise from poor points
        # Or just use all points if count is small.
        
        while eval_count < budget and time.time() < deadline:
            # Refit RBF periodically or every step?
            # Every step is expensive if we solve linear system. 
            # Let's do it every 5 steps or if we have new points.
            
            if len(X_hist) >= 10:
                # Select subset for RBF if too many points to keep it fast and stable
                # Use the best 30 points
                if len(X_hist) > 30:
                    idx = np.argsort(F_hist)[:30]
                    X_sub = np.array(X_hist[idx])
                    F_sub = np.array(F_hist[idx])
                else:
                    X_sub = np.array(X_hist)
                    F_sub = np.array(F_hist)
                
                N = len(X_sub)
                d = self.dim
                
                # Build RBF matrix
                # Kernel: Gaussian with fixed width or adaptive?
                # Fixed width is simpler. Let's estimate width from data spread.
                # Or use a small constant.
                # Let's use a small sigma to make it local, or large to be smooth.
                # For global optimization, a moderate sigma is good.
                
                # Calculate pairwise distances
                # Efficient way:
                # D2 = sum(x^2) + sum(y^2) - 2xy
                X_sub_sq = np.sum(X_sub**2, axis=1)
                D2 = X_sub_sq[:, np.newaxis] + X_sub_sq[np.newaxis, :] - 2 * X_sub @ X_sub.T
                D2 = np.maximum(D2, 0) # Numerical stability
                
                # Choose sigma
                # Median distance heuristic
                dists = np.sqrt(D2)
                np.fill_diagonal(dists, np.inf)
                median_dist = np.median(dists)
                if median_dist < 1e-6:
                    median_dist = 1.0
                sigma = median_dist / 2.0
                
                K = np.exp(-D2 / (2 * sigma**2))
                
                # Add ridge for stability
                K += 1e-8 * np.eye(N)
                
                # Solve for weights
                # We want K w + c * 1 = F
                # This is a system of N+1 equations.
                # Or we can center the data.
                # Simple approach: Add a constant term by appending a column of 1s to K and a row of 1s to the system?
                # Standard RBF interpolation:
                # [ K  1 ] [ w ]   [ F ]
                # [ 1' 0 ] [ c ] = [ 0 ]
                
                A = np.zeros((N+1, N+1))
                A[:N, :N] = K
                A[:N, N] = 1.0
                A[N, :N] = 1.0
                b = np.zeros(N+1)
                b[:N] = F_sub
                
                try:
                    w_c = solve(A, b)
                    w = w_c[:N]
                    c = w_c[N]
                    
                    # Define surrogate function
                    def surrogate(x_new):
                        x_new = np.clip(x_new, -5, 5)
                        # Calculate distances from x_new to X_sub
                        # d2 = sum((x_new - x_i)^2)
                        diff = x_new - X_sub
                        d2 = np.sum(diff**2, axis=1)
                        val = np.sum(w * np.exp(-d2 / (2 * sigma**2))) + c
                        return val
                        
                    # Minimize surrogate
                    # Use multiple starts to avoid local minima of the surrogate
                    best_surrogate_x = None
                    best_surrogate_f = float('inf')
                    
                    # Start from current best
                    starts = [best_x]
                    # Add a few random starts
                    rng = np.random.default_rng(42)
                    for _ in range(3):
                        starts.append(rng.uniform(-5, 5, n))
                        
                    for x0 in starts:
                        res = minimize(surrogate, x0, method='L-BFGS-B', bounds=[(-5, 5)]*n)
                        if res.success and res.fun < best_surrogate_f:
                            best_surrogate_f = res.fun
                            best_surrogate_x = res.x
                            
                    if best_surrogate_x is not None:
                        x_candidate = np.clip(best_surrogate_x, -5, 5)
                        f_candidate = func(x_candidate)
                        eval_count += 1
                        update_best(x_candidate, f_candidate)
                        X_hist.append(x_candidate.copy())
                        F_hist.append(f_candidate)
                    else:
                        # Fallback: random step
                        x_candidate = np.clip(best_x + 0.5 * rng.standard_normal(n), -5, 5)
                        f_candidate = func(x_candidate)
                        eval_count += 1
                        update_best(x_candidate, f_candidate)
                        X_hist.append(x_candidate.copy())
                        F_hist.append(f_candidate)
                        
                except Exception:
                    # If RBF fails, fallback to random
                    x_candidate = np.clip(best_x + 0.5 * np.random.default_rng(42).standard_normal(n), -5, 5)
                    f_candidate = func(x_candidate)
                    eval_count += 1
                    update_best(x_candidate, f_candidate)
                    X_hist.append(x_candidate.copy())
                    F_hist.append(f_candidate)
            else:
                # Not enough points for RBF, do random exploration
                x_candidate = np.clip(best_x + 1.0 * np.random.default_rng(42).standard_normal(n), -5, 5)
                f_candidate = func(x_candidate)
                eval_count += 1
                update_best(x_candidate, f_candidate)
                X_hist.append(x_candidate.copy())
                F_hist.append(f_candidate)

        return best_x, best_f
