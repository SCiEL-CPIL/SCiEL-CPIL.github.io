import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 600.0  # Safety margin
    
    # 1. Initial Guess: Keich Binary Construction
    # This is a known good starting point for Kakeya-type problems.
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    best_X = keich.copy()
    best_Y = keich.copy()
    
    # Precompute indices for the 64 tubes
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    # 2. Exact Volume Calculator for Separable Case
    # The union of tubes is separable if x_k depends only on i and y_k only on j.
    # Volume = Integral_0^1 (UnionLen_X(c) * UnionLen_Y(c)) dc
    
    def _union_length_1d(starts, widths, c_val, i_indices):
        """
        Compute the length of the union of intervals:
        [starts[i] + c_val * i_indices[i] / N, starts[i] + c_val * i_indices[i] / N + widths[i]]
        for i in 0..N-1.
        """
        # starts is shape (N,)
        # i_indices is shape (N,)
        # widths is scalar (1/N)
        
        # Calculate actual start and end points for this c
        s = starts + c_val * i_indices / N
        e = s + widths
        
        # Sort intervals by start point
        order = np.argsort(s)
        s_sorted = s[order]
        e_sorted = e[order]
        
        # Merge intervals to find total length
        if len(s_sorted) == 0:
            return 0.0
            
        total_len = 0.0
        current_start = s_sorted[0]
        current_end = e_sorted[0]
        
        for k in range(1, len(s_sorted)):
            if s_sorted[k] > current_end:
                # Disjoint interval
                total_len += current_end - current_start
                current_start = s_sorted[k]
                current_end = e_sorted[k]
            else:
                # Overlapping interval
                if e_sorted[k] > current_end:
                    current_end = e_sorted[k]
                    
        total_len += current_end - current_start
        return total_len

    def compute_exact_volume(X, Y, num_c_samples=200):
        """
        Compute the exact union volume assuming separability.
        X: array of shape (8,) for x-base positions of rows i
        Y: array of shape (8,) for y-base positions of cols j
        """
        # We use fixed quadrature points for c in [0, 1]
        # Using linspace is sufficient for this smooth-ish function
        c_vals = np.linspace(0, 1, num_c_samples)
        
        # Precompute i_indices and j_indices
        i_idx = np.arange(N, dtype=np.float64)
        j_idx = np.arange(N, dtype=np.float64)
        
        width = 1.0 / N
        
        total_vol = 0.0
        dc = 1.0 / (num_c_samples - 1)
        
        # Optimization: Vectorize the c-loop if possible, or keep it simple.
        # N=8 is small, so 200 iterations of a small sort is very fast.
        for c in c_vals:
            Lx = _union_length_1d(X, width, c, i_idx)
            Ly = _union_length_1d(Y, width, c, j_idx)
            total_vol += Lx * Ly
            
        # Trapezoidal rule approximation (or simple Riemann sum)
        # Since we use linspace including endpoints, trapezoidal is better
        # But for 200 points, simple mean * 1.0 is close.
        # Let's use trapezoidal weights for accuracy
        if num_c_samples > 1:
            weights = np.ones(num_c_samples)
            weights[0] = 0.5
            weights[-1] = 0.5
            total_vol = np.sum(total_vol * weights) * (1.0 / (num_c_samples - 1))
        else:
            total_vol = 0.0
            
        return total_vol

    # 3. Optimization
    def objective(params):
        if time.time() > deadline - 10:
            return 1e18
        X = params[:N]
        Y = params[N:]
        # Use a moderate number of samples for speed during optimization
        # 100 samples is likely sufficient for Nelder-Mead to find the basin
        return compute_exact_volume(X, Y, num_c_samples=100)
    
    initial_params = np.concatenate([best_X, best_Y])
    
    best_params = initial_params.copy()
    best_score = objective(initial_params)
    
    rng_restart = np.random.default_rng(12345)
    
    # Multi-restart Nelder-Mead
    # We have ~600s. Each Nelder-Mead run with 100 samples should be fast (<1s per 1000 iters).
    # We can afford many restarts.
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        if restart == 0:
            start_params = initial_params.copy()
        else:
            # Add noise to escape local minima
            # Scale noise based on current best score? Or fixed?
            # The landscape is likely not too rugged.
            noise_scale = 0.5
            noise = rng_restart.normal(0, noise_scale, size=2*N)
            start_params = best_params + noise
            
        try:
            res = minimize(
                objective, 
                start_params, 
                method='Nelder-Mead',
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10,
                    'adaptive': True,
                    'disp': False
                }
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception:
            pass

    # Final refinement with higher precision volume calculation
    if time.time() < deadline - 5:
        final_X = best_params[:N]
        final_Y = best_params[N:]
        
        # One more polish with higher resolution quadrature
        # This ensures we are at the true minimum of the continuous objective
        # although the evaluator uses MC, minimizing the exact volume is the best proxy.
        
        def final_objective(params):
            X = params[:N]
            Y = params[N:]
            return compute_exact_volume(X, Y, num_c_samples=500)
            
        res_final = minimize(
            final_objective,
            best_params,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        if res_final.fun < best_score:
            best_params = res_final.x.copy()

    # 4. Construct Output
    X_opt = best_params[:N]
    Y_opt = best_params[N:]
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
