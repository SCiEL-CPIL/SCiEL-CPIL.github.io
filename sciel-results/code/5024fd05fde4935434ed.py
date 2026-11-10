# EVOLVE-BLOCK-START
import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses a separable structure where x-positions depend only on i and y-positions 
    depend only on j. Optimizes these 16 parameters using Simulated Annealing 
    with a fast, accurate volume evaluation based on 1D union lengths.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin

    # 1. Initialize with Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    u_init = keich.copy()
    v_init = keich.copy()

    i_k = np.arange(64) // N
    j_k = np.arange(64) % N

    # 2. Fast volume evaluation function
    def union_length(intervals):
        """
        Compute the length of the union of 1D intervals.
        intervals: list of (start, end) tuples
        """
        if not intervals:
            return 0.0
        # Sort by start point
        intervals = sorted(intervals)
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                if e > merged[-1][1]:
                    merged[-1][1] = e
            else:
                merged.append([s, e])
        return sum(e - s for s, e in merged)

    def compute_volume(u, v, num_c=200):
        """
        Compute the volume of the union of tubes given 1D positions u (for x) and v (for y).
        u: array of shape (N,)
        v: array of shape (N,)
        """
        c_vals = np.linspace(0, 1, num_c)
        total_vol = 0.0
        
        # Precompute i/N and j/N factors
        i_factors = np.arange(N) / N
        j_factors = np.arange(N) / N
        width = 1.0 / N
        
        for c in c_vals:
            # x-intervals for each i
            # Interval i is [u[i] + c * i_factors[i], u[i] + c * i_factors[i] + width]
            x_starts = u + c * i_factors
            x_ends = x_starts + width
            x_len = union_length(list(zip(x_starts, x_ends)))
            
            # y-intervals for each j
            y_starts = v + c * j_factors
            y_ends = y_starts + width
            y_len = union_length(list(zip(y_starts, y_ends)))
            
            total_vol += x_len * y_len
            
        # Trapezoidal rule (or simple average if uniform spacing and we treat as Riemann sum)
        # Since c is uniform, integral is approx (1/num_c) * sum
        return total_vol / num_c

    # Initial cost
    best_params = np.concatenate([u_init, v_init])
    best_vol = compute_volume(u_init, v_init, 500) # Higher precision for initial
    
    # 3. Simulated Annealing
    rng = np.random.default_rng(42)
    T = 0.1
    cooling = 0.995
    min_T = 1e-6
    max_iter = 20000
    
    current_params = best_params.copy()
    current_vol = best_vol
    
    # We can also try multiple restarts if time permits
    # But first, let's run one long SA
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        if T < min_T:
            break
            
        # Perturb
        # Step size proportional to T
        step_size = T * 0.5
        delta = rng.normal(0, step_size, size=16)
        trial_params = current_params + delta
        
        u_trial = trial_params[:N]
        v_trial = trial_params[N:]
        
        # Evaluate
        trial_vol = compute_volume(u_trial, v_trial, 100) # Lower precision for speed during SA
        
        delta_vol = trial_vol - current_vol
        
        # Acceptance probability
        if delta_vol < 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_vol / T)
            
        if accept:
            current_params = trial_params
            current_vol = trial_vol
            if current_vol < best_vol:
                best_vol = current_vol
                best_params = current_params.copy()
                
        T *= cooling
        
    # 4. Refine with Nelder-Mead if time remains
    if time.time() < deadline - 60:
        from scipy.optimize import minimize
        
        def cost_func(params):
            u = params[:N]
            v = params[N:]
            return compute_volume(u, v, 500)
            
        # Run a few Nelder-Mead optimizations from the best point found by SA
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Add small noise to best_params for restart
            if restart == 0:
                x0 = best_params
            else:
                x0 = best_params + rng.normal(0, 0.01, size=16)
                
            try:
                res = minimize(cost_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_vol:
                    best_vol = res.fun
                    best_params = res.x
            except Exception:
                pass

    # 5. Final high-precision polish pass
    if time.time() < deadline - 30:
        from scipy.optimize import minimize
        
        def cost_func_high_res(params):
            u = params[:N]
            v = params[N:]
            return compute_volume(u, v, 2000)
            
        try:
            res = minimize(cost_func_high_res, best_params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except Exception:
            pass

    # 6. Construct final positions
    best_u = best_params[:N]
    best_v = best_params[N:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_u[i_flat]
    base_y = best_v[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
