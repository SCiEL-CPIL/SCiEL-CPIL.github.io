import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # 1. Generate Keich baseline
    # a_i = (i+1)/n
    # x_i = sum_{j=1}^k ((1-j)/k) * eps_j * 2^{-j}
    # where eps_j is the j-th bit of i (0-indexed from LSB? No, bin(i) is MSB first)
    # Let's stick to the provided seed logic which worked.
    j_idx = np.arange(1, k + 1)
    positions = np.zeros(n)
    for i in range(n):
        # Get binary representation of i, padded to k bits
        bin_str = bin(i)[2:]
        bin_str = bin_str.zfill(k)
        eps = np.array([int(c) for c in bin_str], dtype=float)
        # The seed code used:
        # eps = np.array(list(str(bin(i))[2:]), dtype=int)
        # eps = np.pad(eps, (k - len(eps), 0), "constant")
        # positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        # Note: j in seed was 1..k. eps was aligned such that first bit of bin(i) corresponds to j=1?
        # bin(1) = '1', pad to '00000001'. j=1..7.
        # If i=1, eps=[0,0,0,0,0,0,1].
        # Term for j=7: (1-7)/7 * 1 * 2^-7 / 7? No, the formula in seed:
        # np.sum((1 - j) * eps * (2.0 ** -j) / k)
        # Let's just replicate the seed exactly to be safe.
        eps_seed = np.array(list(str(bin(i))[2:]), dtype=int)
        eps_seed = np.pad(eps_seed, (k - len(eps_seed), 0), "constant")
        positions[i] = np.sum((1 - j_idx) * eps_seed * (2.0 ** -j_idx) / k)
    
    x0 = positions
    
    # 2. Define objective function
    # Area = mean of union lengths at y_m = (m+0.5)/20000
    M = 20000
    y_vals = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n # shape (n,)
    
    def objective(x):
        # x: shape (n,)
        # L_i(y) = x_i + a_i * y
        # L matrix: shape (n, M)
        L = x[:, np.newaxis] + a[:, np.newaxis] * y_vals[np.newaxis, :]
        
        # w(y) = (1-y)/64
        w = (1.0 - y_vals) / n # shape (M,)
        
        # For each y, sort L and compute union length
        # L_sorted: shape (n, M)
        L_sorted = np.sort(L, axis=0)
        
        # Union length = w + sum_{i=0}^{n-2} min(L_sorted[i+1] - L_sorted[i], w)
        diffs = L_sorted[1:, :] - L_sorted[:-1, :] # shape (n-1, M)
        overlaps = np.minimum(diffs, w[np.newaxis, :]) # shape (n-1, M)
        
        union_lengths = w[np.newaxis, :] + np.sum(overlaps, axis=0) # shape (M,)
        
        area = np.mean(union_lengths)
        return area
    
    # 3. Optimization
    start_time = time.time()
    deadline = 600.0 # Safety margin for 630s limit
    
    # Nelder-Mead is good for this non-smooth problem
    # We minimize the area.
    # options: maxiter, xatol, fatol
    # Since we have 64 dimensions, NM might be slow if not careful.
    # However, the landscape is structured.
    
    best_x = x0.copy()
    best_area = objective(x0)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        if res.fun < best_area:
            best_x = res.x
            best_area = res.fun
            
    except Exception as e:
        # Fallback to coordinate descent if NM fails or times out
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return best_x
    
    # Optional: Coordinate descent refinement if time permits
    # The prompt mentions coordinate descent improved the score.
    # Let's do a few rounds of coordinate descent to polish the result.
    
    def coordinate_descent_step(x, max_steps=1000):
        # Try to improve each coordinate using a small 1D search
        # We can use scipy.optimize.minimize_scalar for each coordinate
        # Or a simple grid search.
        # Given 64 dimensions, 1000 steps is 64000 evaluations.
        # Each evaluation is ~20000 * 64 ops. ~1.2M ops. Very fast in NumPy.
        
        current_area = objective(x)
        improved = True
        step_size = 0.01
        
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                # Define 1D function for x_i
                def f_1d(xi):
                    x_temp = x.copy()
                    x_temp[i] = xi
                    return objective(x_temp)
                
                # Search in a window around current x_i
                # The optimal x_i is likely within [0, 1]
                # But let's search a wider range to be safe, e.g., [x_i - 0.5, x_i + 0.5]
                # Actually, the triangles are in [0,1]x[0,1] roughly?
                # The problem doesn't strictly constrain x_i to [0,1], but the triangles
                # should probably stay in the unit square or at least not go too far.
                # The metric is just area of union.
                # Let's search [0, 1] for simplicity, as Keich values are in [0,1].
                
                res_1d = minimize_scalar(f_1d, bounds=(max(0, x[i]-0.5), min(1, x[i]+0.5)), method='bounded', options={'maxiter': 50})
                
                if res_1d.fun < current_area - 1e-10:
                    x[i] = res_1d.x
                    current_area = res_1d.fun
                    improved = True
                    
        return x, current_area
    
    # Run coordinate descent for a few iterations
    # We can afford this because the objective is fast.
    try:
        best_x, best_area = coordinate_descent_step(best_x, max_steps=200)
    except Exception:
        pass
    
    # Final check
    if time.time() > deadline - 1:
        return best_x

    # Ensure finite floats
    if not np.all(np.isfinite(best_x)):
        return x0
        
    return best_x

# EVOLVE-BLOCK-END
