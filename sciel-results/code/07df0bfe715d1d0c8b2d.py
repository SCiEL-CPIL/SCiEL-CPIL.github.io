import numpy as np
import scipy.optimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # Initial guess: Keich construction
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    # Precompute constants for area calculation
    a = (np.arange(n) + 1) / n
    M = 20000
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    
    def calc_area(x_vals):
        # x_vals: shape (n,)
        # L_i,m = x_i + a_i * y_m
        L = x_vals[:, None] + a[:, None] * y[None, :]
        L_sorted = np.sort(L, axis=0)
        
        # Gaps between consecutive sorted triangles
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        
        # Union length: w + sum(min(d, w))
        mins = np.minimum(d, w[None, :])
        union_lens = w + np.sum(mins, axis=0)
        
        area = np.mean(union_lens)
        return area
    
    start_time = time.time()
    timeout = 590  # Safety margin before 630s kill
    
    # Coordinate Descent with adaptive step size
    # We will perform sweeps until time limit is reached
    best_x = x.copy()
    best_area = calc_area(x)
    
    # Track step size for each variable
    step_sizes = np.ones(n) * 0.1
    min_step = 1e-8
    max_step = 1.0
    
    while time.time() - start_time < timeout:
        improved = False
        for i in range(n):
            if time.time() - start_time > timeout:
                break
            
            x_i_old = x[i]
            
            # Objective function for 1D optimization
            def obj_1d(xi):
                x_new = x.copy()
                x_new[i] = xi
                return calc_area(x_new)
            
            # Use a wider initial search range, then narrow
            # Try multiple step sizes to escape local minima
            current_step = step_sizes[i]
            
            # Try a few different step sizes
            for _ in range(3):
                res = scipy.optimize.minimize_scalar(
                    obj_1d, 
                    bounds=(x_i_old - current_step, x_i_old + current_step), 
                    method='bounded', 
                    options={'xatol': 1e-8}
                )
                
                if res.success:
                    x_new_i = res.x
                    x[i] = x_new_i
                    new_area = calc_area(x)
                    
                    if new_area < best_area:
                        best_area = new_area
                        best_x = x.copy()
                        improved = True
                        # Reduce step size for next time
                        step_sizes[i] = max(current_step * 0.5, min_step)
                        break
                    else:
                        # Revert and try smaller step
                        x[i] = x_i_old
                        current_step = max(current_step * 0.5, min_step)
                else:
                    # Try smaller step
                    current_step = max(current_step * 0.5, min_step)
            
            # If no improvement, keep current value
            if x[i] != x_i_old:
                # Check if this was an improvement
                final_area = calc_area(x)
                if final_area >= best_area:
                    x[i] = x_i_old
    
    # Return the best solution found
    return best_x
