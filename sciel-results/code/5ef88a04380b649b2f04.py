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
    
    # Precompute constants
    a = (np.arange(n) + 1) / n
    M = 20000
    y = (np.arange(M) + 0.5) / M
    A = a[:, None] * y[None, :]
    w = (1.0 - y) / n
    
    def calc_area(x_vals):
        L = x_vals[:, None] + A
        L_sorted = np.sort(L, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        mins = np.minimum(d, w[None, :])
        union_lens = w + np.sum(mins, axis=0)
        return np.mean(union_lens)
    
    start_time = time.time()
    timeout = 595  # Safety margin
    
    best_x = x.copy()
    best_area = calc_area(x)
    
    # Coordinate descent with tight tolerance
    sweep = 0
    while time.time() - start_time < timeout - 60:
        improved = False
        for i in range(n):
            if time.time() - start_time > timeout - 60:
                break
            
            x_i_old = x[i]
            
            def obj_1d(xi):
                x_new = x.copy()
                x_new[i] = xi
                return calc_area(x_new)
            
            # Try a few candidate points
            candidates = [x_i_old]
            # Golden section search bounds
            for offset in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1, -0.2, 0.2]:
                candidates.append(x_i_old + offset)
            
            best_candidate = min(candidates, key=obj_1d)
            
            # Fine-tune around best candidate
            res = scipy.optimize.minimize_scalar(
                obj_1d,
                bounds=(best_candidate - 0.01, best_candidate + 0.01),
                method='bounded',
                options={'xatol': 1e-8, 'maxiter': 50}
            )
            
            if res.success and res.fun < best_area:
                x[i] = res.x
                improved = True
        
        current_area = calc_area(x)
        if current_area < best_area:
            best_area = current_area
            best_x = x.copy()
        
        if not improved:
            break
        
        sweep += 1
    
    # Final refinement with Nelder-Mead on the best solution
    if time.time() - start_time < timeout - 30:
        def obj_full(x_vals):
            return calc_area(x_vals)
        
        res = scipy.optimize.minimize(
            obj_full,
            best_x,
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 5000, 'adaptive': True}
        )
        
        if res.success and res.fun < best_area:
            best_area = res.fun
            best_x = res.x
    
    return best_x
