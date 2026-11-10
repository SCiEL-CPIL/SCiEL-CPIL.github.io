# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _exact_area(x, n=64, M=20000):
    """Compute exact area of union of triangles."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    
    L = x[None, :] + a[None, :] * y[:, None]
    idx = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, idx, axis=1)
    D = L_sorted[:, 1:] - L_sorted[:, :-1]
    
    union_lengths = w + np.sum(np.minimum(D, w[:, None]), axis=1)
    return np.mean(union_lengths)

def _smoothed_area_and_grad(x, eps, n=64, M=20000):
    """
    Compute the smoothed area of the union of triangles and its gradient.
    Uses a log-sum-exp based smooth approximation for min(D, w).
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    
    L = x[None, :] + a[None, :] * y[:, None]
    
    idx = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, idx, axis=1)
    
    D = L_sorted[:, 1:] - L_sorted[:, :-1]
    
    diff = D - w[:, None]
    z = diff / eps
    
    # Stable sigmoid for -z: sigma(-z) = 1/(1+exp(z))
    sig = np.empty_like(z)
    mask_pos = z >= 0
    sig[mask_pos] = 1.0 / (1.0 + np.exp(z[mask_pos]))
    exp_neg_z = np.exp(-z[~mask_pos])
    sig[~mask_pos] = exp_neg_z / (1.0 + exp_neg_z)
    
    log_terms = np.logaddexp(-D / eps, -w[:, None] / eps)
    f_vals = -eps * log_terms
    sum_f = np.sum(f_vals, axis=1)
    
    U = w + sum_f
    area = np.mean(U)
    
    g_sorted = np.zeros_like(L_sorted)
    g_sorted[:, :-1] -= sig
    g_sorted[:, 1:] += sig
    
    G = np.zeros_like(L)
    np.put_along_axis(G, idx, g_sorted, axis=1)
    grad_x = np.sum(G, axis=0) * a / M
    
    return area, grad_x

def _keich_init(n):
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def _coordinate_descent_polish(x, n=64, M=20000, max_passes=200, tol=1e-10, deadline=None):
    """
    Coordinate descent polish using exact area.
    For each coordinate, find the optimal shift by scanning a small range.
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    
    x = x.copy()
    best_area = _exact_area(x, n, M)
    
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        improved = False
        for i in range(n):
            # Scan range around current x[i]
            # The optimal x[i] should be within a reasonable range
            # Since L_i = x_i + a_i * y, and y in [0,1], L_i in [x_i, x_i + a_i]
            # The width w = (1-y)/64
            # A reasonable scan range is [-a_i, a_i] or similar
            # But for polish, we can use a smaller range
            scan_range = 0.05
            n_scan = 200
            
            x_base = x[i]
            scan_points = x_base + np.linspace(-scan_range, scan_range, n_scan)
            
            best_xi = x[i]
            best_au = best_area
            
            for xi in scan_points:
                x[i] = xi
                area = _exact_area(x, n, M)
                if area < best_au:
                    best_au = area
                    best_xi = xi
            
            x[i] = best_xi
            
            if best_au < best_area - tol:
                best_area = best_au
                improved = True
        if not improved:
            break
            
    return x, best_area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    x = _keich_init(n)
    
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin
    
    # Annealing schedule for epsilon
    eps_schedule = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    best_x = x.copy()
    best_area = _exact_area(x, n)
    
    for eps in eps_schedule:
        if time.time() > deadline - 120: # Leave time for coordinate descent
            break
            
        def func_and_grad(x_curr):
            return _smoothed_area_and_grad(x_curr, eps)
        
        try:
            res = minimize(
                func_and_grad, 
                x, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 150, 'maxfun': 400, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            if np.all(np.isfinite(res.x)):
                # Check exact area to see if we improved
                exact_area = _exact_area(res.x, n)
                if exact_area < best_area:
                    best_x = res.x.copy()
                    best_area = exact_area
                    x = res.x
        except Exception:
            pass
            
    # Coordinate descent polish
    if time.time() < deadline - 10:
        try:
            x_polished, area_polished = _coordinate_descent_polish(
                best_x, n, M=20000, max_passes=100, tol=1e-12, deadline=deadline
            )
            if area_polished < best_area:
                best_x = x_polished
                best_area = area_polished
        except Exception:
            pass
            
    # Final quick polish with smaller scan range
    if time.time() < deadline - 5:
        try:
            # Do one more pass with tighter scan
            x_final = best_x.copy()
            for i in range(n):
                if time.time() > deadline - 3:
                    break
                scan_range = 0.01
                n_scan = 100
                x_base = x_final[i]
                scan_points = x_base + np.linspace(-scan_range, scan_range, n_scan)
                
                best_xi = x_final[i]
                best_au = best_area
                
                for xi in scan_points:
                    x_final[i] = xi
                    area = _exact_area(x_final, n)
                    if area < best_au:
                        best_au = area
                        best_xi = xi
                
                x_final[i] = best_xi
            
            if best_area > 0 and np.all(np.isfinite(x_final)):
                best_x = x_final
        except Exception:
            pass

    # Ensure finite
    if not np.all(np.isfinite(best_x)):
        return _keich_init(n)
        
    return best_x

# EVOLVE-BLOCK-END
