import numpy as np
import time
from scipy.optimize import minimize

def _compute_volume_1d(a, num_c=100):
    N = 8
    c_vals = np.linspace(0, 1, num_c)
    total = 0.0
    for c in c_vals:
        starts = a + c * np.arange(N) / N
        ends = starts + 1.0 / N
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        length = 0.0
        curr_start = s_sorted[0]
        curr_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] <= curr_end:
                if e_sorted[k] > curr_end:
                    curr_end = e_sorted[k]
            else:
                length += curr_end - curr_start
                curr_start = s_sorted[k]
                curr_end = e_sorted[k]
        length += curr_end - curr_start
        total += length * length
    return total / num_c

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570.0
    
    N = 8
    
    def objective(x, num_c=50):
        if time.time() > deadline - 5:
            return 1e9
        a = np.zeros(N)
        a[0] = 0.0
        a[1:] = x
        return _compute_volume_1d(a, num_c=num_c)
        
    x0 = np.zeros(N-1)
    
    best_x = x0
    best_val = objective(x0, 100)
    
    rng = np.random.default_rng(42)
    
    # Run several restarts
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        if restart == 0:
            x_curr = x0
        else:
            # Random initialization within a reasonable range
            # The shifts should be small, maybe within [-0.5, 0.5]
            x_curr = rng.uniform(-0.5, 0.5, N-1)
            
        # Use L-BFGS-B
        try:
            res = minimize(lambda x: objective(x, 50), x_curr, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final polish with higher resolution
    if time.time() < deadline - 10:
        try:
            res = minimize(lambda x: objective(x, 200), best_x, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    a_opt = np.zeros(N)
    a_opt[0] = 0.0
    a_opt[1:] = best_x
    
    i_k = np.arange(64) // N
    j_k = np.arange(64) % N
    
    x_pos = a_opt[i_k]
    y_pos = a_opt[j_k]
    
    positions = np.stack([x_pos, y_pos], axis=1)
    
    return positions
