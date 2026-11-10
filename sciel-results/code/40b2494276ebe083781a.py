import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 580.0
    
    def compute_union_length(x, c):
        # x: shape (N,)
        # c: scalar
        # Returns scalar U(c)
        L = x + c * np.arange(N) / N
        R = L + 1.0 / N
        # Sort intervals
        order = np.argsort(L)
        L_sorted = L[order]
        R_sorted = R[order]
        
        # Merge
        # Union length
        curr_end = L_sorted[0]
        curr_start = L_sorted[0]
        total_len = 0.0
        # Vectorized merge is tricky for scalar c, but N=8 is small.
        # Simple loop is fine.
        for k in range(1, N):
            if L_sorted[k] > curr_end:
                total_len += curr_end - curr_start
                curr_start = L_sorted[k]
                curr_end = R_sorted[k]
            else:
                curr_end = max(curr_end, R_sorted[k])
        total_len += curr_end - curr_start
        return total_len

    def exact_objective(x):
        # Compute critical c values
        c_vals = [0.0, 1.0]
        for i in range(N):
            for j in range(N):
                if i == j: continue
                # L_i = L_j => c = 8(x_i - x_j) / (j - i)
                denom = (j - i) * 8.0
                c1 = (x[i] - x[j]) / (denom / 8.0) # Wait.
                # x_i + c i/8 = x_j + c j/8
                # c (j-i)/8 = x_i - x_j
                # c = 8 (x_i - x_j) / (j-i)
                c1 = 8.0 * (x[i] - x[j]) / (j - i)
                
                # L_i = R_j => x_i + c i/8 = x_j + c j/8 + 1/8
                # c (i-j)/8 = x_j - x_i + 1/8
                # c = 8 (x_j - x_i + 1/8) / (i-j)
                c2 = 8.0 * (x[j] - x[i] + 1.0/N) / (i - j)
                
                # R_i = L_j => x_i + c i/8 + 1/8 = x_j + c j/8
                # c (i-j)/8 = x_j - x_i - 1/8
                # c = 8 (x_j - x_i - 1/8) / (i-j)
                c3 = 8.0 * (x[j] - x[i] - 1.0/N) / (i - j)
                
                for c_val in [c1, c2, c3]:
                    if 0.0 < c_val < 1.0:
                        c_vals.append(c_val)
        
        c_vals = np.array(c_vals)
        c_vals = np.unique(c_vals)
        c_vals = np.sort(c_vals)
        
        integral = 0.0
        for k in range(len(c_vals) - 1):
            c_start = c_vals[k]
            c_end = c_vals[k+1]
            if c_end - c_start < 1e-12:
                continue
            U_start = compute_union_length(x, c_start)
            U_end = compute_union_length(x, c_end)
            
            # U(c) is linear on [c_start, c_end]
            # U(c) = A c + B
            # A = (U_end - U_start) / (c_end - c_start)
            # B = U_start - A c_start
            A = (U_end - U_start) / (c_end - c_start)
            B = U_start - A * c_start
            
            # Integral of (Ac+B)^2 from c_start to c_end
            # = [ A^2 c^3/3 + A B c^2 + B^2 c ]
            def F(c):
                return A**2 * c**3 / 3.0 + A * B * c**2 + B**2 * c
            
            integral += F(c_end) - F(c_start)
        
        return integral

    # Optimization
    best_x = None
    best_val = np.inf
    
    # Initial guesses
    # 1. Symmetric
    x0_sym = -np.arange(N) / (2.0 * N)
    # 2. Random
    rng = np.random.default_rng(12345)
    
    candidates = [x0_sym]
    for _ in range(10):
        candidates.append(x0_sym + rng.normal(0, 0.1, N))
    
    for x_start in candidates:
        if time.time() > deadline:
            break
        try:
            res = minimize(exact_objective, x_start, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.fun < best_val and np.all(np.isfinite(res.x)):
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Refine with Powell if time permits
    if best_x is not None and time.time() < deadline - 10:
        try:
            res_powell = minimize(exact_objective, best_x, method='Powell', options={'maxiter': 1000, 'ftol': 1e-12})
            if res_powell.fun < best_val and np.all(np.isfinite(res_powell.x)):
                best_val = res_powell.fun
                best_x = res_powell.x
        except Exception:
            pass

    # Construct positions
    positions = np.zeros((64, 2), dtype=np.float64)
    for k in range(64):
        i = k // 8
        j = k % 8
        positions[k, 0] = best_x[i]
        positions[k, 1] = best_x[j]
        
    return positions
