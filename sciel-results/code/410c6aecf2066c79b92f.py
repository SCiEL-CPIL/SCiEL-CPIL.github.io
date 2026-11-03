import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _autoconv_component(h, k_idx):
    """Compute (h*h)_k for a specific index k (0 to 2N-2)."""
    lo = max(0, k_idx - N + 1)
    hi = min(N - 1, k_idx)
    if lo > hi:
        return 0.0
    i_vals = np.arange(lo, hi + 1)
    j_vals = k_idx - i_vals
    return np.sum(h[i_vals] * h[j_vals])

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Try delta functions at all positions
    for center in range(N):
        h_delta = np.zeros(N)
        h_delta[center] = 1.0
        s = _score_raw(h_delta)
        if s > best_score:
            best_score = s
            best_h = h_delta.copy()
    
    # 2. Try small sparse configurations with 2-3 peaks
    # For 2 peaks at positions i, j with heights a, b (a+b=1):
    # Score = (a^2 + b^2) / (a^2 + b^2 + 2ab*delta_{i,j,k} for peak k)
    # The peak of autoconvolution will be at max(i+i, j+j, i+j) depending on values
    
    # Try all pairs of positions
    for i in range(N):
        for j in range(i+1, N):
            for a in np.linspace(0.1, 0.9, 9):
                b = 1.0 - a
                h_2peak = np.zeros(N)
                h_2peak[i] = a
                h_2peak[j] = b
                s = _score_raw(h_2peak)
                if s > best_score:
                    best_score = s
                    best_h = h_2peak.copy()
    
    # 3. Try 3-peak configurations with symmetric patterns
    for i in range(N):
        for j in range(i+2, N):
            for k in range(j+2, N):
                # Symmetric: equal heights
                h_3peak = np.zeros(N)
                h_3peak[i] = 1.0/3
                h_3peak[j] = 1.0/3
                h_3peak[k] = 1.0/3
                s = _score_raw(h_3peak)
                if s > best_score:
                    best_score = s
                    best_h = h_3peak.copy()
    
    # 4. Random sparse initializations
    for _ in range(200):
        if time.time() > deadline - 30:
            break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(1, 8)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand /= np.sum(h_rand)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()
    
    # 5. Optimization with fixed peak index
    # The key insight: for sparse solutions, the peak index is determined by the peak positions
    # We can fix the peak index and optimize
    
    h_ref = best_h.copy()
    
    for iteration in range(30):
        if time.time() > deadline - 10:
            break
        
        h_conv_ref = np.convolve(h_ref, h_ref, mode='full')
        peak_idx = np.argmax(h_conv_ref)
        
        def neg_obj_fixed_peak(x, k_idx):
            h = np.exp(x)
            h = h / np.sum(h)
            l2_sq = np.sum(h**2)
            val = _autoconv_component(h, k_idx)
            if val <= 0:
                return 0.0
            return -(l2_sq / val)
        
        def grad_obj_fixed_peak(x, k_idx):
            h = np.exp(x)
            h = h / np.sum(h)
            l2_sq = np.sum(h**2)
            val = _autoconv_component(h, k_idx)
            if val <= 0:
                return np.zeros_like(x)
            
            # d(l2_sq)/dx_i = 2*h_i^2 - 2*h_i^3 - 2*h_i*l2_sq
            d_l2 = 2*h**2 - 2*h**3 - 2*h*l2_sq
            
            # d(val)/dx_i where val = sum_{i+j=k} h_i*h_j
            d_val = np.zeros(N)
            for i in range(N):
                j = k_idx - i
                if 0 <= j < N:
                    d_val[i] += h[j]
                    if 2*i == k_idx:
                        d_val[i] += h[i]
            
            grad = -(d_l2 * val - l2_sq * d_val) / (val**2)
            return grad
        
        h_ref[h_ref < 1e-10] = 1e-10
        x_ref = np.log(h_ref)
        
        try:
            res = minimize(neg_obj_fixed_peak, x_ref, args=(peak_idx,),
                          method='L-BFGS-B', jac=lambda x, k_idx: grad_obj_fixed_peak(x, k_idx),
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
                h_ref = h_candidate
        except Exception:
            try:
                res = minimize(neg_obj_fixed_peak, x_ref, args=(peak_idx,),
                              method='L-BFGS-B', options={'maxiter': 800, 'ftol': 1e-14})
                if res.success:
                    h_candidate = np.exp(res.x)
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
                    h_ref = h_candidate
            except Exception:
                break
    
    # 6. Try optimizing with different peak indices explicitly
    # The optimal solution might have its peak at a non-obvious location
    for peak_idx in range(0, 2*N-1, 2):  # Try even indices (symmetric peaks)
        if time.time() > deadline - 5:
            break
        
        h_start = np.zeros(N)
        h_start[peak_idx // 2] = 1.0 if peak_idx < N else 1.0
        h_start[h_start < 1e-10] = 1e-10
        x_start = np.log(h_start)
        
        def neg_obj_peak(x, k_idx):
            h = np.exp(x)
            h = h / np.sum(h)
            l2_sq = np.sum(h**2)
            val = _autoconv_component(h, k_idx)
            if val <= 0:
                return 0.0
            return -(l2_sq / val)
        
        try:
            res = minimize(neg_obj_peak, x_start, args=(peak_idx,),
                          method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-14})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
    
    # 7. Final polish with exact objective
    if time.time() < deadline - 3:
        h_start = best_h / np.sum(best_h)
        h_start[h_start < 1e-10] = 1e-10
        x0 = np.log(h_start)
        
        def neg_obj_exact(x):
            h = np.exp(x)
            h = h / np.sum(h)
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0:
                return 0.0
            return -(l2_sq / max_conv)
        
        try:
            res = minimize(neg_obj_exact, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
    
    return list(best_h)
