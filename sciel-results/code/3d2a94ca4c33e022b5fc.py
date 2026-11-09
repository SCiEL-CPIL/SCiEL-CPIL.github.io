import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _ratio(h):
    # h is a numpy array of nonnegative values
    # Normalize h to sum to 1 to handle scaling invariance
    s = np.sum(h)
    if s == 0:
        return 0.0
    h_norm = h / s
    
    l2_sq = np.sum(h_norm ** 2)
    l1 = 1.0 # since normalized
    
    # Compute autoconvolution
    # c_k = sum_j h_j h_{k-j}
    # Using FFT for speed
    # Zero pad to 2*N-1
    n_fft = 2 * N
    H = np.fft.rfft(h_norm, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    # The first N elements correspond to k=0 to N-1
    # Due to symmetry, we only need to check 0 to N-1
    max_c = np.max(C[:N])
    
    if max_c == 0:
        return 0.0
        
    return l2_sq / (l1 * max_c)

def _neg_ratio(h):
    return -_ratio(h)

def construct():
    start_time = time.time()
    deadline = 100.0 # 100s limit, keep 20s margin
    
    best_score = 0.0
    best_h = np.ones(N) / N
    
    # Initial candidates
    candidates = [
        np.ones(N) / N,
        np.random.exponential(1.0, N),
        np.zeros(N), # Will be handled by normalization check, maybe skip
    ]
    candidates[2][0] = 1.0
    
    # Add some random starts
    for _ in range(10):
        candidates.append(np.random.exponential(1.0, N))
        
    for h0 in candidates:
        if time.time() - start_time > deadline:
            break
        h0 = np.clip(h0, 0, None)
        if np.sum(h0) == 0:
            continue
            
        # Local refinement using Nelder-Mead
        try:
            res = minimize(_neg_ratio, h0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            h_opt = res.x
            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
        except Exception:
            pass
            
    # If time remains, try a few more random restarts with local opt
    while time.time() - start_time < deadline:
        h0 = np.random.exponential(1.0, N)
        try:
            res = minimize(_neg_ratio, h0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            h_opt = res.x
            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
        except Exception:
            pass

    return list(best_h)

