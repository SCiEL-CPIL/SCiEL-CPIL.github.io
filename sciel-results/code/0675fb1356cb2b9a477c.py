import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
warnings.filterwarnings('ignore')

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|) using discrete convolution."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution via discrete convolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _score_neg(h):
    return -_compute_ratio(h)

def _generate_sidon_set(n, size):
    """Generate a Sidon subset of {0,...,n-1} of given size using greedy."""
    S = []
    sums = set()
    for i in range(n):
        if len(S) >= size:
            break
        new_sums = set()
        valid = True
        for j in S + [i]:
            for k in S + [i]:
                if j <= k:
                    s = j + k
                    if s in sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
        if valid:
            S.append(i)
            sums |= new_sums
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try various Sidon set sizes
    for size in range(2, 15):
        if time.time() > deadline - 15:
            break
        
        S = _generate_sidon_set(N, size)
        if len(S) < size:
            continue
        
        # Initialize with equal heights at Sidon positions
        h0 = np.zeros(N)
        h0[S] = 1.0 / size
        
        score0 = _compute_ratio(h0)
        if score0 > best_score:
            best_score = score0
            best_h = h0.copy()
        
        # Optimize the heights with L-BFGS-B
        if time.time() > deadline - 10:
            continue
        try:
            bounds = [(0, 10)] * N
            result = minimize(_score_neg, h0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            h_trial = np.maximum(result.x, 0.0)
            # Only keep nonzero at Sidon positions to maintain structure
            mask = np.zeros(N, dtype=bool)
            mask[S] = True
            h_trial[~mask] = 0.0
            score_trial = _compute_ratio(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    # Also try the original 2-element support as baseline
    if time.time() < deadline - 5:
        h0 = np.zeros(N)
        h0[0] = 2.0/3.0
        h0[1] = 1.0/3.0
        score0 = _compute_ratio(h0)
        if score0 > best_score:
            best_score = score0
            best_h = h0.copy()
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 3:
        try:
            result = minimize(_score_neg, best_h, method='L-BFGS-B',
                           bounds=[(0, 10)] * N,
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_ratio(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    return list(best_h)
