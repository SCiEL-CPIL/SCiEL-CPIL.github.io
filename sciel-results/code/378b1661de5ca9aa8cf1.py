"""Combined program: Multi-start with Thue-Morse and Sparse structures."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit

def _autoconv_max_abs(h):
    """Compute max |autoconvolution| of h."""
    ac = fftconvolve(h, h, mode='full')
    return np.max(np.abs(ac))

def _squared_integral(h):
    """Compute (sum h)^2."""
    s = np.sum(h)
    return s * s

def _score(h):
    """Compute the objective: max|autoconv| / (sum h)^2. Lower is better."""
    si = _squared_integral(h)
    if si < 1e-12:
        return np.inf
    return _autoconv_max_abs(h) / si

def _local_sign_flip_optimize(h, max_rounds=50, time_budget=None):
    """Iteratively flip signs of individual elements to reduce the objective."""
    best_h = h.copy().astype(float)
    best_score = _score(best_h)
    
    for _ in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        # Try flipping each position
        for i in range(len(best_h)):
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            
            best_h[i] *= -1
            new_score = _score(best_h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_h[i] *= -1  # revert
        
        if not improved:
            break
    
    return best_h, best_score

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    tm = np.zeros(n, dtype=int)
    for i in range(n):
        tm[i] = bin(i).count('1') % 2
    return tm

def construct():
    """Construct a signed step function minimizing max|autoconv|/(sum h)^2."""
    candidates = []
    
    # 1. Constant vector (Baseline)
    candidates.append(np.ones(N))
    
    # 2. Thue-Morse based
    # Map Thue-Morse (0,1) to signs (-1, 1)
    tm = _thue_morse(N)
    h_tm = 1 - 2 * tm  # 0->1, 1->-1
    candidates.append(h_tm)
    
    # 3. Sparse Random
    # 50% non-zero, random signs
    mask = np.random.rand(N) < 0.5
    signs = np.random.choice([-1, 1], N)
    h_sparse = mask * signs
    candidates.append(h_sparse)
    
    # 4. Sparse Thue-Morse
    # Use Thue-Morse for signs, but make it sparse?
    # Or just use Thue-Morse signs on a sparse mask
    mask2 = np.random.rand(N) < 0.5
    h_sparse_tm = mask2 * h_tm
    candidates.append(h_sparse_tm)

    best_h = None
    best_score = np.inf
    
    time_budget = _TIME_LIMIT - 10
    
    for h in candidates:
        if time.time() - _START_TIME > time_budget:
            break
        
        # Optimize the candidate
        h_opt, score = _local_sign_flip_optimize(h, max_rounds=10, time_budget=time_budget)
        
        if score < best_score:
            best_score = score
            best_h = h_opt
    
    # If no candidate was processed (timeout), return constant
    if best_h is None:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
