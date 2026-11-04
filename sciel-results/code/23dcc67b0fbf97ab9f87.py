"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _autoconv_bound(f):
    """Compute the autoconvolution upper bound on C3.
    
    The autoconvolution f*f at lag k is sum_i f[i]*f[i+k] (circular or linear).
    For the third autocorrelation inequality, we consider the L1 norm of the
    autoconvolution or its peak value.
    
    Using linear autoconvolution (full), the bound relates to max |f*f|.
    """
    # Linear autoconvolution
    ac = np.convolve(f, f, mode='full')
    # The upper bound on C3 is typically related to the L1 norm or max of |ac|
    # For the "third" inequality, it might be the cubic norm or related
    # Let's use the L1 norm of the autoconvolution as a proxy for the bound
    return np.sum(np.abs(ac))

def _autoconv_bound_v2(f):
    """Alternative: peak autoconvolution value."""
    ac = np.convolve(f, f, mode='full')
    return np.max(np.abs(ac))

def _autoconv_bound_v3(f):
    """Another metric: sum of squares of autoconvolution."""
    ac = np.convolve(f, f, mode='full')
    return np.sum(ac**2)

def _try_pattern(pattern_str):
    """Create a signed step function from a pattern string like '+-+-'."""
    signs = np.array([1 if c == '+' else -1 for c in pattern_str])
    # Repeat to fill N
    if len(signs) < N:
        repeats = N // len(signs) + 1
        signs = np.tile(signs, repeats)[:N]
    return signs

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try several candidate patterns
    candidates = []
    
    # Pattern 1: Alternating +-
    candidates.append(_try_pattern('+-'))
    
    # Pattern 2: ++--
    candidates.append(_try_pattern('++--'))
    
    # Pattern 3: +--
    candidates.append(_try_pattern('+--'))
    
    # Pattern 4: +++---
    candidates.append(_try_pattern('+++---'))
    
    # Pattern 5: +++++----- (5 plus, 5 minus)
    candidates.append(_try_pattern('+++++-----'))
    
    # Pattern 6: ++++++------ (6 plus, 6 minus)
    candidates.append(_try_pattern('++++++------'))
    
    # Pattern 7: ++++++++-------- (8 plus, 8 minus)
    candidates.append(_try_pattern('++++++++--------'))
    
    # Pattern 8: ++++++++-------- (10 plus, 10 minus)
    candidates.append(_try_pattern('++++++++++----------'))
    
    # Pattern 9: Random with zero mean
    rng = np.random.RandomState(42)
    for seed in range(5):
        rng = np.random.RandomState(seed)
        f = rng.choice([1, -1], size=N)
        # Ensure roughly balanced
        candidates.append(f)
    
    # Evaluate all candidates
    best_f = None
    best_score = np.inf
    
    for f in candidates:
        score = _autoconv_bound(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
        if time.time() > deadline:
            break
    
    # Local search: flip individual signs to improve
    if best_f is not None and time.time() < deadline:
        current = best_f.copy()
        current_score = _autoconv_bound(current)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                old = current[i]
                current[i] = -old
                new_score = _autoconv_bound(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old
    
    return current if best_f is None else current
# EVOLVE-BLOCK-END
