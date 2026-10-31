import numpy as np
import time
from scipy.signal import fftconvolve

N = 400

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s (evaluator kills at 150s, but we want buffer)


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
    if si == 0:
        return np.inf
    return _autoconv_max_abs(h) / si


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (0/1 values)."""
    # Thue-Morse: t(n) = popcount(n) mod 2
    # Efficient generation using iterative doubling
    seq = np.array([0])
    while len(seq) < n:
        seq = np.concatenate([seq, 1 - seq])
    return seq[:n]


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


def construct():
    """Construct a signed step function minimizing max|autoconv|/(sum h)^2."""
    # Strategy: sparse signed sequence with Thue-Morse structure
    # Thue-Morse has low autocorrelation properties.
    # We combine it with local sign-flip optimization to polish.
    
    best_h = None
    best_score = np.inf
    
    # Candidate 1: Thue-Morse sequence
    # Thue-Morse is 0/1, map to -1/1
    tm = _thue_morse(N)
    h_tm = np.where(tm == 0, 1.0, -1.0)
    
    # Also try the complement
    h_tm_comp = -h_tm
    
    # Candidate 2: Random sparse with 50% density
    np.random.seed(42)
    mask = np.random.rand(N) < 0.5
    h_rand = np.zeros(N)
    h_rand[mask] = np.random.choice([-1, 1], size=np.sum(mask))
    
    # Candidate 3: Block pattern: +1+1-1-1 repeating
    h_block = np.tile(np.array([1, 1, -1, -1]), N // 4 + 1)[:N]
    
    # Candidate 4: All ones (baseline)
    h_ones = np.ones(N)
    
    # Candidate 5: Alternating
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    
    candidates = [h_tm, h_tm_comp, h_rand, h_block, h_ones, h_alt]
    
    # Optimize each candidate with local sign-flip
    for i, h in enumerate(candidates):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        
        h_opt, sc = _local_sign_flip_optimize(h, max_rounds=20, time_budget=_TIME_LIMIT - 5)
        
        if sc < best_score:
            best_score = sc
            best_h = h_opt.copy()
    
    # Final pass with more rounds if time allows
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 10:
        best_h, best_score = _local_sign_flip_optimize(
            best_h, max_rounds=100, time_budget=_TIME_LIMIT - 5
        )
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
