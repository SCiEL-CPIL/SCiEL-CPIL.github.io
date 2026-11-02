import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def _score_set(U):
    U = np.asarray(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0:
        return 0.0
    if U[0] != 0:
        U = np.concatenate(([0], U))
        n = len(U)
    M = int(U[-1])
    if M < 1:
        return 0.0
    vec = np.zeros(M + 1, dtype=np.float64)
    vec[U] = 1.0
    f = np.fft.rfft(vec, n=2 * (M + 1))
    s = np.fft.irfft(f * f, n=2 * (M + 1))
    sumset = int(np.sum(s >= 0.5))
    g = np.fft.irfft(f * np.conj(f), n=2 * (M + 1))
    diffset = int(np.sum(g >= 0.5))
    ratio = diffset / sumset if sumset > 0 else 0.0
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(base)

def construct():
    best_U = [0]
    best_score = 0.0
    
    # Candidate 1: Geometric progression 2^k - 1
    for k in range(1, 12):
        U = [0] + [2**i - 1 for i in range(1, k+1)]
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
            
    # Candidate 2: Pure powers of 2
    for k in range(1, 12):
        U = [0] + [2**i for i in range(k)]
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
            
    # Candidate 3: Small random search around best found
    # If best is small, try adding elements
    if len(best_U) < 10:
        U = list(best_U)
        cur_score = best_score
        for _ in range(50):
            if time.time() > _DEADLINE - 5:
                break
            # Try adding a random element
            max_val = max(U)
            c = np.random.randint(1, max_val * 2 + 1)
            if c not in U:
                trial = sorted(U + [c])
                s = _score_set(trial)
                if s > cur_score:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    
    return best_U
