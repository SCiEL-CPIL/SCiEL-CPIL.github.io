import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 130.0

def _elapsed():
    return time.time() - _START

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
    for k in range(1, 14):
        U = [0] + [2**i - 1 for i in range(1, k+1)]
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
            
    # Candidate 2: Pure powers of 2
    for k in range(1, 14):
        U = [0] + [2**i for i in range(k)]
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Candidate 3: Golomb-ruler-like constructions
    for k in range(2, 10):
        U = [0]
        step = 1
        for i in range(k-1):
            step *= 2
            U.append(U[-1] + step)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U

    # Candidate 5: Small exhaustive search for optimal small sets
    for n in range(4, 7):
        for max_val in [6, 8, 10, 12]:
            if _elapsed() > 10:
                break
            # Try all combinations of n-1 elements from 1..max_val
            from itertools import combinations
            for combo in combinations(range(1, max_val+1), n-1):
                if _elapsed() > 15:
                    break
                U = sorted([0] + list(combo))
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U
            if _elapsed() > 15:
                break
        if _elapsed() > 15:
            break
    
    # Candidate 4: Sparse random sets with controlled max
    rng = np.random.RandomState(42)
    for max_val in [10, 20, 40, 80, 160, 320]:
        for size in range(3, min(15, max_val//2 + 1)):
            if _elapsed() > 25:
                break
            for _ in range(5):
                candidates = rng.choice(range(1, max_val+1), size=size, replace=False)
                U = sorted([0] + list(candidates))
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U
        if _elapsed() > 25:
            break
    
    # Local search with add, remove, and swap moves
    U = list(best_U)
    cur_score = best_score
    improved = True
    while improved and _elapsed() < 105:
        improved = False
        max_val = max(U) if U else 1
        
        # Try adding elements
        for _ in range(40):
            if _elapsed() > 115:
                break
            c = rng.randint(1, max_val * 4 + 1)
            if c not in U:
                trial = sorted(U + [c])
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
        
        # Try removing elements (if set is large enough)
        if len(U) > 2:
            for idx in range(len(U)):
                if _elapsed() > 115:
                    break
                if U[idx] == 0:
                    continue
                trial = U[:idx] + U[idx+1:]
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
        
        # Try swap moves: replace one element with another
        if len(U) > 2:
            for idx in range(1, len(U)):  # skip index 0 (which is 0)
                if _elapsed() > 115:
                    break
                for _ in range(10):
                    if _elapsed() > 115:
                        break
                    new_val = rng.randint(1, max_val * 4 + 1)
                    if new_val not in U:
                        trial = U.copy()
                        trial[idx] = new_val
                        trial = sorted(trial)
                        s = _score_set(trial)
                        if s > cur_score + 1e-12:
                            cur_score = s
                            U = trial
                            best_U = U
                            best_score = cur_score
                            improved = True
                            break
    
    return best_U
