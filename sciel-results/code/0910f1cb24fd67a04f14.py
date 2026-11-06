import time
import random
import itertools

_DEADLINE = 85.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _square_nonzero_count(coeffs):
    """Return number of nonzero entries in the exact square of coeffs (ascending degree)."""
    n = len(coeffs)
    # convolution
    size = 2 * n - 1
    sq = [0] * size
    for i in range(n):
        ai = coeffs[i]
        if ai == 0:
            continue
        for j in range(i, n):
            aj = coeffs[j]
            if aj == 0:
                continue
            s = 2 * ai * aj
            if i == j:
                sq[i + j] += s
            else:
                sq[i + j] += s
                sq[j + i] += s
    # trim trailing zeros
    while len(sq) > 1 and sq[-1] == 0:
        sq.pop()
    return sum(1 for v in sq if v != 0), sq

def _validate(coeffs):
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    for v in coeffs:
        if not isinstance(v, int) or v < -64 or v > 64:
            return False
    nz = sum(1 for v in coeffs if v != 0)
    if nz != 12:
        return False
    return True

def _score(coeffs):
    if not _validate(coeffs):
        return None
    cnt, _ = _square_nonzero_count(coeffs)
    return cnt

def _local_search(exps, coeffs, rng, max_iters=500):
    """Hill-climb on exponents and coefficients."""
    current_exps = list(exps)
    current_coeffs = list(coeffs)
    best_score, _ = _square_nonzero_count(_build_coeffs(current_exps, current_coeffs))
    
    for _ in range(max_iters):
        if _elapsed() > _DEADLINE:
            break
        # Randomly perturb exponent or coefficient
        if rng.random() < 0.5:
            pos = rng.randrange(12)
            old_exp = current_exps[pos]
            # Try shifting exponent
            delta = rng.choice([-2, -1, 1, 2])
            new_exp = old_exp + delta
            if 0 <= new_exp <= 40 and new_exp not in current_exps:
                current_exps[pos] = new_exp
                arr = _build_coeffs(current_exps, current_coeffs)
                sc, _ = _square_nonzero_count(arr)
                if sc < best_score:
                    best_score = sc
                else:
                    current_exps[pos] = old_exp
        else:
            pos = rng.randrange(12)
            old_val = current_coeffs[pos]
            # Try changing coefficient
            new_val = rng.choice([-2, -1, 1, 2, -3, 3, -4, 4])
            if new_val != old_val:
                current_coeffs[pos] = new_val
                arr = _build_coeffs(current_exps, current_coeffs)
                sc, _ = _square_nonzero_count(arr)
                if sc < best_score:
                    best_score = sc
                else:
                    current_coeffs[pos] = old_val
    
    return _build_coeffs(current_exps, current_coeffs), best_score

def _build_coeffs(exps, coeffs):
    max_deg = max(exps)
    arr = [0] * (max_deg + 1)
    for e, c in zip(exps, coeffs):
        arr[e] = c
    return arr

def construct():
    rng = random.Random(42)
    best_score = 999
    best_coeffs = None
    
    # Initial candidates
    candidates = []
    
    # 1. Simple arithmetic progression
    exps = list(range(12))
    coeffs = [1]*12
    candidates.append((exps, coeffs))
    
    # 2. Symmetric clusters
    exps = [0,1,2,3,4,5, 20,21,22,23,24,25]
    coeffs = [1]*6 + [1]*6
    candidates.append((exps, coeffs))
    
    # 3. Alternating signs in clusters
    exps = [0,1,2,3,4,5, 20,21,22,23,24,25]
    coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    candidates.append((exps, coeffs))
    
    # 4. Power of 2 exponents (Golomb ruler)
    exps = [0, 1, 2, 4, 8, 16, 20, 24, 30, 36, 38, 40]
    coeffs = [1]*12
    candidates.append((exps, coeffs))
    
    # 5. Random starts
    for _ in range(10):
        exps = sorted(rng.sample(range(41), 12))
        coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
        candidates.append((exps, coeffs))
    
    for exps, coeffs in candidates:
        if _elapsed() > _DEADLINE:
            break
        arr, sc = _local_search(exps, coeffs, rng, max_iters=200)
        if sc < best_score:
            best_score = sc
            best_coeffs = arr
    
    # Final validation
    if not _validate(best_coeffs):
        best_coeffs = [1]*12
    
    return best_coeffs
