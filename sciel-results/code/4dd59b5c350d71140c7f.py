import time
import random
import numpy as np

_START = time.time()
_DEADLINE = 55.0  # safety margin before 60s kill

def _sq_nonzero_count(coefs):
    """Count nonzero terms in the exact square using numpy convolution."""
    arr = np.array(coefs, dtype=np.int64)
    sq = np.convolve(arr, arr)
    return int(np.count_nonzero(sq))

def _valid(coefs):
    if len(coefs) > 41:
        return False
    nz = 0
    for c in coefs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    if coefs[-1] == 0:
        return False
    return True

def _polish(best, support, rng):
    """Local hill-climbing polish using numpy-accelerated scoring."""
    hi = max(support)
    best_local = best[1][:] if best[1] else None
    if best_local is None:
        return best
    best_score = best[0]
    
    # Value candidates to try
    val_candidates = [-64,-32,-16,-8,-4,-3,-2,-1,1,2,3,4,8,16,32,64]
    
    for _pass in range(5):
        if time.time() > _DEADLINE:
            break
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            for d in support:
                if time.time() > _DEADLINE:
                    break
                old_val = best_local[d]
                for new_val in val_candidates:
                    if new_val == old_val:
                        continue
                    best_local[d] = new_val
                    if best_local[hi] == 0:
                        best_local[d] = old_val
                        continue
                    sc = _sq_nonzero_count(best_local)
                    if sc < best_score:
                        best_score = sc
                        best[0] = sc
                        best[1] = best_local[:]
                        improved = True
                    else:
                        best_local[d] = old_val
    return best

def _try_place(best, support, rng):
    """Assign coefficients to support and optimize."""
    n = len(support)
    hi = max(support)
    best_local = None
    best_score = 10**9
    val_pool = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    for _ in range(200):
        if time.time() > _DEADLINE:
            break
        coefs = [0] * 41
        for d in support:
            c = rng.choice(val_pool)
            coefs[d] = c
        if coefs[hi] == 0:
            coefs[hi] = 1
        sc = _sq_nonzero_count(coefs)
        if sc < best_score:
            best_score = sc
            best_local = coefs[:]
        
        # Local perturbation
        if best_local is not None:
            cand = best_local[:]
            d = support[rng.randrange(n)]
            old = cand[d]
            new = old + rng.choice([-4,-3,-2,-1,1,2,3,4])
            if -64 <= new <= 64 and new != 0:
                cand[d] = new
            if cand[hi] == 0:
                cand[hi] = 1
            sc2 = _sq_nonzero_count(cand)
            if sc2 < best_score:
                best_score = sc2
                best_local = cand[:]
    
    if best_local is not None and _valid(best_local):
        sc = _sq_nonzero_count(best_local)
        if sc < best[0]:
            best[0] = sc
            best[1] = best_local[:]
            _polish(best, support, rng)
    return best

def _make_symmetric_support(rng):
    """Create palindromic support around degree 20."""
    pairs = [(d, 40-d) for d in range(0, 21) if d != 40-d]
    rng.shuffle(pairs)
    support = set()
    for d1, d2 in pairs:
        if len(support) >= 12:
            break
        support.add(d1)
        support.add(d2)
    if 40 not in support:
        support = set(list(support)[:11] + [40])
    support = sorted(support)
    if len(support) != 12:
        if len(support) > 12:
            support = sorted(support[-12:])
        else:
            for d in range(40, -1, -1):
                if d not in support:
                    support.append(d)
                    support = sorted(support)
                if len(support) == 12:
                    break
    return support[:12]

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 55.0
    rng = random.Random(12345)
    best = [10**9, None]

    # 1) Palindromic/symmetric supports
    for t in range(50):
        if time.time() > _DEADLINE:
            break
        sup = _make_symmetric_support(rng)
        _try_place(best, sup, rng)

    # 2) Arithmetic progressions
    for start in range(0, 10):
        for gap in range(2, 14):
            if time.time() > _DEADLINE:
                break
            sup = []
            d = start
            while d <= 40 and len(sup) < 12:
                sup.append(d)
                d += gap
            if len(sup) < 12:
                continue
            if 40 not in sup:
                sup = sup[:-1] + [40]
                sup = sorted(set(sup))
                if len(sup) != 12:
                    continue
            _try_place(best, sup, rng)

    # 3) Randomized supports
    def make_support():
        pool = list(range(0, 40))
        rng.shuffle(pool)
        chosen = sorted(pool[:11]) + [40]
        return chosen

    for t in range(40):
        if time.time() > _DEADLINE:
            break
        sup = make_support()
        _try_place(best, sup, rng)

    if best[1] is None:
        coefs = [0] * 41
        for i in range(12):
            coefs[i] = 1
        return coefs

    return best[1]
