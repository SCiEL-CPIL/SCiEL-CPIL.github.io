import time
import math
import random
import numpy as np

def _score_of(U):
    n = len(U)
    if n < 2:
        return 0.0
    a = np.asarray(U, dtype=np.int64)
    s = a[:, None] + a[None, :]
    d = a[:, None] - a[None, :]
    ns = len(np.unique(s))
    nd = len(np.unique(d))
    if ns <= 0:
        return 0.0
    ratio = nd / ns
    if ratio <= 0:
        return 0.0
    maxU = a.max()
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    def now_ok():
        return (time.time() - start_time) < DEADLINE

    best_U = [0, 1, 3]
    best_score = _score_of(best_U)

    def try_candidate(U):
        nonlocal best_U, best_score
        if not now_ok():
            return False
        U = sorted(set(int(x) for x in U))
        if 0 not in U or len(U) < 2:
            return now_ok()
        s = _score_of(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        return now_ok()

    # 1. Search for geometric sets {0, b, b^2}
    # Score for {0, b, b^2} is 1 + log((2b^2+2b+1)/6) / log(2b^2+1)
    # This is maximized for large b.
    for b in range(2, 10000):
        if not now_ok():
            break
        if b % 10 != 0 and b > 100: # coarse scan
            continue
        U = [0, b, b*b]
        try_candidate(U)
        
    # Fine scan around best b if time permits
    # Actually, the function is monotonic increasing for b > 1.
    # So just take the largest b possible within time.
    # But we also need to check other structures.

    # 2. Search for {0, b, b^2, c}
    # Optimize c for a given b
    for b in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        if not now_ok():
            break
        base = [0, b, b*b]
        # c should be large, maybe > b^2
        # Scan c
        c_lo = b*b + 1
        c_hi = b*b * 10
        step = max(1, (c_hi - c_lo) // 100)
        best_c = None
        best_s = -1
        for c in range(c_lo, c_hi, step):
            if not now_ok():
                break
            U = base + [c]
            s = _score_of(U)
            if s > best_s:
                best_s = s
                best_c = c
        if best_c:
            try_candidate(base + [best_c])
            # Fine scan
            lo = max(c_lo, best_c - 100)
            hi = min(c_hi, best_c + 100)
            for c in range(lo, hi):
                if not now_ok():
                    break
                U = base + [c]
                try_candidate(U)

    # 3. Random Hill Climbing
    rng = random.Random(12345)
    current_U = best_U[:]
    
    def perturb(U):
        U = sorted(set(U))
        op = rng.random()
        if op < 0.3 and len(U) > 2:
            idx = rng.randrange(1, len(U))
            U = U[:idx] + U[idx+1:]
        elif op < 0.7:
            mx = max(U) if U else 0
            if rng.random() < 0.5:
                x = rng.randint(1, max(2, mx // 2))
            else:
                x = mx + rng.randint(1, max(2, mx // 3))
            U.append(x)
        else:
            idx = rng.randrange(1, len(U))
            old = U[idx]
            delta = rng.randint(1, max(1, old // 2))
            if rng.random() < 0.5:
                new = old - delta
            else:
                new = old + delta
            if new <= 0:
                new = old + delta
            U[idx] = new
            U = sorted(set(U))
        return U

    for _ in range(20000):
        if not now_ok():
            break
        U = perturb(current_U)
        if try_candidate(U):
            if _score_of(U) > best_score:
                current_U = U
        else:
            break

    return best_U

if __name__ == "__main__":
    U = construct()
    print(U)
    print(_score_of(U))
