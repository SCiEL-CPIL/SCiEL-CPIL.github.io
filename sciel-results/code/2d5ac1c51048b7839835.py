import time
import math
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def _elapsed():
    return time.time() - _START

def _score_set(U):
    """Compute the score for a set U containing 0 using FFT for larger sets."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        U = [0] + [x for x in U if x > 0]
    if len(U) < 2:
        return 0.0
    M = U[-1]
    if M < 1:
        return 0.0

    # For very small sets, compute directly to avoid FFT overhead
    if len(U) <= 25:
        U_arr = np.array(U, dtype=np.int64)
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        sumset = len(np.unique(sums))
        diffset = len(np.unique(diffs))
    else:
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

def _golomb_ruler(n):
    """Generate a Golomb ruler with n marks (including 0)."""
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99],
        16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115],
        17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132],
        18: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151],
        19: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171],
        20: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192],
    }
    if n in known:
        return known[n]
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def _exhaustive_small_search(max_val=18, min_size=3, max_size=8):
    """Exhaustive search over small sets to find good configurations."""
    best_U = None
    best_score = -1.0
    
    # Generate all subsets of {0, 1, ..., max_val} of size between min_size and max_size
    # that include 0
    from itertools import combinations
    
    range_vals = list(range(1, max_val + 1))
    
    for size in range(min_size, max_size + 1):
        if _elapsed() > _DEADLINE - 20:
            break
        for combo in combinations(range_vals, size - 1):
            U = sorted([0] + list(combo))
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
    
    return best_U, best_score

def _local_search_add(U, time_limit=5.0):
    """Local search by adding elements."""
    U = sorted(set(U))
    max_u = U[-1] if U else 0
    best_score = _score_set(U)
    best_U = list(U)
    start_time = time.time()
    rng = random.Random(123)
    
    while time.time() - start_time < time_limit:
        margin = max(10, max_u // 3)
        new_val = rng.randint(0, max_u + margin)
        if new_val in U:
            continue
        
        new_U = sorted(U + [new_val])
        new_score = _score_set(new_U)
        
        if new_score > best_score:
            best_score = new_score
            best_U = new_U
            U = new_U
            max_u = U[-1]
    
    return best_U, best_score

def _local_search_swap(U, time_limit=5.0):
    """Local search by swapping elements."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return U, _score_set(U)
    
    max_u = U[-1]
    best_score = _score_set(U)
    best_U = list(U)
    start_time = time.time()
    rng = random.Random(456)
    
    while time.time() - start_time < time_limit:
        i = rng.randint(1, n - 1)  # Don't swap 0
        old_val = U[i]
        margin = max(10, max_u // 3)
        new_val = rng.randint(0, max_u + margin)
        if new_val == old_val or new_val in U:
            continue
        
        new_U = list(U)
        new_U[i] = new_val
        new_U = sorted(new_U)
        new_score = _score_set(new_U)
        
        if new_score > best_score:
            best_score = new_score
            best_U = new_U
            U = new_U
            max_u = U[-1]
    
    return best_U, best_score

def _local_search_remove(U, time_limit=3.0):
    """Local search by removing elements."""
    U = sorted(set(U))
    n = len(U)
    if n < 3:
        return U, _score_set(U)
    
    best_score = _score_set(U)
    best_U = list(U)
    start_time = time.time()
    rng = random.Random(789)
    
    while time.time() - start_time < time_limit:
        if len(U) < 3:
            break
        i = rng.randint(1, len(U) - 1)  # Don't remove 0
        new_U = U[:i] + U[i+1:]
        new_score = _score_set(new_U)
        
        if new_score > best_score:
            best_score = new_score
            best_U = new_U
            U = new_U
    
    return best_U, best_score

def _hill_climbing(U, time_limit=15.0):
    """Combined local search with add, swap, and remove operations."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = list(U)
    start_time = time.time()
    rng = random.Random(321)
    
    while time.time() - start_time < time_limit:
        # Choose operation
        op = rng.choice(['add', 'swap', 'remove'])
        
        if op == 'add':
            max_u = U[-1] if U else 0
            margin = max(10, max_u // 3)
            new_val = rng.randint(0, max_u + margin)
            if new_val not in U:
                new_U = sorted(U + [new_val])
                new_score = _score_set(new_U)
                if new_score > best_score:
                    best_score = new_score
                    best_U = new_U
                    U = new_U
        
        elif op == 'swap' and len(U) > 2:
            i = rng.randint(1, len(U) - 1)
            old_val = U[i]
            max_u = U[-1]
            margin = max(10, max_u // 3)
            new_val = rng.randint(0, max_u + margin)
            if new_val != old_val and new_val not in U:
                new_U = list(U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                new_score = _score_set(new_U)
                if new_score > best_score:
                    best_score = new_score
                    best_U = new_U
                    U = new_U
        
        elif op == 'remove' and len(U) > 3:
            i = rng.randint(1, len(U) - 1)
            new_U = U[:i] + U[i+1:]
            new_score = _score_set(new_U)
            if new_score > best_score:
                best_score = new_score
                best_U = new_U
                U = new_U
    
    return best_U, best_score

def construct():
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Golomb rulers of various sizes
    for n in range(3, 22):
        if _elapsed() > _DEADLINE - 80:
            break
        U = _golomb_ruler(n)
        candidates.append(U)
    
    # Prime-based sets
    def primes_up_to(limit):
        sieve = [True] * (limit + 1)
        if limit >= 0:
            sieve[0] = False
        if limit >= 1:
            sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    primes = primes_up_to(200)
    for k in range(2, 15):
        if k <= len(primes):
            U = [0] + primes[:k]
            candidates.append(U)
    
    # Structured patterns
    def generate_pattern(steps, k):
        U = [0]
        for i in range(k - 1):
            if i < len(steps):
                U.append(U[-1] + steps[i])
            else:
                U.append(U[-1] + steps[-1])
        return U
    
    for steps in [[1, 3, 2], [1, 2, 3], [2, 3, 1], [1, 4, 2, 3], [1, 1, 3, 2]]:
        for k in range(3, 15):
            candidates.append(generate_pattern(steps, k))
    
    # Powers of 2
    for k in range(2, 12):
        candidates.append([0] + [2**i for i in range(k)])
    
    # Fibonacci-like
    def fib_set(k):
        U = [0, 1]
        for i in range(2, k):
            U.append(U[-1] + U[-2])
        return U
    
    for k in range(3, 15):
        candidates.append(fib_set(k))
    
    # Random sparse sets
    rng = np.random.RandomState(42)
    for max_val in [20, 40, 80, 160, 320, 600]:
        for size in range(3, min(12, max_val // 5 + 1)):
            for _ in range(3):
                if _elapsed() > _DEADLINE - 70:
                    break
                cands = rng.choice(range(1, max_val + 1), size=size, replace=False)
                U = sorted([0] + list(cands))
                candidates.append(U)
    
    # Evaluate all candidates
    for U in candidates:
        if _elapsed() > _DEADLINE - 60:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Phase 2: Exhaustive search over small sets (key insight from findings)
    if _elapsed() < _DEADLINE - 50:
        ex_U, ex_score = _exhaustive_small_search(max_val=18, min_size=3, max_size=8)
        if ex_U and ex_score > best_score:
            best_score = ex_score
            best_U = ex_U
    
    # Phase 3: Hill climbing from best candidate
    if _elapsed() < _DEADLINE - 30:
        hc_U, hc_score = _hill_climbing(list(best_U), time_limit=20.0)
        if hc_score > best_score:
            best_score = hc_score
            best_U = hc_U
    
    # Phase 4: Try hill climbing from other good candidates
    if _elapsed() < _DEADLINE - 20:
        # Take top few candidates and hill climb from them
        scored_candidates = []
        for U in candidates[:50]:
            s = _score_set(U)
            scored_candidates.append((s, U))
        scored_candidates.sort(key=lambda x: -x[0])
        
        for s, U in scored_candidates[:5]:
            if _elapsed() > _DEADLINE - 15:
                break
            if s < best_score - 0.01:
                break
            hc_U, hc_score = _hill_climbing(list(U), time_limit=5.0)
            if hc_score > best_score:
                best_score = hc_score
                best_U = hc_U
    
    # Phase 5: Final polish with local search
    if _elapsed() < _DEADLINE - 10:
        U, score = _local_search_add(list(best_U), time_limit=5.0)
        if score > best_score:
            best_score = score
            best_U = U
        
        U, score = _local_search_swap(list(best_U), time_limit=5.0)
        if score > best_score:
            best_score = score
            best_U = U
        
        U, score = _local_search_remove(list(best_U), time_limit=3.0)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)
    
    return sorted(best_U)
