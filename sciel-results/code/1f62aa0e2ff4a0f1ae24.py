import time
import numpy as np

def _max_discrepancy(signs):
    """Compute max absolute signed sum over all APs in 0-indexed array of length n."""
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int8)
    max_abs = 1  # at least 1 from singletons
    # For each step d from 1 to n-1
    for d in range(1, n):
        # For each residue class r in 0..d-1
        for r in range(d):
            # AP: r, r+d, r+2d, ... < n
            idx = list(range(r, n, d))
            if len(idx) < 2:
                continue
            s = signs[idx]
            # Compute all prefix sums
            ps = np.cumsum(s)
            m = int(np.max(np.abs(ps)))
            if m > max_abs:
                max_abs = m
                if max_abs >= 3:  # early exit if already bad
                    return max_abs
    return max_abs

def _local_search(signs, deadline, max_flips=200000):
    """Hill-climbing local search: flip one sign at a time."""
    n = len(signs)
    signs = list(signs)
    current_max = _max_discrepancy(signs)
    if current_max <= 1:
        return signs, current_max
    
    flips = 0
    while flips < max_flips:
        if time.time() > deadline:
            break
        improved = False
        # Try flipping each position
        for i in range(n):
            old = signs[i]
            signs[i] = -old
            new_max = _max_discrepancy(signs)
            if new_max < current_max:
                current_max = new_max
                improved = True
                flips += 1
                if current_max <= 1:
                    return signs, current_max
                break
            else:
                signs[i] = old
        if not improved:
            break
    return signs, current_max

def _sa_search(signs, deadline, max_iters=100000):
    """Simulated annealing."""
    n = len(signs)
    signs = list(signs)
    current_max = _max_discrepancy(signs)
    best = list(signs)
    best_max = current_max
    if best_max <= 1:
        return best, best_max
    
    temp = 1.0
    cooling = 0.999
    iters = 0
    while iters < max_iters:
        if time.time() > deadline:
            break
        i = np.random.randint(n)
        signs[i] = -signs[i]
        new_max = _max_discrepancy(signs)
        delta = new_max - current_max
        if delta <= 0 or np.random.random() < np.exp(-delta / max(temp, 0.01)):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best = list(signs)
                if best_max <= 1:
                    return best, best_max
        else:
            signs[i] = -signs[i]
        temp *= cooling
        iters += 1
    return best, best_max

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    n = 256
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Thue-Morse sequence
    tm = [(-1) ** bin(i).count('1') for i in range(n)]
    candidates.append(tm)
    
    # 2. Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(alt)
    
    # 3. Period-3: +1, +1, -1
    p3 = [1 if i % 3 != 2 else -1 for i in range(n)]
    candidates.append(p3)
    
    # 4. Period-4: +1, -1, +1, -1 (same as alternating)
    # 5. Period-4: +1, +1, -1, -1
    p4 = [1 if i % 4 < 2 else -1 for i in range(n)]
    candidates.append(p4)
    
    # 6. Rademacher-like: sign of sin
    for k in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]:
        seq = [1 if np.sin(2 * np.pi * k * i / n) >= 0 else -1 for i in range(n)]
        candidates.append(seq)
    
    # 7. Random candidates
    rng = np.random.RandomState(42)
    for seed in range(20):
        rng = np.random.RandomState(seed)
        candidates.append(list(rng.choice([-1, 1], size=n)))
    
    # Evaluate all candidates and pick best
    best_signs = None
    best_max = float('inf')
    for cand in candidates:
        m = _max_discrepancy(cand)
        if m < best_max:
            best_max = m
            best_signs = list(cand)
        if best_max <= 1:
            return best_signs
    
    # Local search on best candidate
    if time.time() < deadline - 10:
        best_signs, best_max = _local_search(best_signs, deadline - 10)
        if best_max <= 1:
            return best_signs
    
    # If still not at 1, try SA on a few diverse starting points
    if time.time() < deadline - 5:
        for cand in candidates[:10]:
            if time.time() > deadline - 5:
                break
            sa_signs, sa_max = _sa_search(cand, deadline - 5, max_iters=50000)
            if sa_max < best_max:
                best_max = sa_max
                best_signs = sa_signs
            if best_max <= 1:
                return best_signs
    
    # Final local search on best
    if time.time() < deadline:
        best_signs, best_max = _local_search(best_signs, deadline, max_flips=100000)
    
    return best_signs
