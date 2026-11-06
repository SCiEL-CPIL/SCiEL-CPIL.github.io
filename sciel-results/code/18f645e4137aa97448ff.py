import time
import random
import numpy as np

def _score(A):
    n = len(A)
    sums = set()
    prods = set()
    # Using a list and converting to set might be faster than inserting one by one?
    # Or just insert.
    for i in range(n):
        ai = A[i]
        for j in range(n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _is_valid(A, n=20, maximum=10000):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_A = list(range(1, 21))
    best_score = _score(best_A)
    
    rng = random.Random(42)
    
    def try_candidate(A):
        nonlocal best_A, best_score
        if not _is_valid(A, 20, 10000):
            return
        s = _score(A)
        if s < best_score:
            best_score = s
            best_A = sorted(A)
            
    # Phase 1: Diverse Initial Candidates
    # 1. Arithmetic
    try_candidate(list(range(1, 21)))
    try_candidate(list(range(1, 41, 2))) # 1, 3, ..., 39
    
    # 2. Geometric
    A_geo = [2**i for i in range(14)] # 1 to 8192
    # Fill remaining 6 with large numbers? Or small numbers?
    # If we fill with small numbers, sum set grows.
    # If we fill with large numbers, product set grows.
    # Let's try filling with numbers close to the last one.
    last = A_geo[-1]
    for i in range(6):
        A_geo.append(last * 2)
        if A_geo[-1] > 10000:
             A_geo[-1] = 10000 - i # Just to make them distinct and valid, but this is bad
    # Better: just take a subset of geometric that fits, and fill with arithmetic tail?
    # Let's just use a pure geometric if it fits, else scale it.
    
    # Scaled Geometric
    for scale in [1, 2, 3]:
        A = [scale * (2**i) for i in range(14)]
        if max(A) <= 10000:
             # Fill 6 more
             # Try to keep them close to maintain structure
             # Just use random for now in this phase
             pass

    # Let's use a smarter initial candidate: 
    # A set that is "multiplicatively dense" but "additively sparse"?
    # Or vice versa.
    
    # Let's run a local search from multiple starting points.
    
    def local_search(A, max_iters=5000):
        A = sorted(A[:])
        current_score = _score(A)
        improved = True
        it = 0
        while improved and it < max_iters and time.time() < deadline - 2:
            improved = False
            it += 1
            # Randomly pick an index to perturb
            i = rng.randint(0, 19)
            # Try small deltas
            deltas = rng.sample([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], 3)
            for d in deltas:
                new_A = A[:]
                new_A[i] += d
                if 1 <= new_A[i] <= 10000 and len(set(new_A)) == 20:
                    s = _score(new_A)
                    if s < current_score:
                        current_score = s
                        A = new_A
                        improved = True
                        break
        return A, current_score

    # Start points
    start_points = [
        list(range(1, 21)),
        [2**i for i in range(13)] + [8193, 8194, 8195, 8196, 8197, 8198], # 13 powers + 6
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 100, 200, 300, 400, 500, 600],
        list(range(1, 10001, 500))[:20] # Large gaps
    ]
    
    for sp in start_points:
        if len(sp) == 20:
            try_candidate(sp)
            if time.time() < deadline - 5:
                improved_A, improved_score = local_search(sp, max_iters=2000)
                if improved_score < best_score:
                    best_score = improved_score
                    best_A = improved_A

    # Random restarts
    while time.time() < deadline - 5:
        A = rng.sample(range(1, 10001), 20)
        try_candidate(A)
        if time.time() < deadline - 5:
             improved_A, improved_score = local_search(A, max_iters=500)
             if improved_score < best_score:
                 best_score = improved_score
                 best_A = improved_A

    return sorted(best_A)
