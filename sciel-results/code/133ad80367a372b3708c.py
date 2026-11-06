import time
import numpy as np

def _score_set(arr):
    """Compute max(|A+A|, |A*A|) for a numpy array of distinct positive ints."""
    # Sumset A+A
    sums = np.add.outer(arr, arr)
    # Product set A*A
    prods = np.multiply.outer(arr, arr)
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _score_set_fast(arr):
    """Faster scoring using Python sets for small n (n=20)."""
    n = len(arr)
    sums = set()
    prods = set()
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            aj = arr[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin out of 90s
    
    n = 20
    max_val = 10000
    
    # Initial candidates: powers of 2 give small product sets
    # 2^0=1, 2^1=2, ..., 2^19=524288 -> too large
    # Let's use smaller bases or mixed sets
    
    # Strategy: Start with a geometric progression (good for products) 
    # and refine via local search.
    
    # Initial set: powers of 2 up to 2^13 = 8192, then fill in
    # Actually, let's try a few different initializations
    
    best_set = None
    best_score = float('inf')
    
    # Candidate 1: Powers of 2 (1, 2, 4, ..., 2^19 is too big, use 2^0..2^13 and then 14 more)
    # 2^13 = 8192, 2^14 = 16384 > 10000
    # So max power of 2 is 8192. We need 20 distinct.
    # Let's use 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14 elements)
    # Then add 6 more numbers that minimize the objective
    
    # Better initial: geometric progression with ratio close to 1 but > 1
    # Or start with a random set and optimize
    
    # Let's use a smarter initialization: 
    # Products: if A = {a_1 < a_2 < ... < a_n}, then A*A has at least n(n+1)/2 distinct products
    # if all a_i * a_j are distinct. For powers of 2, they are all distinct.
    # |A*A| = n(n+1)/2 = 20*21/2 = 210 for powers of 2.
    # |A+A| for powers of 2: also all distinct, so 210.
    # So max = 210. Can we do better?
    
    # If we use an arithmetic progression, |A+A| = 2n-1 = 39, but |A*A| can be large.
    # For AP {1,2,...,20}: |A+A| = 39, |A*A| = ? 
    # Products 1*1 to 20*20. Many distinct products.
    # Let's compute: for {1,...,20}, products range from 1 to 400.
    # Not all are present. Let's just compute it.
    
    # Strategy: local search with multiple restarts
    # Use simulated annealing or hill climbing
    
    np.random.seed(42)
    
    # Function to generate a random valid set
    def random_set():
        return np.sort(np.random.choice(range(1, max_val+1), size=n, replace=False))
    
    # Function to generate a geometric-like set
    def geo_set(base, ratio, n_elem):
        arr = []
        val = base
        for i in range(n_elem):
            if val > max_val:
                break
            arr.append(int(val))
            val *= ratio
        # Fill remaining with random distinct values
        while len(arr) < n_elem:
            v = np.random.randint(1, max_val+1)
            if v not in arr:
                arr.append(v)
        return np.sort(np.array(arr))
    
    # Initial candidates
    candidates = []
    
    # 1. Arithmetic progression
    candidates.append(np.arange(1, n+1))
    
    # 2. Powers of 2 (truncated)
    p2 = [2**i for i in range(14)]  # up to 8192
    # Add 6 more
    for i in range(6):
        p2.append(8192 + i*100)
    p2 = sorted(set(p2))[:n]
    while len(p2) < n:
        p2.append(np.random.randint(1, max_val+1))
    candidates.append(np.array(sorted(set(p2))))
    
    # 3. Random
    for _ in range(5):
        candidates.append(random_set())
    
    # 4. Geometric with small ratio
    for ratio in [1.1, 1.2, 1.5]:
        candidates.append(geo_set(1, ratio, n))
    
    # Evaluate all candidates
    for cand in candidates:
        if len(set(cand)) != n:
            continue
        if cand.max() > max_val:
            continue
        s = _score_set_fast(cand)
        if s < best_score:
            best_score = s
            best_set = cand.copy()
    
    # Local search: hill climbing with random restarts
    def local_search(arr, max_iters=5000):
        arr = arr.copy()
        current_score = _score_set_fast(arr)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5:
                break
            # Try swapping one element with a random other
            idx = np.random.randint(n)
            old_val = arr[idx]
            new_val = np.random.randint(1, max_val+1)
            if new_val == old_val:
                continue
            if new_val in set(arr):
                continue
            arr[idx] = new_val
            new_score = _score_set_fast(arr)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                arr[idx] = old_val
        return arr, current_score
    
    # Run local search from best candidate
    if best_set is not None:
        best_set, best_score = local_search(best_set, max_iters=10000)
    
    # Try more restarts
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        cand = random_set()
        arr, score = local_search(cand, max_iters=5000)
        if score < best_score:
            best_score = score
            best_set = arr
    
    # Final polish: try small perturbations
    if best_set is not None and time.time() < deadline - 5:
        # Try replacing each element with nearby values
        for i in range(n):
            if time.time() > deadline - 2:
                break
            old_val = best_set[i]
            for delta in range(-50, 51):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 1 or new_val > max_val:
                    continue
                if new_val in set(best_set):
                    continue
                best_set[i] = new_val
                new_score = _score_set_fast(best_set)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_set[i] = old_val
    
    return list(best_set)
