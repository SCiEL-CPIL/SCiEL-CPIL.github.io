"""Rewritten program: structural change to a targeted construction + local search."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _build_spf(limit):
    """Build smallest-prime-factor table up to limit."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _factor_set(n, spf):
    """Return set of prime factors of n using spf table."""
    s = set()
    while n > 1:
        p = spf[n]
        s.add(p)
        while n % p == 0:
            n //= p
    return s

def _pairwise_prime_union(arr, spf):
    """Return the set of all distinct prime factors of pairwise sums of distinct-index pairs."""
    ps = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            s = ai + arr[j]
            while s > 1:
                p = spf[s]
                ps.add(p)
                while s % p == 0:
                    s //= p
    return ps

def _score(arr, spf):
    """Count distinct prime factors across all pairwise sums."""
    return len(_pairwise_prime_union(arr, spf))

def _score_delta_replace(arr, idx, new_val, spf, current_union):
    """
    Compute the new prime set after replacing arr[idx] with new_val.
    Returns (new_union, new_score).
    We remove primes that only appeared via pairs involving idx, then add new pairs.
    """
    n = len(arr)
    old_val = arr[idx]
    if new_val == old_val:
        return current_union, len(current_union)
    
    # We need to figure out which primes in current_union are ONLY from pairs involving idx.
    # This is expensive to do exactly. Instead, rebuild the union from scratch but
    # only for pairs NOT involving idx, then add new pairs involving idx.
    
    # Primes from pairs not involving idx
    ps_without_idx = set()
    for i in range(n):
        if i == idx:
            continue
        ai = arr[i]
        for j in range(i + 1, n):
            if j == idx:
                continue
            s = ai + arr[j]
            while s > 1:
                p = spf[s]
                ps_without_idx.add(p)
                while s % p == 0:
                    s //= p
    
    # Primes from new pairs involving idx
    for j in range(n):
        if j == idx:
            continue
        s = new_val + arr[j]
        while s > 1:
            p = spf[s]
            ps_without_idx.add(p)
            while s % p == 0:
                s //= p
    
    return ps_without_idx, len(ps_without_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    
    MAX_VAL = 10000
    # We need spf up to 20000 (max sum of two values <= 10000)
    spf = _build_spf(20000)
    
    random.seed(12345)
    
    best_set = None
    best_score = float('inf')
    best_union = set()
    
    def try_update(arr):
        nonlocal best_set, best_score, best_union
        arr_sorted = sorted(arr)
        sc = _score(arr_sorted, spf)
        if sc < best_score:
            best_score = sc
            best_set = arr_sorted[:]
            best_union = _pairwise_prime_union(arr_sorted, spf)
            return True
        return False
    
    # === Phase 1: Targeted constructions ===
    
    # Strategy: Use numbers of the form a_i = k * d + r_i where all sums are
    # structured to have few prime factors.
    
    # Key insight: If we pick all numbers to be multiples of a common factor d,
    # then all pairwise sums are multiples of d. The primes dividing d are always
    # present. We want d to have few prime factors, and the reduced sums to also
    # have few prime factors.
    
    # Even better: pick d such that d has exactly one prime factor (a prime power),
    # and the reduced values are chosen so their pairwise sums only involve a small
    # set of primes.
    
    # Let's try: all numbers are multiples of 2, 3, 5, 6, 10, etc.
    # For multiples of 6: sums are multiples of 6, so primes 2 and 3 always present.
    # Reduced values r_i: need r_i + r_j to have few prime factors.
    
    # Even stronger: use the fact that if we pick r_i from {1,2,3,...} such that
    # r_i + r_j only involves primes from a small set.
    
    # Let's try a specific construction:
    # Pick d = 2^a * 3^b (so primes 2,3 are always in the union)
    # Then reduced values should have pairwise sums with only primes from a small set.
    
    # Construction: Let reduced values be powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    # Sums of powers of 2: 1+2=3, 1+4=5, 2+4=6, 1+8=9, 2+8=10, 4+8=12, etc.
    # These involve primes 2,3,5,7,11,13,... not great.
    
    # Better: use reduced values that are all congruent mod something.
    
    # Let's try: reduced values = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    # Sums range from 3 to 23. Primes: 2,3,5,7,11,13,17,19,23. That's 9 primes.
    # Plus primes of d.
    
    # Try d=6 (primes 2,3). Reduced 1..12: sums 3..23, primes {2,3,5,7,11,13,17,19,23}.
    # Total: {2,3,5,7,11,13,17,19,23} = 9 primes.
    
    # Can we do better? What if reduced values are chosen to minimize prime variety?
    
    # Try reduced values all even: 2,4,6,8,10,12,14,16,18,20,22,24
    # Sums: 6..46, all even. Primes of sums: 2,3,5,7,11,13,17,19,23. Still 9.
    
    # Try reduced values: 1,3,5,7,9,11,13,15,17,19,21,23 (odd numbers)
    # Sums: 4..44, all even. Primes: 2,3,5,7,11,13,17,19,23. Still 9.
    
    # The issue is that sums of small integers cover a range that hits many primes.
    
    # What if we use larger reduced values with structure?
    # E.g., reduced values = 1, 1+2, 1+4, 1+6, ... no, they need to be distinct.
    
    # Alternative: use a geometric-like sequence where sums are "nice".
    # r_i = 2^i for i=0..11: 1,2,4,8,16,32,64,128,256,512,1024,2048
    # Sums: 3,5,6,9,10,12,17,18,20,24,33,34,36,40,48,65,66,68,72,80,129,130,132,136,144,193,194,196,200,208,257,258,260,264,272,385,386,388,392,400,513,514,516,520,528,769,770,772,776,784,1025,1026,1028,1032,1040
    # Let me factor some: 3,5,6=2*3,9=3^2,10=2*5,12=2^2*3,17,18=2*3^2,20=2^2*5,24=2^3*3,33=3*11,34=2*17,36=2^2*3^2,40=2^3*5,48=2^4*3,65=5*13,66=2*3*11,68=2^2*17,72=2^3*3^2,80=2^4*5,129=3*43,130=2*5*13,132=2^2*3*11,136=2^3*17,144=2^4*3^2,193,194=2*97,196=2^2*7^2,200=2^3*5^2,208=2^4*13,257,258=2*3*43,260=2^2*5*13,264=2^3*3*11,272=2^4*17,385=5*7*11,386=2*193,388=2^2*97,392=2^3*7^2,400=2^4*5^2,513=3^3*19,514=2*257,516=2^2*3*43,520=2^4*5*13,528=2^4*3*11,769,770=2*5*7*11,772=2^2*193,776=2^3*97,784=2^4*7^2,1025=5^2*41,1026=2*3^3*19,1028=2^2*257,1032=2^3*3*43,1040=2^5*5*13
    # Primes: 2,3,5,7,11,13,17,19,41,43,97,193,257. That's 13 primes. Worse.
    
    # Let's try a different approach: use local search from multiple good starting points.
    
    # Starting point 1: multiples of 6
    d = 6
    max_r = MAX_VAL // d
    if max_r >= 12:
        init = [d * r for r in range(1, 13)]
        try_update(init)
    
    # Starting point 2: multiples of 10
    d = 10
    max_r = MAX_VAL // d
    if max_r >= 12:
        init = [d * r for r in range(1, 13)]
        try_update(init)
    
    # Starting point 3: multiples of 30
    d = 30
    max_r = MAX_VAL // d
    if max_r >= 12:
        init = [d * r for r in range(1, 13)]
        try_update(init)
    
    # Starting point 4: multiples of 2
    d = 2
    max_r = MAX_VAL // d
    if max_r >= 12:
        init = [d * r for r in range(1, 13)]
        try_update(init)
    
    # Starting point 5: random multiples of 6
    for _ in range(20):
        d = 6
        max_r = MAX_VAL // d
        if max_r >= 12:
            rs = random.sample(range(1, max_r + 1), 12)
            init = [d * r for r in sorted(rs)]
            try_update(init)
    
    # Starting point 6: random multiples of 30
    for _ in range(20):
        d = 30
        max_r = MAX_VAL // d
        if max_r >= 12:
            rs = random.sample(range(1, max_r + 1), 12)
            init = [d * r for r in sorted(rs)]
            try_update(init)
    
    # Starting point 7: random even numbers
    for _ in range(20):
        evens = list(range(2, MAX_VAL + 1, 2))
        init = sorted(random.sample(evens, 12))
        try_update(init)
    
    # Starting point 8: numbers of form 2^k * 3^m
    # Generate candidates
    candidates = set()
    for k in range(0, 14):
        for m in range(0, 10):
            v = (2 ** k) * (3 ** m)
            if 1 <= v <= MAX_VAL:
                candidates.add(v)
    candidates = sorted(candidates)
    if len(candidates) >= 12:
        # Try various subsets
        for _ in range(50):
            init = sorted(random.sample(candidates, 12))
            try_update(init)
    
    # === Phase 2: Local search ===
    
    def local_search(arr, spf, deadline, max_iters=5000):
        """Hill-climbing local search: try replacing one element at a time."""
        arr = sorted(arr[:])
        current_score = _score(arr, spf)
        current_union = _pairwise_prime_union(arr, spf)
        n = len(arr)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Shuffle order to avoid bias
            indices = list(range(n))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                
                old_val = arr[idx]
                # Generate candidate replacements
                # Try values near old_val, and also random values with same structure
                candidates_to_try = set()
                
                # Nearby values
                for delta in range(-50, 51):
                    nv = old_val + delta
                    if 1 <= nv <= MAX_VAL and nv != old_val:
                        candidates_to_try.add(nv)
                
                # Random values
                for _ in range(20):
                    candidates_to_try.add(random.randint(1, MAX_VAL))
                
                # Values with same gcd structure
                g = old_val
                for k in range(1, 100):
                    nv = g * k
                    if 1 <= nv <= MAX_VAL and nv != old_val:
                        candidates_to_try.add(nv)
                
                best_new_val = None
                best_new_score = current_score
                best_new_union = None
                
                for nv in candidates_to_try:
                    if nv in arr:
                        continue
                    
                    # Quick check: compute new union
                    new_union, new_score = _score_delta_replace(arr, idx, nv, spf, current_union)
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = nv
                        best_new_union = new_union
                
                if best_new_val is not None:
                    arr[idx] = best_new_val
                    arr.sort()
                    current_score = best_new_score
                    current_union = best_new_union
                    improved = True
                    break  # Restart the index loop
        
            if not improved:
                break
        
        return arr, current_score
    
    # Run local search from best found so far
    if best_set is not None:
        arr, sc = local_search(best_set, spf, deadline, max_iters=3000)
        try_update(arr)
    
    # Run local search from several random starts
    for trial in range(10):
        if time.time() > deadline:
            break
        
        # Random start with structure
        d = random.choice([2, 3, 5, 6, 10, 15, 30, 60])
        max_r = MAX_VAL // d
        if max_r < 12:
            continue
        rs = random.sample(range(1, max_r + 1), 12)
        init = sorted([d * r for r in rs])
        
        arr, sc = local_search(init, spf, deadline, max_iters=2000)
        try_update(arr)
    
    # === Phase 3: More targeted local search with better moves ===
    
    if time.time() < deadline:
        # Try swap-based local search
        def swap_local_search(arr, spf, deadline, max_iters=2000):
            arr = sorted(arr[:])
            current_score = _score(arr, spf)
            n = len(arr)
            
            for iteration in range(max_iters):
                if time.time() > deadline:
                    break
                
                improved = False
                # Try swapping two elements with random values
                for _ in range(100):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j:
                        continue
                    
                    old_i, old_j = arr[i], arr[j]
                    new_i = random.randint(1, MAX_VAL)
                    new_j = random.randint(1, MAX_VAL)
                    
                    if new_i in arr or new_j in arr or new_i == new_j:
                        continue
                    
                    test_arr = arr[:]
                    test_arr[i] = new_i
                    test_arr[j] = new_j
                    test_arr.sort()
                    
                    sc = _score(test_arr, spf)
                    if sc < current_score:
                        arr = test_arr
                        current_score = sc
                        improved = True
                        break
                
                # Also try single replacement
                if not improved:
                    i = random.randint(0, n - 1)
                    old_i = arr[i]
                    for _ in range(50):
                        new_i = random.randint(1, MAX_VAL)
                        if new_i in arr:
                            continue
                        
                        test_arr = arr[:]
                        test_arr[i] = new_i
                        test_arr.sort()
                        
                        sc = _score(test_arr, spf)
                        if sc < current_score:
                            arr = test_arr
                            current_score = sc
                            improved = True
                            break
                
                if not improved:
                    break
            
            return arr, current_score
        
        if best_set is not None:
            arr, sc = swap_local_search(best_set, spf, deadline, max_iters=1500)
            try_update(arr)
        
        # More random restarts
        for trial in range(15):
            if time.time() > deadline:
                break
            
            d = random.choice([2, 3, 5, 6, 10, 15, 30, 60, 120])
            max_r = MAX_VAL // d
            if max_r < 12:
                continue
            rs = random.sample(range(1, max_r + 1), 12)
            init = sorted([d * r for r in rs])
            
            arr, sc = swap_local_search(init, spf, deadline, max_iters=1000)
            try_update(arr)
    
    # === Phase 4: Final polish ===
    if time.time() < deadline and best_set is not None:
        # Try to improve by replacing elements with values that have few prime factors in sums
        arr = best_set[:]
        current_score = _score(arr, spf)
        n = len(arr)
        
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            improved = False
            i = random.randint(0, n - 1)
            old_i = arr[i]
            
            # Try values that are multiples of common factors
            for d in [2, 3, 5, 6, 10, 15, 30]:
                max_k = MAX_VAL // d
                for _ in range(10):
                    k = random.randint(1, max_k)
                    nv = d * k
                    if nv in arr or nv == old_i:
                        continue
                    
                    test_arr = arr[:]
                    test_arr[i] = nv
                    test_arr.sort()
                    sc = _score(test_arr, spf)
                    if sc < current_score:
                        arr = test_arr
                        current_score = sc
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try random
                for _ in range(20):
                    nv = random.randint(1, MAX_VAL)
                    if nv in arr or nv == old_i:
                        continue
                    
                    test_arr = arr[:]
                    test_arr[i] = nv
                    test_arr.sort()
                    sc = _score(test_arr, spf)
                    if sc < current_score:
                        arr = test_arr
                        current_score = sc
                        improved = True
                        break
        
        try_update(arr)
    
    return best_set

# EVOLVE-BLOCK-END
