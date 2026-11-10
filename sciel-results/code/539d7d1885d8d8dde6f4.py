import time
import random
import math

def _build_spf(limit):
    """Build smallest-prime-factor table up to limit."""
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _prime_factors_spf(n, spf):
    """Return set of distinct prime factors of n using SPF table."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _count_distinct_primes(arr, spf):
    """Count distinct primes in all pairwise sums of distinct elements."""
    all_primes = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            s = ai + arr[j]
            all_primes |= _prime_factors_spf(s, spf)
    return len(all_primes)

def _local_search(arr, spf, deadline, rng):
    """Swap-based local search to minimize distinct prime count."""
    n = len(arr)
    current = list(arr)
    best_count = _count_distinct_primes(current, spf)
    best_arr = list(current)
    
    # Precompute current prime set for incremental updates
    # For speed, we'll recompute fully each time (66 sums, each fast with SPF)
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve >= max_no_improve:
            break
        
        improved = False
        # Try random swaps
        for _ in range(200):
            if time.time() > deadline:
                break
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            # Swap
            current[i], current[j] = current[j], current[i]
            new_count = _count_distinct_primes(current, spf)
            if new_count < best_count:
                best_count = new_count
                best_arr = list(current)
                improved = True
                no_improve = 0
            else:
                # Revert
                current[i], current[j] = current[j], current[i]
                no_improve += 1
        
        if not improved and no_improve >= max_no_improve:
            break
    
    return best_arr, best_count

def _try_smooth_clique(primes, spf, deadline):
    """
    Try to find 12 integers whose pairwise sums are all P-smooth.
    Uses backtracking on a precomputed compatibility graph.
    """
    L = 300  # search space
    # Precompute smoothness
    prime_set = set(primes)
    
    def is_smooth(n):
        temp = n
        for p in primes:
            while temp % p == 0:
                temp //= p
        return temp == 1
    
    # Build compatibility: for each i in 1..L, which j > i have smooth sum
    valid = [set() for _ in range(L + 1)]
    for i in range(1, L + 1):
        for j in range(i + 1, L + 1):
            if is_smooth(i + j):
                valid[i].add(j)
                valid[j].add(i)
    
    # Sort by degree
    nodes = list(range(1, L + 1))
    nodes.sort(key=lambda x: len(valid[x]), reverse=True)
    
    target = 12
    clique = []
    
    def backtrack(start_idx, current):
        if len(current) == target:
            return current[:]
        if len(current) + (len(nodes) - start_idx) < target:
            return None
        for k in range(start_idx, len(nodes)):
            node = nodes[k]
            if all(node in valid[c] for c in current):
                res = backtrack(k + 1, current + [node])
                if res:
                    return res
        return None
    
    result = backtrack(0, [])
    if result:
        max_x = max(result)
        k = 0
        while max_x * (2 ** (k + 1)) <= 10000:
            k += 1
        scale = 2 ** k
        a_set = [x * scale for x in result]
        if len(set(a_set)) == 12 and all(1 <= a <= 10000 for a in a_set):
            return a_set
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    # Build SPF table
    spf = _build_spf(20000)
    rng = random.Random(42)
    
    best_arr = None
    best_count = float('inf')
    
    # Phase 1: Try smooth clique approach for small prime sets
    # This can give very low prime counts (2-4)
    small_prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 5],
        [2, 7],
        [3, 5],
        [2, 3, 11],
        [2, 3, 13],
        [2, 3, 5, 11],
    ]
    
    for P in small_prime_sets:
        if time.time() > deadline:
            break
        result = _try_smooth_clique(P, spf, deadline)
        if result:
            count = _count_distinct_primes(result, spf)
            if count < best_count:
                best_count = count
                best_arr = list(result)
    
    # Phase 2: Local search with various initializations
    # Targeted initializations
    initializations = []
    
    # Multiples of 6 (divisible by 2 and 3)
    init1 = [6 * i for i in range(1, 13)]  # 6, 12, 18, ..., 72
    initializations.append(init1)
    
    # Even numbers
    init2 = [2 * i for i in range(1, 13)]  # 2, 4, 6, ..., 24
    initializations.append(init2)
    
    # Multiples of 12
    init3 = [12 * i for i in range(1, 13)]  # 12, 24, ..., 144
    initializations.append(init3)
    
    # Powers of 2
    init4 = [2 ** i for i in range(1, 13)]  # 2, 4, 8, ..., 4096
    initializations.append(init4)
    
    # Multiples of 30 (2*3*5)
    init5 = [30 * i for i in range(1, 13)]  # 30, 60, ..., 360
    initializations.append(init5)
    
    # Random even numbers
    init6 = rng.sample(range(2, 10001, 2), 12)
    initializations.append(init6)
    
    # Random multiples of 6
    init7 = rng.sample(range(6, 10001, 6), 12)
    initializations.append(init7)
    
    # Random multiples of 12
    init8 = rng.sample(range(12, 10001, 12), 12)
    initializations.append(init8)
    
    # Random small numbers
    init9 = rng.sample(range(1, 200), 12)
    initializations.append(init9)
    
    # Random numbers up to 500
    init10 = rng.sample(range(1, 501), 12)
    initializations.append(init10)
    
    # Try each initialization with local search
    for init in initializations:
        if time.time() > deadline:
            break
        # Ensure distinct and in range
        if len(set(init)) != 12 or any(x < 1 or x > 10000 for x in init):
            continue
        
        # Quick evaluation
        count = _count_distinct_primes(init, spf)
        if count < best_count:
            best_count = count
            best_arr = list(init)
        
        # Local search
        arr, lc = _local_search(init, spf, deadline, rng)
        if lc < best_count:
            best_count = lc
            best_arr = list(arr)
    
    # Phase 3: More random restarts with local search
    remaining_deadline = deadline - time.time()
    if remaining_deadline > 2.0:
        num_restarts = int(remaining_deadline / 1.0)  # ~1s per restart
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
            # Random initialization with bias toward smooth numbers
            candidates = list(range(1, 10001))
            # Bias: prefer numbers with small prime factors
            def smooth_score(x):
                temp = x
                score = 0
                for p in [2, 3, 5, 7, 11, 13]:
                    while temp % p == 0:
                        score += 1
                        temp //= p
                if temp > 1:
                    score -= 2  # penalty for large prime factor
                return score
            
            # Sample with bias
            weights = [smooth_score(x) + 1 for x in candidates]
            total_w = sum(weights)
            # Just pick random, but prefer smaller numbers
            init = rng.sample(range(1, 500), 12)
            
            arr, lc = _local_search(init, spf, deadline, rng)
            if lc < best_count:
                best_count = lc
                best_arr = list(arr)
    
    # Final: ensure we have a valid answer
    if best_arr is None:
        best_arr = list(range(1, 13))
    
    # Verify
    assert len(best_arr) == 12
    assert len(set(best_arr)) == 12
    assert all(1 <= x <= 10000 for x in best_arr)
    
    return best_arr
