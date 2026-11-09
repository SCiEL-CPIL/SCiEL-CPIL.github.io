def construct():
    import time
    from collections import defaultdict
    
    N = 123457
    start_time = time.time()
    DEADLINE = 85  # Safety margin
    
    # Generate k-smooth numbers for increasing k
    # A k-smooth number has all prime factors <= k
    # We want to find the smallest k such that there exist a, b with a+b=N,
    # both k-smooth. Then max(P(a),P(b)) <= k.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    def generate_smooth(prime_list, limit):
        """Generate all numbers <= limit whose prime factors are in prime_list."""
        result = set()
        result.add(1)
        # Use a priority queue approach
        import heapq
        pq = [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]  # (value, e2, e3, e5, e7, e11, e13, e17, e19, e23, e29, e31)
        n_primes = len(prime_list)
        # Actually, let's use a simpler nested approach for small prime lists
        # For primes [2,3,5], we can do triple nested loops
        if n_primes == 1:
            p = prime_list[0]
            val = 1
            while val <= limit:
                result.add(val)
                val *= p
        elif n_primes == 2:
            p1, p2 = prime_list
            v1 = 1
            while v1 <= limit:
                v2 = 1
                while v1 * v2 <= limit:
                    result.add(v1 * v2)
                    v2 *= p2
                v1 *= p1
        elif n_primes == 3:
            p1, p2, p3 = prime_list
            v1 = 1
            while v1 <= limit:
                v2 = 1
                while v1 * v2 <= limit:
                    v3 = 1
                    while v1 * v2 * v3 <= limit:
                        result.add(v1 * v2 * v3)
                        v3 *= p3
                    v2 *= p2
                v1 *= p1
        else:
            # General case using recursive generation
            def _gen(idx, current, limit):
                if idx == n_primes:
                    result.add(current)
                    return
                p = prime_list[idx]
                val = 1
                while current * val <= limit:
                    _gen(idx + 1, current * val, limit)
                    val *= p
            _gen(0, 1, limit)
        return result
    
    # Try increasing sets of primes
    # Start with {2}, {2,3}, {2,3,5}, {2,3,5,7}, ...
    for num_primes in range(1, len(primes) + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        prime_subset = primes[:num_primes]
        smooth_set = generate_smooth(prime_subset, N)
        
        # Check if there's a pair (a, N-a) both in smooth_set
        found = False
        best_a = 1
        best_b = N - 1
        best_max_pf = float('inf')
        
        for a in smooth_set:
            if a > N:
                continue
            b = N - a
            if b < 1:
                continue
            if b in smooth_set:
                # Both are smooth, so max prime factor <= prime_subset[-1]
                # But we need the actual max prime factor
                # Since both are in the smooth set, P(a) <= prime_subset[-1] and P(b) <= prime_subset[-1]
                # The actual max could be less, but we know it's <= prime_subset[-1]
                found = True
                best_a = a
                best_b = b
                break
        
        if found:
            return [best_a, best_b]
    
    # Fallback: use the original sieve approach if no smooth pair found within deadline
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        if current_max < best_max_pf:
            best_max_pf = current_max
            best_a = a
            best_b = b
            if best_max_pf == 2:
                break
    
    return [best_a, best_b]
