def construct():
    import time
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    N = 123457
    
    # Fast path: Check for small prime factor bounds (smooth numbers)
    # If we find a pair where both are k-smooth, max(P(a), P(b)) <= k
    # We check increasing k.
    primes_to_check = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    for k in primes_to_check:
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate all k-smooth numbers up to N
        smooth_nums = set()
        # We can use a recursive or iterative approach
        # Iterative: start with [1], multiply by primes <= k
        # But we need to avoid duplicates and stay <= N
        
        # Efficient generation:
        # For each number in smooth_nums, try multiplying by each prime <= k
        # This is O(size * num_primes)
        
        current_smooths = [1]
        smooth_nums.add(1)
        
        # Simple BFS-like generation
        # Actually, for small k, the count is small.
        # Let's just iterate through all numbers and check? No, that's N.
        # Better to generate.
        
        # Using a priority queue or just list and sort?
        # Let's use a set and a list to process.
        to_process = [1]
        processed = set()
        
        while to_process:
            x = to_process.pop(0) # This is slow, use deque or index
            # Better: just use a list and iterate, adding new ones
            # Since N is small, we can just check all numbers?
            # No, N is 123457. Checking all numbers is 123k ops.
            # Generating smooth numbers is much faster.
            
            # Let's stick to the sieve for the main logic as it's robust.
            # The smooth number check is an optimization if a solution exists.
            # Given the previous score, it's unlikely a very small k solution exists.
            # But let's try up to 11 as per inspiration.
            
            pass # Skipping complex generation for now to keep code simple and reliable.
            # The sieve is fast enough.
            
    # Fallback: Sieve for largest prime factor
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    # Scan all decompositions
    for a in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
        if best_max_pf == 1:
            break
    
    return [best_a, best_b]
