def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Sieve of Eratosthenes to find all primes <= 10000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(n+1) if is_prime[i]]
    
    primes = sieve(10000)
    n = len(primes)
    
    if n <= 1:
        return primes
    
    # dp[i] = list where dp[i][g] = max length of valid subseq ending at primes[i] with last gap = g
    # We'll use a dict or array. Gaps can be up to 10000.
    # Use array of size max_gap+1 for each prime index.
    max_gap = 10000
    
    # dp[i][g] for i in range(n), g in range(max_gap+1)
    # Initialize to 0 (no valid subsequence)
    # dp[i][0] = 1 for all i (single element, gap 0 conceptually)
    
    # Actually, a single prime has no gap. Let's handle base case: length 1.
    # For sequences of length >= 2, the last gap is well-defined.
    
    # Let's define: dp[i][g] = max length of valid subsequence ending at primes[i] 
    #                where the last gap (between second-to-last and last) is exactly g.
    # Base: a single prime has length 1, no gap. We'll handle this separately.
    
    # Better: let's track for each prime i, the best length achievable with each possible last gap.
    # A single prime: length 1, "last gap" = 0 (sentinel).
    
    # dp[i][g] = max length ending at i with last gap g. g=0 means single element.
    
    dp = [[0] * (max_gap + 1) for _ in range(n)]
    for i in range(n):
        dp[i][0] = 1  # single element
    
    # prefix_max[i][g] = max(dp[i][0], ..., dp[i][g])
    prefix_max = [[0] * (max_gap + 1) for _ in range(n)]
    for i in range(n):
        prefix_max[i][0] = dp[i][0]
        for g in range(1, max_gap + 1):
            prefix_max[i][g] = max(prefix_max[i][g-1], dp[i][g])
    
    # Now iterate: for each i, for each j < i, gap = primes[i] - primes[j]
    # dp[i][gap] = max(dp[i][gap], prefix_max[j][gap] + 1)
    # But we need to recompute prefix_max for i after all updates? No, prefix_max is for j only.
    
    # Wait, we update dp[i][gap] using prefix_max[j][gap]. But prefix_max[j] is already computed
    # since j < i. So this works.
    
    # However, we also need to update prefix_max[i] as we go? No, prefix_max[i] is only used
    # when i is the "j" in a future iteration, which happens after we've finished all updates to dp[i].
    
    # Actually, we process i in order. When we process i, we read from all j < i (whose prefix_max 
    # is already final). We update dp[i]. After finishing all j for a given i, we compute prefix_max[i].
    
    # Let me restructure:
    
    # Reset dp
    dp = [[0] * (max_gap + 1) for _ in range(n)]
    for i in range(n):
        dp[i][0] = 1
    
    # Process primes in order
    for i in range(n):
        # For each j < i, update dp[i][gap]
        for j in range(i):
            gap = primes[i] - primes[j]
            if gap > max_gap:
                continue
            # We can extend any sequence ending at j with last gap <= gap
            best_prev = prefix_max[j][gap] if gap <= max_gap else 0
            # Wait, prefix_max[j][gap] = max over g' in [0, gap] of dp[j][g']
            if best_prev > 0:
                new_len = best_prev + 1
                if new_len > dp[i][gap]:
                    dp[i][gap] = new_len
        
        # Now compute prefix_max[i]
        prefix_max[i][0] = dp[i][0]
        for g in range(1, max_gap + 1):
            prefix_max[i][g] = max(prefix_max[i][g-1], dp[i][g])
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
    
    # Find the maximum length and reconstruct
    best_len = 0
    best_i = -1
    best_g = 0
    for i in range(n):
        for g in range(max_gap + 1):
            if dp[i][g] > best_len:
                best_len = dp[i][g]
                best_i = i
                best_g = g
    
    if best_i == -1:
        return primes[:1]
    
    # Reconstruct the subsequence
    # We need to trace back. dp[i][g] means we came from some j with gap = primes[i] - primes[j] = g
    # and the previous last gap g' <= g, with dp[j][g'] = dp[i][g] - 1
    
    result = [primes[best_i]]
    cur_i = best_i
    cur_g = best_g
    cur_len = dp[best_i][best_g]
    
    while cur_len > 1:
        # Find j < cur_i such that primes[cur_i] - primes[j] = cur_g and dp[j][g'] = cur_len - 1 for some g' <= cur_g
        found = False
        for j in range(cur_i - 1, -1, -1):
            gap = primes[cur_i] - primes[j]
            if gap != cur_g:
                continue
            # Check if there exists g' <= cur_g with dp[j][g'] = cur_len - 1
            # We can check prefix_max[j][cur_g] == cur_len - 1 and dp[j][cur_g] might be less
            # Actually we need to find the specific g'
            # Since prefix_max[j][cur_g] = max over g'<=cur_g of dp[j][g'], and we know it's >= cur_len-1
            # We need to find g' such that dp[j][g'] = cur_len - 1 and g' <= cur_g
            for g_prime in range(cur_g, -1, -1):
                if dp[j][g_prime] == cur_len - 1:
                    result.append(primes[j])
                    cur_i = j
                    cur_g = g_prime
                    cur_len -= 1
                    found = True
                    break
            if found:
                break
        if not found:
            # This shouldn't happen if DP is correct
            break
    
    result.reverse()
    return result
