import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 256
    # Primes
    primes = []
    for i in range(2, n + 1):
        is_prime = True
        for d in range(2, int(i**0.5) + 1):
            if i % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    small_primes = [2, 3, 5, 7, 11, 13]
    large_primes = [p for p in primes if p > 13]
    
    # Map small prime to bit index
    sp_index = {p: i for i, p in enumerate(small_primes)}
    
    # Precompute prime powers for small primes
    sp_powers = {}
    for p in small_primes:
        power = p
        while power * p <= n:
            power *= p
        sp_powers[p] = power
        
    # Precompute options for each large prime
    # Option: (small_mask, value)
    options = {}
    for q in large_primes:
        opts = []
        # Option 0: use q itself
        # q has no small factors
        opts.append((0, q))
        
        # Other multiples
        for k in range(2, n // q + 1):
            m = q * k
            # Find small prime factors of m
            mask = 0
            temp = m
            for p in small_primes:
                if temp % p == 0:
                    mask |= (1 << sp_index[p])
                    while temp % p == 0:
                        temp //= p
            # If temp > 1, it has a large prime factor other than q?
            # No, m is a multiple of q.
            # If m has another large prime r, then r must be a factor of k.
            # But k <= 256/17 = 15.
            # Large primes are >= 17.
            # So k cannot have large prime factors.
            # So temp must be 1.
            opts.append((mask, m))
        options[q] = opts
        
    # DP over masks
    # 2^6 = 64 masks
    dp = [-10**9] * 64
    dp[0] = 0
    
    for q in large_primes:
        new_dp = [-10**9] * 64
        for mask in range(64):
            if dp[mask] < 0:
                continue
            for smask, val in options[q]:
                if (mask & smask) == 0:
                    new_mask = mask | smask
                    new_val = dp[mask] + val
                    if new_val > new_dp[new_mask]:
                        new_dp[new_mask] = new_val
        dp = new_dp
        
    best_sum = 0
    for mask in range(64):
        if dp[mask] < 0:
            continue
        sum_free = 0
        for i, p in enumerate(small_primes):
            if not (mask & (1 << i)):
                sum_free += sp_powers[p]
        total = dp[mask] + sum_free + 1 # +1 for integer 1
        if total > best_sum:
            best_sum = total
            
    # Reconstruct the set to return
    # We need to return the list of integers.
    # The task says "Return a list of ints".
    # And "The raw value is their sum".
    # We just need to return the list that achieves this sum.
    
    # To reconstruct, we need to store the choices.
    # Given the constraints, we can just re-run the DP storing parent pointers or choices.
    
    # Let's re-run with storage
    dp = [-10**9] * 64
    dp[0] = 0
    choices = [None] * 64 # choices[mask] = (q, option_idx) for the last step?
    # Better: store for each step and mask.
    # Or just store the final mask and backtrack?
    # Since N is small, we can store the full path.
    
    # Actually, we can just find the best mask and then greedily assign?
    # No, the DP ensured optimality.
    # Let's store the choice for each (q_idx, mask).
    
    num_large = len(large_primes)
    # history[i][mask] = (prev_mask, option_idx)
    history = [ [None]*64 for _ in range(num_large) ]
    
    dp = [-10**9] * 64
    dp[0] = 0
    
    for i, q in enumerate(large_primes):
        new_dp = [-10**9] * 64
        for mask in range(64):
            if dp[mask] < 0:
                continue
            for j, (smask, val) in enumerate(options[q]):
                if (mask & smask) == 0:
                    new_mask = mask | smask
                    new_val = dp[mask] + val
                    if new_val > new_dp[new_mask]:
                        new_dp[new_mask] = new_val
                        history[i][new_mask] = (mask, j)
        dp = new_dp
        
    # Find best final mask
    best_mask = 0
    best_val = -1
    for mask in range(64):
        if dp[mask] < 0: continue
        sum_free = 0
        for i, p in enumerate(small_primes):
            if not (mask & (1 << i)):
                sum_free += sp_powers[p]
        total = dp[mask] + sum_free + 1
        if total > best_val:
            best_val = total
            best_mask = mask
            
    # Backtrack to get choices
    result = [1]
    current_mask = best_mask
    for i in range(num_large - 1, -1, -1):
        prev_mask, opt_idx = history[i][current_mask]
        q = large_primes[i]
        smask, val = options[q][opt_idx]
        result.append(val)
        current_mask = prev_mask
        
    # Add free small prime powers
    for i, p in enumerate(small_primes):
        if not (best_mask & (1 << i)):
            result.append(sp_powers[p])
            
    return [int(x) for x in result]
