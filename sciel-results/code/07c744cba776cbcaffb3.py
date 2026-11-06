import time

def construct():
    start_time = time.time()
    LIMIT = 10000
    
    # Sieve
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, LIMIT + 1) if is_prime[i]]
    
    # dp[p] is a dict: gap -> length
    # parent[p] is a dict: gap -> prev_prime
    # prev_gap[p] is a dict: gap -> prev_gap_value
    dp = {p: {} for p in primes}
    parent = {p: {} for p in primes}
    prev_gap = {p: {} for p in primes}
    
    # prefix_max_val[p] is a list of size LIMIT+1
    # prefix_max_gap[p] is a list of size LIMIT+1
    prefix_max_val = {}
    prefix_max_gap = {}
    
    # We process primes in increasing order
    # For each p, we need prefix_max for all q < p
    # So we compute prefix_max for q as soon as dp[q] is complete
    
    # Wait, dp[q] is complete when we finish processing all p > q?
    # No, dp[q] depends on primes < q.
    # So when we are at prime p, all primes q < p have their dp[q] fully computed
    # because their dependencies (primes < q) are also < p and thus processed earlier.
    
    # So:
    # 1. Iterate p in primes
    # 2. For each p, dp[p] is empty initially.
    # 3. For each q in primes such that q < p:
    #      g = p - q
    #      val = prefix_max_val[q][g]  # Max length of seq ending at q with gap <= g
    #      if val == 0: # No valid sequence ending at q with gap <= g?
    #          # But q is prime, so seq [q] exists.
    #          # Our logic: dp stores len >= 2.
    #          # If val is 0, it means no dp entry in q has gap <= g.
    #          # We can still form [q, p] of length 2.
    #          new_len = 2
    #          new_parent = q
    #          new_prev_g = 0
    #      else:
    #          new_len = val + 1
    #          new_parent = q
    #          new_prev_g = prefix_max_gap[q][g]
    #      
    #      if new_len > dp[p].get(g, 0):
    #          dp[p][g] = new_len
    #          parent[p][g] = new_parent
    #          prev_gap[p][g] = new_prev_g
    #
    # 4. After inner loop, compute prefix_max for p
    #    prefix_max_val[p] = [0]*(LIMIT+1)
    #    prefix_max_gap[p] = [0]*(LIMIT+1)
    #    max_v = 0
    #    max_g = 0
    #    for g in range(1, LIMIT+1):
    #        if g in dp[p] and dp[p][g] > max_v:
    #            max_v = dp[p][g]
    #            max_g = g
    #        prefix_max_val[p][g] = max_v
    #        prefix_max_gap[p][g] = max_g
    #
    # 5. Track global best
    
    best_len = 0
    best_p = None
    best_g = None
    
    for p in primes:
        # Check time
        if time.time() - start_time > 50:
            break
            
        # Process transitions from q < p
        for q in primes:
            if q >= p:
                break
            g = p - q
            # q is prime, p is prime.
            
            # Get best from q with gap <= g
            # prefix_max_val[q] must exist
            # Since q < p, and we process in order, prefix_max_val[q] is ready.
            
            val = prefix_max_val[q][g]
            
            if val == 0:
                new_len = 2
                new_parent = q
                new_prev_g = 0
            else:
                new_len = val + 1
                new_parent = q
                new_prev_g = prefix_max_gap[q][g]
                
            if new_len > dp[p].get(g, 0):
                dp[p][g] = new_len
                parent[p][g] = new_parent
                prev_gap[p][g] = new_prev_g
                
                if new_len > best_len:
                    best_len = new_len
                    best_p = p
                    best_g = g
                    
        # Compute prefix max for p
        pm_val = [0] * (LIMIT + 1)
        pm_gap = [0] * (LIMIT + 1)
        max_v = 0
        max_g = 0
        # Optimize: only iterate gaps that exist?
        # But we need pm[g] for all g because future p's will query any g.
        # So we must fill the whole array.
        for g in range(1, LIMIT + 1):
            if g in dp[p]:
                if dp[p][g] > max_v:
                    max_v = dp[p][g]
                    max_g = g
            pm_val[g] = max_v
            pm_gap[g] = max_g
            
        prefix_max_val[p] = pm_val
        prefix_max_gap[p] = pm_gap

    # Reconstruct
    if best_p is None:
        return [2]
        
    seq = [best_p]
    cur_p = best_p
    cur_g = best_g
    while cur_g > 0:
        prev_p = parent[cur_p][cur_g]
        seq.append(prev_p)
        cur_g = prev_gap[cur_p][cur_g]
        cur_p = prev_p
        
    seq.reverse()
    return seq
