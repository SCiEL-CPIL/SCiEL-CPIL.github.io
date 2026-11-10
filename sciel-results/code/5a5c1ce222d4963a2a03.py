import time
import numpy as np

def _is_prime_sieve(n):
    if n < 2:
        return np.zeros(n + 1, dtype=bool)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime

def construct():
    deadline = time.time() + 50  # Safety margin
    
    N_LIMIT = 10000
    is_prime = _is_prime_sieve(N_LIMIT)
    primes = [i for i in range(2, N_LIMIT + 1) if is_prime[i]]
    n_primes = len(primes)
    
    if n_primes == 0:
        return []
        
    max_gap = N_LIMIT
    # dp[i][g] = max length ending at i with last gap g
    dp = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    # PMax[i][g] = max_{h<=g} dp[i][h]
    PMax = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    # argmax_PMax[i][g] = argmax_{h<=g} dp[i][h]
    argmax_PMax = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    # parent tracking
    parent_i = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    parent_gap = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    
    # primes array for fast access
    p_arr = np.array(primes, dtype=np.int32)
    
    for j in range(n_primes):
        if time.time() > deadline:
            break
            
        # For each i < j
        # gap = p_arr[j] - p_arr[i]
        # We can vectorize the gap calculation for all i < j
        i_indices = np.arange(j)
        gaps = p_arr[j] - p_arr[i_indices]
        
        # For each i, we need PMax[i][gaps[i]]
        # This is a gather operation: PMax[i_indices, gaps]
        # But PMax is 2D.
        # vals = PMax[i_indices, gaps]
        
        # However, we also need to update dp[j][gap] with max(2, vals + 1)
        # And update parents.
        
        # Since j is fixed, we iterate i.
        # Vectorizing the inner loop over i is tricky because we need to update
        # dp[j][gap] which might have multiple i's contributing to the same gap?
        # No, gap = p_j - p_i. Since p_i are distinct, gaps are distinct for fixed j.
        # So for a fixed j, each i gives a unique gap.
        # Thus, we can process all i < j in parallel for a fixed j?
        # We need to compute candidate values for all i.
        
        # Let's try vectorized approach for the inner loop.
        # i_range = np.arange(j)
        # gaps = p_arr[j] - p_arr[i_range]
        # prev_lens = PMax[i_range, gaps]
        # cands = np.where(prev_lens > 0, prev_lens + 1, 2)
        
        # Now we need to update dp[j][gaps] with cands.
        # dp[j, gaps] = np.maximum(dp[j, gaps], cands)
        
        # But we also need to update parent_i and parent_gap ONLY if cands > old_dp.
        # mask = cands > dp[j, gaps]
        # dp[j, gaps[mask]] = cands[mask]
        # parent_i[j, gaps[mask]] = i_range[mask]
        # parent_gap[j, gaps[mask]] = argmax_PMax[i_range[mask], gaps[mask]]
        
        if j > 0:
            i_range = np.arange(j)
            gaps = p_arr[j] - p_arr[i_range]
            
            # Get previous max lengths
            # Note: PMax[i][g] is available for all i < j
            prev_lens = PMax[i_range, gaps]
            
            # Calculate candidates
            cands = np.where(prev_lens > 0, prev_lens + 1, 2)
            
            # Current dp values for these gaps
            current_dp = dp[j, gaps]
            
            # Find where candidate is better
            mask = cands > current_dp
            
            if np.any(mask):
                # Update dp
                dp[j, gaps[mask]] = cands[mask]
                
                # Update parents
                parent_i[j, gaps[mask]] = i_range[mask]
                parent_gap[j, gaps[mask]] = argmax_PMax[i_range[mask], gaps[mask]]
        
        # Compute PMax[j] and argmax_PMax[j]
        # PMax[j][g] = max(PMax[j][g-1], dp[j][g])
        # We can do this with np.maximum.accumulate
        PMax[j, :] = np.maximum.accumulate(dp[j, :])
        
        # argmax_PMax[j][g]
        # We need the index of the max value in dp[j][0..g]
        # This is not directly provided by accumulate.
        # We can compute it by keeping track of the running argmax.
        # Since max_gap is 10000, a Python loop for this is 10000 iterations per j.
        # 1229 * 10000 = 12M iterations. Might be slow in Python.
        # Let's see if we can vectorize or use a faster method.
        
        # Alternative: 
        # We only need argmax_PMax for indices g that are actually used as gaps in future steps.
        # But we don't know which ones are used.
        # However, we can compute argmax_PMax more efficiently?
        # Maybe just use the loop if it's fast enough.
        # 12M simple integer comparisons in Python might take ~1-2 seconds.
        # Total time limit is 60s. So it should be fine.
        
        running_max = 0
        running_arg = 0
        # dp[j, 0] is always 0.
        # g from 1 to max_gap
        # We can use a list comprehension or loop.
        # To speed up, we can break early if dp[j, g] is 0 and running_max is 0? No, we need the array filled.
        
        # Let's try a numpy-based trick?
        # Actually, `np.argmax` on cumulative max doesn't give the index in the original array directly for each prefix.
        
        # Let's just use the loop. It's simple and likely fast enough.
        # Optimization: only compute up to the max gap seen so far? No, gaps can be large.
        
        # We can speed up the loop by using local variables.
        dp_j = dp[j]
        argmax_j = argmax_PMax[j]
        
        for g in range(1, max_gap + 1):
            if dp_j[g] > running_max:
                running_max = dp_j[g]
                running_arg = g
            argmax_j[g] = running_arg

    # Find global max
    max_len = 0
    best_j = -1
    best_gap = -1
    
    # We can use np.unravel_index on the flattened dp
    # But dp is large.
    # Iterating is fine.
    for j in range(n_primes):
        row_max = np.max(dp[j, :])
        if row_max > max_len:
            max_len = row_max
            best_j = j
            best_gap = int(np.argmax(dp[j, :]))
            
    if max_len < 2:
        return [primes[0]]
        
    # Reconstruct
    result = []
    cur_j = best_j
    cur_gap = best_gap
    
    while cur_j >= 0:
        result.append(primes[cur_j])
        pi = parent_i[cur_j, cur_gap]
        pg = parent_gap[cur_j, cur_gap]
        
        if pi < 0:
            break
            
        cur_j = pi
        cur_gap = pg
        
        # If the length at this step was 2, the parent is the start
        # How do we know? 
        # If dp[cur_j][cur_gap] == 2, then it was formed from a single prime.
        # In that case, parent_gap would be argmax_PMax[pi][gap] which is 0 (since dp[pi] was 0).
        # So if cur_gap == 0, we append the last prime and stop.
        if cur_gap == 0:
            result.append(primes[cur_j])
            break
            
    result.reverse()
    
    # Verify
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if any(gaps[i+1] < gaps[i] for i in range(len(gaps)-1)):
             # Fallback
             values = []
             for p in primes:
                 if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                     values.append(p)
             return values
             
    return result
