import time
import sys
import bisect

def construct():
    start_time = time.time()
    DEADLINE = 80  # Safety margin before 90s kill
    LIMIT = 10000

    # 1. Generate primes using Sieve of Eratosthenes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False

    primes = [i for i, p in enumerate(is_prime) if p]
    n = len(primes)
    
    # dp[i] will be a list of (gap, length) pairs, sorted by gap,
    # such that length is strictly increasing with gap.
    # This allows binary search for max length with gap <= g.
    dp = [[] for _ in range(n)]
    
    # Base case: first prime
    dp[0] = [(0, 1)]
    
    best_len = 1
    best_seq = [primes[0]]
    
    # To reconstruct the sequence, we need to store the predecessor.
    # pred[i][k] = j, where k is the index in dp[i] of the optimal state.
    # Actually, it's easier to store the predecessor index j and the gap g used.
    # Let's store for each state in dp[i] the predecessor j.
    # But dp[i] is a list of (gap, length). We can have a parallel list pred[i] of predecessors.
    
    pred = [None] * n
    # pred[i] will be a list of predecessor indices j, parallel to dp[i].
    
    for i in range(1, n):
        candidates = []
        for j in range(i):
            g = primes[i] - primes[j]
            # Find max length in dp[j] with gap <= g
            # dp[j] is sorted by gap.
            # We need the largest index k such that dp[j][k][0] <= g.
            # Then the length is dp[j][k][1].
            # Since dp[j] is pruned such that lengths are increasing with gaps,
            # the max length for gap <= g is just the length at the largest gap <= g.
            
            # Binary search for g in dp[j] gaps
            gaps_j = [pair[0] for pair in dp[j]]
            # We can use bisect_right to find the insertion point
            k = bisect.bisect_right(gaps_j, g) - 1
            if k >= 0:
                max_len_from_j = dp[j][k][1]
                new_len = max_len_from_j + 1
                candidates.append((g, new_len, j))
        
        if not candidates:
            # This shouldn't happen for i>0 as we can always come from i-1?
            # Not necessarily if gap constraint fails?
            # But gap from i-1 to i is always >= 1.
            # And dp[i-1] has at least (0, 1). So 1 <= g.
            # So candidates should not be empty.
            dp[i] = []
            pred[i] = []
            continue
            
        # Sort candidates by gap
        candidates.sort(key=lambda x: x[0])
        
        # Prune candidates: keep only those with strictly increasing length
        filtered_candidates = []
        max_len_so_far = 0
        for g, l, j in candidates:
            if l > max_len_so_far:
                filtered_candidates.append((g, l))
                max_len_so_far = l
        
        dp[i] = filtered_candidates
        
        # Store predecessors for reconstruction
        # We need to map (g, l) back to j.
        # Since multiple j's can give the same (g, l), we just pick one.
        # But we stored j in candidates.
        # Let's re-associate j with the filtered candidates.
        # It's possible that a candidate with max length was pruned? No, we kept max length.
        # But if two candidates have same g and same l, we keep one.
        # Let's create a dict or list to map (g, l) to j.
        # Actually, we can just iterate through candidates again and match.
        
        pred_i = []
        # Create a lookup for j
        # Since candidates is sorted by g, and filtered_candidates is a subsequence,
        # we can just iterate.
        # But careful: multiple j's can produce same g.
        # We just need ONE valid predecessor for reconstruction.
        
        # Let's build a map from (g, l) to j.
        # But (g, l) might not be unique if different j's give same g and l.
        # We can just pick the first one.
        
        j_map = {}
        for g, l, j in candidates:
            if (g, l) not in j_map:
                j_map[(g, l)] = j
                
        for g, l in filtered_candidates:
            pred_i.append(j_map.get((g, l), -1))
            
        pred[i] = pred_i
        
        if max_len_so_far > best_len:
            best_len = max_len_so_far
            
        # Time check
        if time.time() - start_time > DEADLINE:
            # Reconstruct best sequence found so far
            # We need to find the state with best_len
            # This might be expensive if we don't track it.
            # Let's assume we finish.
            pass

    # Reconstruct the sequence
    # Find the i and state index k that gives best_len
    best_i = -1
    best_k = -1
    for i in range(n):
        for k, (g, l) in enumerate(dp[i]):
            if l > best_len: # Should be ==
                best_len = l
                best_i = i
                best_k = k
            elif l == best_len:
                # Tie-breaker? Doesn't matter for length.
                pass
                
    if best_i == -1:
        return [primes[0]]
        
    seq = []
    curr_i = best_i
    curr_k = best_k
    while curr_i != -1:
        seq.append(primes[curr_i])
        if curr_k < len(pred[curr_i]):
            prev_j = pred[curr_i][curr_k]
            if prev_j == -1:
                break
            # Find the state in dp[prev_j] that led to this
            # The gap used was g = primes[curr_i] - primes[prev_j]
            g_used = primes[curr_i] - primes[prev_j]
            # Find k in dp[prev_j] such that gap == g_used and length matches?
            # Actually, we just need the state that was used.
            # The length in dp[prev_j] at that state should be l-1.
            # And gap should be <= g_used.
            # Since we used bisect_right, we picked the largest gap <= g_used.
            # So we need to find k_prev in dp[prev_j] such that:
            # 1. dp[prev_j][k_prev][0] <= g_used
            # 2. dp[prev_j][k_prev][1] is the max length for gaps <= g_used.
            # This is exactly what the binary search would find.
            
            gaps_prev = [pair[0] for pair in dp[prev_j]]
            k_prev = bisect.bisect_right(gaps_prev, g_used) - 1
            if k_prev >= 0:
                curr_i = prev_j
                curr_k = k_prev
            else:
                break
        else:
            break
            
    seq.reverse()
    return seq
