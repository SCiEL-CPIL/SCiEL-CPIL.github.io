import time
import math
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)
    
    # dp[i] maps gap -> (length, prev_j, prev_gap)
    dp = [dict() for _ in range(n)]
    # For each j, after dp[j] is complete, build:
    #   j_gaps[j] = sorted list of gaps in dp[j]
    #   j_best[j] = for each position k in sorted gaps, the best (length, gap, prev_j, prev_g) 
    #     among all gaps <= j_gaps[j][k]
    # Actually simpler: j_prefix[j] = list of (gap, length, prev_j, prev_g) where each entry
    #   has the max length seen so far up to that gap
    j_gaps = [None] * n
    j_prefix = [None] * n  # list of (gap, best_len, prev_j, prev_g) with non-decreasing gap
    
    best_len = 1
    best_i = 0
    best_g = 0
    
    for i in range(n):
        if time.time() > deadline:
            break
        
        dp[i][0] = (1, -1, -1)
        
        for j in range(i):
            new_gap = primes[i] - primes[j]
            
            # Query j_prefix[j] for best entry with gap <= new_gap
            best_val = -1
            best_prev_j = -1
            best_prev_g = -1
            if j_prefix[j] is not None:
                # j_gaps[j] is sorted, j_prefix[j] has same length
                idx = bisect.bisect_right(j_gaps[j], new_gap) - 1
                if idx >= 0:
                    gap_k, len_k, pj_k, pg_k = j_prefix[j][idx]
                    best_val = len_k
                    best_prev_j = pj_k
                    best_prev_g = pg_k
            
            if best_val > 0:
                new_len = best_val + 1
                if new_gap not in dp[i] or new_len > dp[i][new_gap][0]:
                    dp[i][new_gap] = (new_len, j, best_prev_g)
                if new_len > best_len:
                    best_len = new_len
                    best_i = i
                    best_g = new_gap
        
        # Build j_prefix[i] from dp[i]
        if dp[i]:
            items = sorted(dp[i].items(), key=lambda x: x[0])
            gaps_list = [g for g, v in items]
            prefix_list = []
            max_len = -1
            best_pj = -1
            best_pg = -1
            for g, (L, pj, pg) in items:
                if L > max_len:
                    max_len = L
                    best_pj = pj
                    best_pg = pg
                prefix_list.append((g, max_len, best_pj, best_pg))
            j_gaps[i] = gaps_list
            j_prefix[i] = prefix_list
    
    # Reconstruct
    seq = []
    i = best_i
    g = best_g
    while i != -1:
        L, pj, pg = dp[i][g]
        seq.append(primes[i])
        i = pj
        g = pg
    seq.reverse()
    return seq
