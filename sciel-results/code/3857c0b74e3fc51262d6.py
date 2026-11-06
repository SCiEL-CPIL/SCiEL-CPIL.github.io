"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import isqrt
    import bisect

    deadline = time.time() + 55.0

    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = [False] * (((N - start) // step) + 1)
    
    primes = [i for i, v in enumerate(is_prime) if v]
    K = len(primes)
    
    if K == 0:
        return []
    if K == 1:
        return [primes[0]]

    gaps_at = [None] * K
    max_vals_at = [None] * K
    argmax_at = [None] * K
    best_prev = [None] * K
    
    best_chain = [primes[0]]
    best_len = 1
    
    for j in range(K):
        p_j = primes[j]
        curr_dp = {}
        curr_prev_info = {}
        
        for i in range(j):
            p_i = primes[i]
            g = p_j - p_i
            
            if i == 0:
                cand_len = 2
                cand_prev = (i, 0)
            else:
                gaps_i = gaps_at[i]
                if not gaps_i:
                    continue
                max_vals_i = max_vals_at[i]
                argmax_i = argmax_at[i]
                
                idx = bisect.bisect_right(gaps_i, g)
                if idx == 0:
                    continue
                k = idx - 1
                cand_len = max_vals_i[k] + 1
                best_g_idx = argmax_i[k]
                g_prev = gaps_i[best_g_idx]
                cand_prev = (i, g_prev)
            
            if g in curr_dp:
                if cand_len > curr_dp[g]:
                    curr_dp[g] = cand_len
                    curr_prev_info[g] = cand_prev
            else:
                curr_dp[g] = cand_len
                curr_prev_info[g] = cand_prev
                
        if curr_dp:
            sorted_gaps = sorted(curr_dp.keys())
            gaps_at[j] = sorted_gaps
            dp_vals = [curr_dp[g] for g in sorted_gaps]
            
            max_vals = [0] * len(sorted_gaps)
            argmax = [0] * len(sorted_gaps)
            current_max = 0
            current_arg = 0
            for k, val in enumerate(dp_vals):
                if val > current_max:
                    current_max = val
                    current_arg = k
                max_vals[k] = current_max
                argmax[k] = current_arg
                
            max_vals_at[j] = max_vals
            argmax_at[j] = argmax
            best_prev[j] = curr_prev_info
            
            if max_vals_at[j][-1] > best_len:
                best_len = max_vals_at[j][-1]
                chain = []
                cur_j = j
                cur_g = gaps_at[j][argmax_at[j][-1]]
                
                while True:
                    chain.append(primes[cur_j])
                    if cur_j == 0:
                        break
                    prev_info = best_prev[cur_j][cur_g]
                    cur_j, cur_g = prev_info
                chain.reverse()
                best_chain = chain
                
    return best_chain
# EVOLVE-BLOCK-END
