import time
import numpy as np
from math import isqrt

def _is_prime_sieve(n):
    if n < 2:
        return np.zeros(n + 1, dtype=bool)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime

def construct():
    deadline = time.time() + 80  # Safety margin
    
    N_LIMIT = 10000
    is_prime = _is_prime_sieve(N_LIMIT)
    primes = [i for i in range(2, N_LIMIT + 1) if is_prime[i]]
    n_primes = len(primes)
    
    if n_primes == 0:
        return []
    if n_primes == 1:
        return [primes[0]]
    
    p_arr = np.array(primes, dtype=np.int32)
    max_gap = N_LIMIT  # Maximum possible gap
    
    # dp[i][g] = max length of valid sequence ending at primes[i] with last gap exactly g
    # PMax[i][g] = max_{h <= g} dp[i][h]  (prefix max)
    # argmax_PMax[i][g] = argmax_{h <= g} dp[i][h]
    # parent_i[i][g] = the prime index j that preceded i in the optimal chain
    # parent_gap[i][g] = the gap used at j (i.e., which gap state at j was used)
    
    dp = np.zeros((n_primes, max_gap + 1), dtype=np.int16)
    PMax = np.zeros((n_primes, max_gap + 1), dtype=np.int16)
    argmax_PMax = np.zeros((n_primes, max_gap + 1), dtype=np.int16)
    parent_i = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    parent_gap = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    
    best_len = 1
    best_j = 0
    best_gap = 0
    
    for j in range(1, n_primes):
        if time.time() > deadline:
            break
            
        gaps = p_arr[j] - p_arr[:j]  # gaps from all previous primes to j
        
        # For each previous prime i, we can extend any state at i with gap <= gaps[i]
        # The best we can do is PMax[i][gaps[i]]
        prev_best = PMax[:j, gaps]  # shape (j,)
        
        # Candidates: if prev_best[i] > 0, we can extend; else start new (length 2)
        cands = np.where(prev_best > 0, prev_best + 1, 2)
        
        # Update dp[j] at the specific gap positions
        # For each i, dp[j][gaps[i]] = max(dp[j][gaps[i]], cands[i])
        # We need to handle potential collisions (multiple i giving same gap)
        for i in range(j):
            g = gaps[i]
            c = cands[i]
            if c > dp[j, g]:
                dp[j, g] = c
                parent_i[j, g] = i
                parent_gap[j, g] = argmax_PMax[i, g]
        
        # Compute prefix max and argmax for row j
        # PMax[j][g] = max(dp[j][0], dp[j][1], ..., dp[j][g])
        running_max = 0
        running_arg = 0
        for g in range(max_gap + 1):
            v = dp[j, g]
            if v > running_max:
                running_max = v
                running_arg = g
            PMax[j, g] = running_max
            argmax_PMax[j, g] = running_arg
            if running_max > best_len:
                best_len = running_max
                best_j = j
                best_gap = g
    
    if best_len < 2:
        return [primes[0]]
    
    # Reconstruct the chain
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
        
        # If we've reached a state with no valid parent (start of chain)
        # parent_i should be -1 for the first element
        # But we also need to handle the case where the chain length is 2
        # In that case, dp[j][gap] = 2 and parent_i[j][gap] = i, parent_gap[j][gap] = 0
        # When we go to i, dp[i][0] should be 0 (no state with gap 0)
        # So we need to check if cur_gap == 0 and dp[cur_j][cur_gap] == 0
        if cur_gap == 0 and dp[cur_j, cur_gap] == 0:
            result.append(primes[cur_j])
            break
    
    result.reverse()
    
    # Verify the result
    if len(result) >= 2:
        valid = True
        for i in range(len(result) - 1):
            if result[i+1] <= result[i]:
                valid = False
                break
            if i > 0:
                gap_curr = result[i+1] - result[i]
                gap_prev = result[i] - result[i-1]
                if gap_curr < gap_prev:
                    valid = False
                    break
        if valid and all(2 <= p <= N_LIMIT and is_prime[p] for p in result):
            return result
    
    # Fallback: greedy approach
    values = []
    for p in primes:
        if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
            values.append(p)
    return values
