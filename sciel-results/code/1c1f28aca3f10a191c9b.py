import time
import math
import numpy as np
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Sieve for primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False

    primes = np.array([i for i, v in enumerate(is_prime) if v], dtype=np.int32)
    N = len(primes)

    if N == 0:
        return []
    if N == 1:
        return [int(primes[0])]

    # dp[i][j] = length of longest valid sequence ending with pair (primes[i], primes[j])
    # parent[i][j] = k, such that the sequence ending at (i, j) came from (k, i).
    
    dp = np.zeros((N, N), dtype=np.int32)
    parent = np.full((N, N), -1, dtype=np.int32)
    
    best_len = 2
    best_i = 0
    best_j = 1
    
    # Precomputed structures for each i
    # struct[i] = (sorted_gaps, pref_vals, pref_idxs)
    # sorted_gaps: list of gaps
    # pref_vals: prefix max of dp[k][i]
    # pref_idxs: index k (original index) that achieved the prefix max
    
    structs = [None] * N
    
    for i in range(N):
        if time.time() > deadline:
            break
            
        # Build structure for predecessors of i
        if i > 0:
            items = []
            for k in range(i):
                v = int(dp[k][i])
                if v > 0:
                    g = int(primes[i] - primes[k])
                    items.append((g, v, k))
            
            if items:
                # Sort by gap
                items.sort(key=lambda x: x[0])
                sorted_gaps = [x[0] for x in items]
                # pref_vals[m] = max value in items[0..m]
                # pref_idxs[m] = index k of the item that gave the max value
                pref_vals = [0] * len(items)
                pref_idxs = [0] * len(items)
                
                current_max_val = 0
                current_max_idx = -1
                
                for m in range(len(items)):
                    g, v, k = items[m]
                    if v > current_max_val:
                        current_max_val = v
                        current_max_idx = k
                    pref_vals[m] = current_max_val
                    pref_idxs[m] = current_max_idx
                structs[i] = (sorted_gaps, pref_vals, pref_idxs)
            else:
                structs[i] = ([], [], [])
        else:
            structs[i] = ([], [], [])

        # Compute dp[i][j] for all j > i
        for j in range(i + 1, N):
            if time.time() > deadline:
                break
                
            G = int(primes[j] - primes[i])
            
            sg, pv, pi = structs[i]
            
            if not sg:
                # No predecessors
                dp[i][j] = 2
                parent[i][j] = -1
            else:
                # Binary search for largest index m such that sg[m] <= G
                pos = bisect.bisect_right(sg, G)
                
                if pos > 0:
                    m = pos - 1
                    max_val = pv[m]
                    k = pi[m]
                    dp[i][j] = max_val + 1
                    parent[i][j] = k
                else:
                    dp[i][j] = 2
                    parent[i][j] = -1
            
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_i = i
                best_j = j

    # Reconstruct
    if best_len <= 1:
        return [int(primes[0])]
        
    path = []
    curr_j = best_j
    curr_i = best_i
    
    # The sequence ends with (primes[curr_i], primes[curr_j])
    path.append(int(primes[curr_j]))
    
    while curr_i != -1:
        path.append(int(primes[curr_i]))
        prev_k = int(parent[curr_i][curr_j])
        if prev_k == -1:
            break
        curr_j = curr_i
        curr_i = prev_k
        
    path.reverse()
    return path

