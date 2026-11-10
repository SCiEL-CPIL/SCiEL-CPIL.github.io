import time
import bisect

def _sieve_primes(limit):
    """Return list of all primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes[:]
    
    # dp[u][v] for u < v
    # dp[u][v] is the length of the longest valid subsequence ending with primes[u], primes[v]
    # parent[u][v] is the index w such that primes[w] is the element before primes[u] in the subsequence
    dp = [[0] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]
    
    # S_val[j][k] = max(dp[x][j]) for x in [k, j-1]
    # S_arg[j][k] = argmax x
    S_val = [None] * n
    S_arg = [None] * n
    
    best_len = 2
    best_u = 0
    best_v = 1
    
    # v is the last index
    for v in range(1, n):
        if time.time() > deadline:
            break
            
        # u is the second-to-last index
        for u in range(0, v):
            gap_v_u = primes[v] - primes[u]
            
            # Condition: primes[u] - primes[w] <= gap_v_u
            # primes[w] >= primes[u] - gap_v_u = 2*primes[u] - primes[v]
            threshold = 2 * primes[u] - primes[v]
            
            # Find smallest w < u such that primes[w] >= threshold
            # bisect_left returns the first index where value >= threshold
            # We search in primes[0:u]
            w_min = bisect.bisect_left(primes, threshold, 0, u)
            
            if w_min < u:
                # Valid w exists in [w_min, u-1]
                # We need max dp[x][u] for x in [w_min, u-1]
                # This is stored in S_val[u][w_min]
                # Note: S_val[u] is computed after all dp[x][u] for x < u are known.
                # Since we are at v > u, S_val[u] has already been computed in a previous iteration of the outer loop (when v was u).
                # Wait, the outer loop is v. The inner loop is u.
                # S_val[u] depends on dp[x][u] for x < u.
                # dp[x][u] is computed when the outer loop variable was u.
                # So when the outer loop is at v, all u < v have their S_val computed.
                max_prev_len = S_val[u][w_min]
                best_w = S_arg[u][w_min]
                
                dp[u][v] = max_prev_len + 1
                parent[u][v] = best_w
            else:
                # No valid predecessor w < u
                # The subsequence is just [primes[u], primes[v]]
                dp[u][v] = 2
                parent[u][v] = -1
            
            if dp[u][v] > best_len:
                best_len = dp[u][v]
                best_u = u
                best_v = v
        
        # Build S_val[v] and S_arg[v]
        # S_val[v][k] = max(dp[x][v]) for x in [k, v-1]
        if v > 0:
            sv = [0] * v
            sa = [0] * v
            # Initialize with the last element x = v-1
            sv[v-1] = dp[v-1][v]
            sa[v-1] = v-1
            
            for k in range(v-2, -1, -1):
                if dp[k][v] >= sv[k+1]:
                    sv[k] = dp[k][v]
                    sa[k] = k
                else:
                    sv[k] = sv[k+1]
                    sa[k] = sa[k+1]
                    
            S_val[v] = sv
            S_arg[v] = sa
        else:
            S_val[v] = []
            S_arg[v] = []

    # Reconstruct sequence
    seq = [primes[best_v], primes[best_u]]
    cur_u = best_u
    cur_v = best_v
    
    while parent[cur_u][cur_v] != -1:
        w = parent[cur_u][cur_v]
        seq.append(primes[w])
        cur_v = cur_u
        cur_u = w
        
    seq.reverse()
    
    return seq
