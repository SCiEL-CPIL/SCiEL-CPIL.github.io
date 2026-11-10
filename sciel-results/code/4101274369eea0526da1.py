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
    deadline = start_time + 85.0

    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes[:]
    
    # dp[u][v] for u < v
    # We use a list of lists. dp[u] is a list of length n, but only indices > u are valid.
    # To save memory and initialization time, we can use a dict or just initialize on the fly.
    # But n is small (~1229), so a 2D list is fine.
    # dp[u][v] = length
    # parent[u][v] = w
    dp = [[0] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]
    
    # structures[u] = (keys, prefix_maxes, prefix_argmaxes)
    # keys: sorted list of p_u - p_w for w < u
    # prefix_maxes: prefix max of dp[w][u] corresponding to keys
    # prefix_argmaxes: prefix argmax (w) corresponding to keys
    structures = [None] * n
    
    best_len = 2
    best_u = 0
    best_v = 1
    
    # v is the last index
    for v in range(1, n):
        # u is the second-to-last index
        for u in range(0, v):
            gap_v_u = primes[v] - primes[u]
            
            if structures[u] is not None:
                keys, prefix_maxes, prefix_argmaxes = structures[u]
                # Find rightmost index with key <= gap_v_u
                pos = bisect.bisect_right(keys, gap_v_u)
                if pos > 0:
                    idx = pos - 1
                    max_prev_len = prefix_maxes[idx]
                    best_w = prefix_argmaxes[idx]
                    dp[u][v] = max_prev_len + 1
                    parent[u][v] = best_w
                else:
                    dp[u][v] = 2
                    parent[u][v] = -1
            else:
                dp[u][v] = 2
                parent[u][v] = -1
                
            if dp[u][v] > best_len:
                best_len = dp[u][v]
                best_u = u
                best_v = v
        
        # Build structure for index v
        # We need dp[w][v] for all w < v
        # These were just computed in the inner loop above
        pairs = []
        for w in range(0, v):
            if dp[w][v] > 0:
                key = primes[v] - primes[w]
                pairs.append((key, dp[w][v], w))
        
        if pairs:
            pairs.sort(key=lambda x: x[0])
            keys = [p[0] for p in pairs]
            vals = [p[1] for p in pairs]
            ws = [p[2] for p in pairs]
            
            prefix_maxes = [0] * len(vals)
            prefix_argmaxes = [0] * len(vals)
            prefix_maxes[0] = vals[0]
            prefix_argmaxes[0] = ws[0]
            
            for i in range(1, len(vals)):
                if vals[i] > prefix_maxes[i-1]:
                    prefix_maxes[i] = vals[i]
                    prefix_argmaxes[i] = ws[i]
                else:
                    prefix_maxes[i] = prefix_maxes[i-1]
                    prefix_argmaxes[i] = prefix_argmaxes[i-1]
                    
            structures[v] = (keys, prefix_maxes, prefix_argmaxes)
        else:
            structures[v] = None
            
        if time.time() > deadline:
            break

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
