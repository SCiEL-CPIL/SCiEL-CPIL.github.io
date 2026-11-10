import bisect

def construct():
    # Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    m = len(primes)
    
    if m < 2:
        return primes
        
    # dp_len[i][j] = max length of valid sequence ending at primes[i] with previous prime primes[j]
    # dp_parent[i][j] = index k such that the sequence ending at (i, j) extends the sequence ending at (j, k)
    # Note: j < i
    
    dp_len = [[0] * i for i in range(m)]
    dp_parent = [[-1] * i for i in range(m)]
    
    # suf_val[j][k] = max(dp_len[j][x] for x in [k, j-1])
    # suf_idx[j][k] = argmax x
    suf_val = [[0] * (i + 1) for i in range(m)]
    suf_idx = [[-1] * (i + 1) for i in range(m)]
    
    best_len = 2
    best_i = 0
    best_j = 0
    
    for i in range(1, m):
        # Compute dp_len[i][j] for all j < i
        for j in range(i):
            gap = primes[i] - primes[j]
            threshold = primes[j] - gap
            
            # Find smallest k < j such that primes[k] >= threshold
            # bisect_left returns the insertion point
            k_min = bisect.bisect_left(primes, threshold, 0, j)
            
            if k_min < j:
                # There exists a valid k
                prev_len = suf_val[j][k_min]
                prev_k = suf_idx[j][k_min]
                dp_len[i][j] = prev_len + 1
                dp_parent[i][j] = prev_k
            else:
                dp_len[i][j] = 2
                dp_parent[i][j] = -1
            
            if dp_len[i][j] > best_len:
                best_len = dp_len[i][j]
                best_i = i
                best_j = j
        
        # Build suf_val and suf_idx for index i
        # suf_val[i][i] = 0, suf_idx[i][i] = -1 (already initialized)
        for k in range(i - 1, -1, -1):
            if dp_len[i][k] > suf_val[i][k + 1]:
                suf_val[i][k] = dp_len[i][k]
                suf_idx[i][k] = k
            else:
                suf_val[i][k] = suf_val[i][k + 1]
                suf_idx[i][k] = suf_idx[i][k + 1]
                
    # Reconstruct the sequence
    seq = []
    curr_i = best_i
    curr_j = best_j
    
    # The sequence ends with primes[curr_i], primes[curr_j]
    seq.append(primes[curr_i])
    seq.append(primes[curr_j])
    
    while curr_j != -1:
        k = dp_parent[curr_i][curr_j]
        if k == -1:
            break
        seq.append(primes[k])
        curr_i, curr_j = curr_j, k
        
    seq.reverse()
    return seq
