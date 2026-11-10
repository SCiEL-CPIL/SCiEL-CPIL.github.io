import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin (limit is 90s)
    
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    M = 10**6 + 24  # We need indices up to 1000000 + 24
    
    # Precompute p_part using sieve method
    # p_part[j] = product of p^v_p(j) for p in P
    p_part = [1] * (M + 1)
    
    for p in P:
        pk = p
        while pk <= M:
            for m in range(pk, M + 1, pk):
                p_part[m] *= p
            # Check time during sieve
            if time.time() > deadline - 5.0:
                return 1000
            pk *= p
    
    # Sliding window: find n in [1000, 1000000] minimizing distinct p_part values in window [n+1, n+24]
    best_n = 1000
    best_score = 25  # Start high
    
    # Initialize frequency for window [1001, 1024]
    freq = {}
    for i in range(1001, 1025):
        v = p_part[i]
        freq[v] = freq.get(v, 0) + 1
    distinct = len(freq)
    
    if distinct < best_score:
        best_score = distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, 10**6 + 1):
        if time.time() > deadline:
            break
        
        # Remove n (which was at position n, i.e., the leftmost element of previous window was n)
        # Previous window was [n, n+23], new window is [n+1, n+24]
        # Remove p_part[n], add p_part[n+24]
        v_out = p_part[n]
        freq[v_out] -= 1
        if freq[v_out] == 0:
            del freq[v_out]
        
        v_in = p_part[n + 24]
        if v_in not in freq:
            freq[v_in] = 1
            distinct += 1
        else:
            freq[v_in] += 1
        
        if distinct < best_score:
            best_score = distinct
            best_n = n
            if best_score == 2:
                break
    
    return best_n
