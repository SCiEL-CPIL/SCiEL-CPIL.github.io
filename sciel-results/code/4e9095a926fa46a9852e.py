import time

def _gen_smooth_set(max_val, primes):
    """
    Generate a set of all numbers <= max_val whose prime factors are in 'primes'.
    """
    smooth = set([1])
    # Use a list to iterate, but add to set to avoid duplicates
    # Simple iterative expansion
    # For small primes, nested loops or BFS is fine.
    # BFS/Queue approach
    from collections import deque
    q = deque([1])
    while q:
        val = q.popleft()
        if val > max_val:
            continue
        for p in primes:
            new_val = val * p
            if new_val > max_val:
                break
            if new_val not in smooth:
                smooth.add(new_val)
                q.append(new_val)
    return smooth

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    N = 123457
    
    # Try increasing smoothness bounds: 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    # Primes for each bound
    bounds_primes = {
        3: [2, 3],
        5: [2, 3, 5],
        7: [2, 3, 5, 7],
        11: [2, 3, 5, 7, 11],
        13: [2, 3, 5, 7, 11, 13],
        17: [2, 3, 5, 7, 11, 13, 17],
        19: [2, 3, 5, 7, 11, 13, 17, 19],
        23: [2, 3, 5, 7, 11, 13, 17, 19, 23],
        29: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        31: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
        37: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37],
        41: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
        43: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
        47: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
        53: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53],
        59: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59],
        61: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61],
        67: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67],
        71: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71],
        73: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73],
        79: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79],
        83: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83],
        89: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89],
        97: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    }
    
    for bound in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if time.time() > deadline:
            break
            
        primes = bounds_primes[bound]
        smooth_set = _gen_smooth_set(N, primes)
        
        # Check for pair a + b = N
        # Iterate through smaller half to avoid duplicate checks
        # Since set lookup is O(1), this is fast
        found = False
        best_pair = None
        # We want to return ANY pair found in this bound, as it minimizes the max prime factor to 'bound'
        # Actually, we just need one.
        for a in smooth_set:
            b = N - a
            if b in smooth_set:
                # Ensure a, b are positive
                if a >= 1 and b >= 1:
                    best_pair = (a, b)
                    found = True
                    break
        
        if found:
            return list(best_pair)
            
    # Fallback if no smooth pair found up to 97
    # This is unlikely, but we need a valid return
    # Use a simple heuristic: try to find a with small P(a)
    # Or just return [1, N-1] if we are out of time
    if time.time() > deadline - 5:
        return [1, N - 1]
        
    # General search for best max(P(a), P(b))
    # This is slow, so we only do it if we have time
    # We can limit the search to a < N/2
    best_val = float('inf')
    best_a = 1
    # Sample some values or do a quick scan
    # Since we failed smooth checks, P(a) or P(b) is likely large.
    # We'll just do a linear scan with a time check
    for a in range(1, N // 2 + 1):
        b = N - a
        # Quick factorization for max prime factor
        # This is too slow for N=123457 in 60s if done naively for all a.
        # But we only get here if smooth checks failed.
        # Let's just return [1, N-1] as a last resort valid answer
        # Or try a few randoms?
        # Given the constraints, [1, N-1] is safe.
        pass
        
    return [1, N - 1]
