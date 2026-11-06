import time
import bisect

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    N = 10000
    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(N ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes

    # dp[i][j] = length of longest sequence ending with primes[j], primes[i]
    # prev[i][j] = index h such that the sequence ending with primes[h], primes[j] 
    #              is the best predecessor for primes[i]
    
    # Use lists of lists for flexibility, or fixed size arrays
    # n is ~1229, so n*n is ~1.5M. 
    # Python lists of lists are okay, but numpy might be faster for storage?
    # However, we need to store indices too.
    # Let's stick to standard lists for simplicity and clarity, performance should be fine.
    
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    # hist[j] = list of (gap, max_len, index_h)
    # sorted by gap
    hist = [[] for _ in range(n)]
    
    best_len = 1
    best_i = 0
    best_j = -1
    
    for i in range(n):
        if time.time() - t0 >= DEADLINE:
            break
            
        # Process transitions ending at i
        for j in range(i):
            if time.time() - t0 >= DEADLINE:
                break
                
            gap_ji = primes[i] - primes[j]
            
            # Base case: sequence of length 2 (just j, i)
            current_len = 2
            current_prev = -1
            
            # Query hist[j] for best predecessor h
            # We need max dp[j][h] such that primes[j] - primes[h] <= gap_ji
            h_list = hist[j]
            if h_list:
                # h_list is sorted by gap
                # Find rightmost index with gap <= gap_ji
                # Extract gaps for bisect
                # To avoid extracting every time, we could store gaps separately,
                # but h_list is small (size <= j). 
                # Let's just use bisect on a list of gaps.
                # Optimization: store gaps in a separate list for each j?
                # Or just use bisect on the tuple list? bisect works on first element.
                
                # h_list contains (gap, len, h)
                # We want to find the last item with gap <= gap_ji
                # bisect_right returns the insertion point.
                # We need to extract gaps for efficient bisect if not done previously.
                # Let's maintain a separate list of gaps for each j to make bisect O(1) access?
                # No, bisect on a list of tuples uses the first element.
                
                # However, bisect_right on a list of tuples (gap, len, h) 
                # compares the whole tuple if gaps are equal.
                # We want the last index where gap <= gap_ji.
                # If we use bisect_right(gaps_list, gap_ji), it's cleaner.
                
                # Let's store hist[j] as two parallel lists or a list of tuples.
                # Given the small size, extracting gaps might be okay, but doing it inside the loop is bad.
                # Let's store hist_gaps[j] separately.
                
                pass # Placeholder for logic below

        # Build hist[i]
        # We need to collect all (gap, dp[i][h], h) for h < i
        # Sort by gap
        # Compress to max length prefix
        
        # Update global best
        for j in range(i):
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_i = i
                best_j = j

    # If no pair was found (n<2), return single prime
    if best_j == -1:
        return [primes[0]]

    # Reconstruct
    result = []
    cur_i = best_i
    cur_j = best_j
    while cur_j != -1:
        result.append(primes[cur_i])
        next_j = prev[cur_i][cur_j]
        cur_i = cur_j
        cur_j = next_j
    result.append(primes[cur_i])
    result.reverse()
    return result
