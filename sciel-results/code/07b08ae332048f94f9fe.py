import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    MIN_N = 1000
    MAX_N = 1000000
    W = 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    Nmax = MAX_N + W
    smooth = [1] * (Nmax + 1)
    
    for i in range(2, Nmax + 1):
        n = i
        s = 1
        for p in primes:
            while n % p == 0:
                s *= p
                n //= p
            if n == 1:
                break
        smooth[i] = s

    # Use a list for frequency counting since smooth values <= Nmax
    # This is faster than a defaultdict.
    freq = [0] * (Nmax + 1)
    
    # Initialize window for n = MIN_N: indices MIN_N+1 .. MIN_N+W
    distinct_count = 0
    for i in range(MIN_N + 1, MIN_N + W + 1):
        v = smooth[i]
        if freq[v] == 0:
            distinct_count += 1
        freq[v] += 1

    best_count = distinct_count
    best_n = MIN_N

    # Slide
    for n in range(MIN_N + 1, MAX_N + 1):
        # remove smooth[n]
        out_val = smooth[n]
        freq[out_val] -= 1
        if freq[out_val] == 0:
            distinct_count -= 1
            
        # add smooth[n+W]
        in_val = smooth[n + W]
        if freq[in_val] == 0:
            distinct_count += 1
        freq[in_val] += 1

        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            # Theoretical minimum is 2 (1 is impossible for consecutive integers)
            if best_count == 2:
                # We can't do better than 2. 
                # Note: If 2 is possible, this finds it.
                # If 2 is not possible, this never triggers.
                pass 
                # Should we break? 
                # If we break, we assume 2 is the global min.
                # If 2 is possible, it's the global min.
                # So breaking is safe IF 2 is the absolute theoretical min.
                # But what if there's a window with 1? (Proven impossible).
                # So 2 is the lower bound.
                # If we find 2, we found the optimum.
                break

        if time.time() - start_time > DEADLINE:
            break

    return best_n
