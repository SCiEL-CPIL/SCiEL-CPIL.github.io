"""Construct N with many representations as sum of two powerful numbers."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    from math import isqrt
    import numpy as np

    deadline = time.time() + 130  # safety margin before 150s limit

    def is_powerful(n):
        """Check if n is powerful: every prime factor appears with exponent >= 2."""
        if n < 1:
            return False
        if n == 1:
            return True
        # Trial division up to cube root, then check remaining
        temp = n
        d = 2
        while d * d * d <= temp:
            if temp % d == 0:
                cnt = 0
                while temp % d == 0:
                    temp //= d
                    cnt += 1
                if cnt == 1:
                    return False
            d += 1 if d == 2 else 2
        # Now temp is either 1, a prime, or a product of at most two primes each with exponent >= 2
        # Actually after removing all small primes, temp could be p^k where k>=1
        # If temp > 1, it's either prime (exponent 1 -> not powerful) or composite
        # If temp is prime -> exponent 1, not powerful (unless it was already removed)
        # If temp is p^2 -> powerful
        # If temp is p*q with p,q distinct primes each exponent 1 -> not powerful
        # If temp is p^k for k >= 2 -> powerful
        if temp > 1:
            r = isqrt(temp)
            if r * r == temp:
                # temp is a perfect square, so its prime factors all have even exponents >= 2
                return True
            # Check if temp is prime
            d2 = 2
            while d2 * d2 <= temp:
                if temp % d2 == 0:
                    return False  # temp has a prime factor with exponent 1
                d2 += 1 if d2 == 2 else 2
            # temp is prime with exponent 1
            return False
        return True

    def generate_powerful_numbers(limit):
        """Generate all powerful numbers up to limit."""
        pn = set()
        # Powerful numbers are of form x^2 * y^3 where y is square-free
        # Actually, a powerful number n = prod p_i^{e_i} where e_i >= 2
        # We can generate them by: for each square-free y, n = x^2 * y^3
        # y must be square-free, and x^2 * y^3 <= limit
        # y^3 <= limit => y <= limit^(1/3)
        # For each square-free y, x <= sqrt(limit / y^3)
        
        # First generate square-free numbers up to limit^(1/3)
        max_y = int(round(limit ** (1/3))) + 1
        
        def is_square_free(n):
            if n < 1:
                return False
            d = 2
            while d * d <= n:
                if n % (d * d) == 0:
                    return False
                d += 1 if d == 2 else 2
            return True
        
        square_free = [y for y in range(1, max_y + 1) if is_square_free(y)]
        
        for y in square_free:
            y3 = y ** 3
            max_x = isqrt(limit // y3)
            for x in range(1, max_x + 1):
                n = x * x * y3
                if n <= limit:
                    pn.add(n)
        
        return sorted(pn)

    # Strategy: generate powerful numbers up to some limit, then find N with max representations
    # N = a + b where a, b are powerful, a <= b, a + b = N
    # For a given powerful number a, and N, b = N - a must be powerful
    # So we want N such that many pairs (a, N-a) both in powerful set
    
    # Generate powerful numbers up to a limit
    # N is at most 10^18, so a and b are at most 10^18
    # But we can't generate all powerful numbers up to 10^18
    # Instead, use a smaller limit and find N within that range
    
    # The key insight: if we have powerful numbers up to M, we can find N up to 2M
    # with representations where both a and b <= M
    
    # Let's use M = 200_000_000 (200 million) to get enough powerful numbers
    # Number of powerful numbers up to M is roughly M^(1/2) * constant + M^(1/3) * constant
    # For M = 2e8: sqrt(M) ~ 14142, cube_root(M) ~ 585
    # So maybe ~20000 powerful numbers, which is manageable
    
    M = 200_000_000
    
    pn = generate_powerful_numbers(M)
    pn_set = set(pn)
    pn_arr = np.array(pn, dtype=np.int64)
    
    print(f"Generated {len(pn)} powerful numbers up to {M}")
    
    # Now find N with max number of representations
    # For each powerful a <= M, and powerful b <= M, N = a + b
    # We want to count for each N how many pairs (a, b) exist with a <= b
    
    # Use a dictionary to count representations
    # But N can be up to 2*M = 4e8, and we have ~20000 powerful numbers
    # So up to 20000^2 / 2 = 2e8 pairs, which might be too many
    
    # Better approach: for each a in pn (a <= M), for each b in pn (b >= a, b <= M):
    #   N = a + b, count[N] += 1
    # But this is O(|pn|^2) which could be 4e8 operations
    
    # Alternative: iterate through possible N values and count representations
    # For a given N, count = sum over a in pn where a <= N/2 and N - a in pn_set
    
    # Let's try a hybrid: pick candidate N values that are sums of small powerful numbers
    # and evaluate them
    
    # Actually, let's just do the O(|pn|^2) approach but with numpy for speed
    # |pn| ~ 20000, so |pn|^2 = 4e8, which is too slow in Python
    
    # Better: use the fact that we only need a <= b, so a <= N/2
    # Let's iterate a in pn, and for each a, find b in pn where b >= a and b <= M
    # N = a + b
    
    # Use numpy for vectorized operations
    # For each a, b can range from a to M, so N ranges from 2*a to a+M
    
    # Let's just do it with a loop over a, and for each a, compute all valid b
    # and accumulate counts in a dict
    
    counts = {}
    
    n_pn = len(pn)
    for i in range(n_pn):
        a = pn[i]
        if time.time() > deadline - 30:
            break
        # b must be >= a and <= M, so b in pn[i:]
        b_arr = pn_arr[i:]
        # N = a + b
        N_arr = a + b_arr
        # Count occurrences
        # Use np.unique with return_counts
        if len(N_arr) > 0:
            unique_N, cnt = np.unique(N_arr, return_counts=True)
            for j in range(len(unique_N)):
                N_val = int(unique_N[j])
                if N_val not in counts:
                    counts[N_val] = 0
                counts[N_val] += int(cnt[j])
    
    print(f"Evaluated up to i={i}, found {len(counts)} distinct N values")
    
    # Find the N with maximum count
    if not counts:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    print(f"Best N = {best_N} with {best_count} representations")
    
    # Generate the actual pairs
    pairs = []
    for a in pn:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in pn_set and a <= b:
            pairs.append([a, b])
    
    # Verify
    assert len(pairs) == best_count, f"Expected {best_count} pairs, got {len(pairs)}"
    for a, b in pairs:
        assert a + b == best_N
        assert a <= b
        assert is_powerful(a), f"{a} is not powerful"
        assert is_powerful(b), f"{b} is not powerful"
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
