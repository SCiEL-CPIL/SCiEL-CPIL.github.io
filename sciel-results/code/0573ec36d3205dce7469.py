def construct():
    from math import isqrt, gcd
    from itertools import product

    def find_uv(p):
        """Find u, v such that p = u^2 + v^2 for prime p ≡ 1 mod 4."""
        for u in range(1, isqrt(p) + 1):
            v2 = p - u * u
            v = isqrt(v2)
            if v * v == v2 and v > 0:
                return u, v
        return 0, 0

    def generate_sums(n, prms):
        """Generate all representations n = x^2 + y^2 using primes in prms (all ≡ 1 mod 4, exponent 1)."""
        # For each prime p, the Gaussian prime factors are (u+iv) and (u-iv).
        # For n = p1 * p2 * ... * pk, the representations come from choosing
        # one of {pi+vi*i, pi-vi*i} for each pi, then multiplying.
        # The result is a Gaussian integer Z = x + yi where x^2 + y^2 = n.
        
        # Start with Z = 1 + 0i
        Zs = [(1, 0)]  # list of (re, im)
        
        for (u, v) in prms:
            new_Zs = []
            for (re, im) in Zs:
                # Multiply by (u + vi)
                re1 = re * u - im * v
                im1 = re * v + im * u
                new_Zs.append((re1, im1))
                # Multiply by (u - vi)
                re2 = re * u + im * v
                im2 = -re * v + im * u
                new_Zs.append((re2, im2))
            Zs = new_Zs
        
        pairs = set()
        for (x, y) in Zs:
            x = abs(x)
            y = abs(y)
            if x == 0 or y == 0:
                continue
            a, b = min(x, y), max(x, y)
            pairs.add((a, b))
        
        return pairs

    # Primes ≡ 1 mod 4, chosen to maximize the product of (e_i + 1) while keeping N <= 10^18
    # For each prime, we use exponent 1 (since e=1 gives 2 representations per prime)
    # The number of representations is 2^k where k is the number of primes.
    # We want to maximize k such that p1 * p2 * ... * pk <= 10^18.
    
    # Primes ≡ 1 mod 4: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, ...
    # Let's try to use as many as possible.
    
    # Strategy: use the smallest primes ≡ 1 mod 4 to maximize count.
    # 5*13*17*29*37*41*53*61*73*89*97*101*109*113*137*149*157*173*181*193*...
    
    # Let's compute the product and see how many we can fit.
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 613, 617, 641, 653, 661, 673, 677, 701, 709, 733, 757, 761, 769, 773, 797, 809, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 997]
    
    # Find the maximum number of primes we can multiply while staying <= 10^18
    N_max = 10**18
    N = 1
    chosen_primes = []
    for p in primes_1mod4:
        if N * p <= N_max:
            N *= p
            chosen_primes.append(p)
        else:
            break
    
    # Now find all representations of N as x^2 + y^2
    prms = [find_uv(p) for p in chosen_primes]
    pairs_xy = generate_sums(N, prms)
    
    # For each representation x^2 + y^2 = N, we have a = x, b = y (both perfect squares)
    # So the pairs of powerful numbers are (x^2, y^2) where x^2 + y^2 = N
    # Wait, a = x^2 and b = y^2, so a + b = x^2 + y^2 = N. Yes!
    
    final_pairs = []
    for (x, y) in pairs_xy:
        a = x * x
        b = y * y
        final_pairs.append([a, b])
    
    final_pairs.sort()
    
    return {'N': N, 'pairs': final_pairs}
