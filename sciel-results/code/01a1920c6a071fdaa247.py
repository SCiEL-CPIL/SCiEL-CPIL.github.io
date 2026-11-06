import math
from functools import reduce

def construct():
    # Step 1: Find N <= 10^18 maximizing the number of representations as sum of two squares.
    # This corresponds to maximizing prod(e_i + 1) for N = prod p_i^e_i with p_i == 1 mod 4.
    
    # Primes 1 mod 4
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 733, 757, 761, 769, 773, 797, 809, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 997]
    
    # We will use a DFS to find the best exponents.
    # To keep it fast, we limit the depth.
    best_N = 1
    best_score = 0
    best_exponents = {}
    
    # Use a recursive search with pruning
    # Primes are sorted ascending.
    # We assign exponents e_1, e_2, ...
    # Constraint: N <= 10^18
    
    def dfs(idx, current_N, current_score, exponents):
        nonlocal best_N, best_score, best_exponents
        if current_score > best_score:
            best_score = current_score
            best_N = current_N
            best_exponents = exponents.copy()
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        # Max exponent for this prime
        max_e = 0
        temp_N = current_N
        while temp_N * p <= 10**18:
            temp_N *= p
            max_e += 1
            
        # Try exponents from max_e down to 0
        # It's usually better to have smaller primes with higher exponents
        for e in range(max_e, -1, -1):
            new_N = current_N * (p ** e)
            if new_N > 10**18:
                continue
            new_score = current_score * (e + 1)
            # Pruning: even if we add all remaining primes with exponent 1, can we beat best?
            # Rough estimate: remaining primes count * 2 (factor 2 for e=1)
            # This might be too loose, but let's try.
            dfs(idx + 1, new_N, new_score, exponents + {p: e})
            
    dfs(0, 1, 1, {})
    
    N = best_N
    
    # Step 2: Generate pairs (x, y) such that x^2 + y^2 = N.
    # Factor N
    factors = {}
    temp_N = N
    for p in primes:
        while temp_N % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp_N //= p
        if temp_N == 1:
            break
    
    # Find Gaussian integer factors for each prime
    # p = a^2 + b^2
    gaussian_factors = {}
    for p, e in factors.items():
        a = 1
        found = False
        while a * a <= p:
            b2 = p - a * a
            b = int(math.isqrt(b2))
            if b * b == b2:
                gaussian_factors[p] = (a, b, e)
                found = True
                break
            a += 1
        if not found:
            # Should not happen for p=1 mod 4
            gaussian_factors[p] = (1, 0, e) # Fallback
            
    # Generate all combinations
    # z = prod (a+bi)^k (a-bi)^(e-k)
    # We use complex numbers
    
    pairs = []
    
    # Recursive generation of z
    def gen_z(idx, current_z):
        if idx >= len(factors):
            x = int(current_z.real)
            y = int(current_z.imag)
            # We want x^2 + y^2 = N
            # And x, y > 0
            # And x <= y to avoid duplicates
            if x > 0 and y > 0:
                if x > y:
                    x, y = y, x
                a_val = x * x
                b_val = y * y
                pairs.append([a_val, b_val])
            return
            
        p, e = list(factors.items())[idx]
        a, b, exp_e = gaussian_factors[p]
        # Base factor: pi = a+bi, pi_bar = a-bi
        # We need to multiply by pi^k * pi_bar^(e-k) for k in 0..e
        # Note: pi^k * pi_bar^(e-k) = pi^e * (pi_bar/pi)^k
        # Simpler: just compute the power
        
        pi = complex(a, b)
        pi_bar = complex(a, -b)
        
        for k in range(e + 1):
            # term = pi^k * pi_bar^(e-k)
            # To avoid large intermediate numbers, we can compute iteratively
            # But e is small (max ~16)
            term = (pi ** k) * (pi_bar ** (e - k))
            gen_z(idx + 1, current_z * term)
            
    gen_z(0, 1+0j)
    
    # Remove duplicates (just in case, though x<=y should handle it)
    unique_pairs = []
    seen = set()
    for p in pairs:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_pairs.append(p)
            
    return {'N': N, 'pairs': unique_pairs}
