import time
import math

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s limit
    results = []
    
    # Precompute smallest prime factor for fast factorization
    max_val = 256 * 256 * 4  # max B = n*x, n<=256, x<=3n/4, so B <= 256*192 = 49152
    # Actually B = n*x where x <= 3n/4, so B <= 3n^2/4 <= 3*256^2/4 = 49152
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        """Get prime factorization of num as dict {prime: exponent} using SPF."""
        factors = {}
        while num > 1:
            p = spf[num]
            count = 0
            while num % p == 0:
                num //= p
                count += 1
            factors[p] = count
        return factors
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from prime factorization dict."""
        divisors = [1]
        for p, e in factors.items():
            new_divs = []
            p_power = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return divisors
    
    for n in range(3, 257):
        if time.time() > deadline:
            results.append(None)
            continue
        
        found = None
        # x must satisfy n/4 < x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values in order of "niceness" - prefer x that make A=4x-n small
        # Sort x by |4x - n| to try simpler cases first
        x_candidates = sorted(range(x_min, x_max + 1), key=lambda x: abs(4*x - n))
        
        for x in x_candidates:
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need 1/y + 1/z = A/B
            # (Ay - B)(Az - B) = B²
            # Let u = Ay - B, v = Az - B, then u * v = B²
            # y = (u + B) / A, z = (v + B) / A
            # u and v must both be positive, so u > 0 and v > 0
            
            # Factorize B to get divisors of B²
            b_factors = factorize(B)
            # For B², each exponent is doubled
            b2_factors = {p: 2*e for p, e in b_factors.items()}
            divs = get_divisors_from_factors(b2_factors)
            
            for u in divs:
                v = B * B // u
                # Check y = (u + B) / A is integer
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                if y <= 0:
                    continue
                
                # Check z = (v + B) / A is integer
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                if z <= 0:
                    continue
                
                # Check bounds
                if y > 10**12 or z > 10**12:
                    continue
                
                # Ensure distinctness and positive
                if x == y or x == z or y == z:
                    continue
                
                # Verify by cross multiplication: n*(yz + xz + xy) == 4*xyz
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = sorted([x, y, z])
                    break
            
            if found:
                break
        
        results.append(found)
    
    return results
