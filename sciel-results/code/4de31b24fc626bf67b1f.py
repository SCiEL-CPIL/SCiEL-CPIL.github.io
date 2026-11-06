import time
from math import isqrt, gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    answers = [None] * 254
    LIMIT = 10**12

    # Primes up to 1000 for efficient factorization
    # B <= 256 * 256 * 2 (approx) = 131072. sqrt(B) <= 362.
    # But B can be larger if x goes up to 3n/4. B = n*x. Max x ~ 192. B ~ 256*192 = 49152.
    # sqrt(49152) ~ 222.
    # Let's use primes up to 500 to be safe.
    primes = []
    is_prime = [True] * 500
    is_prime[0] = is_prime[1] = False
    for i in range(2, 500):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 500, i):
                is_prime[j] = False

    def factorize(n):
        factors = {}
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                factors[p] = cnt
        if temp > 1:
            factors[temp] = 1
        return factors

    def get_divisors_squared(factors):
        # Generate divisors of B^2
        # If B = p1^e1 * p2^e2, B^2 = p1^(2e1) * p2^(2e2)
        divs = [1]
        for p, e in factors.items():
            max_e = 2 * e
            new_divs = []
            for d in divs:
                p_pow = 1
                for i in range(max_e + 1):
                    new_divs.append(d * p_pow)
                    p_pow *= p
            divs = new_divs
        return divs

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        
        # x_min is derived from 1/x < 4/n => x > n/4
        # x_max is derived from 1/y + 1/z < 2/x. Since y,x >= x, 2/x <= 4/n - 1/x.
        # Actually, if x is too large, 1/x is small, so 1/y+1/z is large, requiring y,z small.
        # But y > x, so 1/y < 1/x. Thus 1/y + 1/z < 2/x.
        # We need 4/n - 1/x < 2/x => 4/n < 3/x => x < 3n/4.
        # So x must be in (n/4, 3n/4).
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Optimization: Try x values that make B = n*x highly composite?
        # Or just iterate x.
        # For each x, A = 4x - n, B = n*x.
        # We need U | B^2 such that (U+B) % A == 0 and (B^2/U + B) % A == 0.
        # y = (U+B)/A, z = (B^2/U + B)/A.
        # Conditions:
        # 1. x < y < z
        # 2. y, z <= LIMIT
        # 3. x, y, z distinct (implied by strict inequality if we order them)
        
        # We can sort x, y, z at the end? No, the formula assumes x is the first term.
        # The problem asks for *distinct* denominators. The order in the list [x,y,z] doesn't matter for the sum,
        # but the code structure assigns x as the loop variable.
        # If we find a solution with y < x, we should swap them?
        # The task says "three DISTINCT positive integer denominators x,y,z".
        # It does not require x < y < z in the output, but it's good practice.
        # However, the derivation 1/y + 1/z = 4/n - 1/x assumes x is one of the denominators.
        # If we find a valid triple (a,b,c), any permutation is valid.
        # To ensure we don't miss solutions where the "smallest" denominator is not the one we looped as x,
        # we should ensure that for a given set {x,y,z}, we only find it when x is the smallest?
        # Or just check if the found y,z are distinct from x and each other.
        # If y < x, we can swap x and y? No, because A and B change.
        # Actually, if (x,y,z) is a solution, then (y,x,z) is also a solution.
        # If we loop x from small to large, and find y < x, we can just return [y, x, z] (sorted).
        # But wait, if y < x, then 1/y > 1/x.
        # The equation 1/y + 1/z = 4/n - 1/x was derived.
        # If we swap, we'd need to re-derive for the new x.
        # However, if we just check the final triple for validity, it's safer.
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            
            # Factor B to get divisors of B^2
            # Caching factorization might help if B repeats? Unlikely for different x.
            factors_B = factorize(B)
            divs_B2 = get_divisors_squared(factors_B)
            
            # Sort divisors to find smallest valid y first?
            # y = (U+B)/A. To minimize y, we want small U.
            # But we need y > x.
            # Also z > y.
            # Let's iterate U.
            
            # Optimization:
            # U must be such that y = (U+B)/A is integer.
            # U + B = k * A  => U = k * A - B.
            # Also U | B^2.
            # This implies (k*A - B) | B^2.
            
            # Instead of iterating all divisors of B^2 (which can be many),
            # we can iterate k?
            # y = k.
            # Condition: (k*A - B) | B^2.
            # Let d = k*A - B. Then d | B^2.
            # Also k = (d+B)/A.
            # Since y > x, k > x.
            # d = k*A - B > x*A - B = x(4x-n) - nx = 4x^2 - nx - nx = 4x^2 - 2nx = 2x(2x-n).
            # If x > n/2, d > 0. If x < n/2, d can be negative?
            # But d must be a divisor of B^2, so d > 0.
            # So k*A - B > 0 => k > B/A.
            
            # Range of k:
            # k > max(x, B//A + 1)
            # Also z = B*k / (k*A - B).
            # We need z > y => z > k.
            # B*k / (k*A - B) > k  => B / (k*A - B) > 1 => B > k*A - B => 2B > k*A => k < 2B/A.
            # So k is in (B/A, 2B/A).
            # This range has size B/A = nx / (4x-n).
            # For x close to n/4, B/A is large.
            # For x close to 3n/4, B/A = 3n^2/4 / (3n - n) = 3n/8. Small!
            # So iterating k is efficient for large x.
            
            # Let's try iterating k for x where the range is small.
            # Range size R = 2B/A - B/A = B/A.
            # If B/A < 1000, iterate k.
            # Else, iterate divisors U.
            
            B_over_A = B / A
            if B_over_A < 5000:
                # Iterate k
                k_min = int(B_over_A) + 1
                # Ensure k > x
                if k_min <= x:
                    k_min = x + 1
                    
                k_max = int(2 * B_over_A)
                # z > y implies k < 2B/A.
                # Also need z integer.
                
                for k in range(k_min, k_max + 1):
                    if time.time() > deadline:
                        break
                    d = k * A - B
                    if d <= 0:
                        continue
                    # Check if d divides B^2
                    # B^2 % d == 0
                    if (B * B) % d != 0:
                        continue
                    
                    # y = k
                    y = k
                    # z = B*k / d
                    z = (B * k) // d
                    
                    # Check distinctness and bounds
                    if y == x or z == x or z == y:
                        continue
                    if y > LIMIT or z > LIMIT:
                        continue
                    if y < 1 or z < 1:
                        continue
                    
                    # Verify solution
                    # 1/x + 1/y + 1/z == 4/n
                    # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        # Sort the triple to be consistent?
                        # The problem doesn't require sorting, but distinctness is key.
                        # Let's return [x, y, z]
                        found = [x, y, z]
                        break
            else:
                # Iterate divisors U of B^2
                # We want y > x.
                # y = (U+B)/A.
                # (U+B)/A > x => U+B > A*x => U > A*x - B.
                # Let U_min = max(1, A*x - B + 1)
                # Also z > y => z > k => U < B (as derived before: if U < B, z > y)
                # Wait, earlier derivation:
                # y = (U+B)/A, z = (B^2/U + B)/A.
                # z > y <=> B^2/U + B > U + B <=> B^2/U > U <=> U^2 < B^2 <=> U < B.
                # So we need U < B.
                
                U_min = max(1, A * x - B + 1)
                
                # Iterate through divs_B2
                # We can filter divs_B2 for those in [U_min, B)
                # divs_B2 is not sorted. Let's sort it? Or just check.
                # Number of divisors of B^2 is manageable.
                for U in divs_B2:
                    if time.time() > deadline:
                        break
                    if U < U_min or U >= B:
                        continue
                    if (U + B) % A != 0:
                        continue
                    
                    y = (U + B) // A
                    # y > x is guaranteed by U_min choice, but check anyway
                    if y <= x:
                        continue
                    if y > LIMIT:
                        continue
                    
                    # Check z
                    z_num = (B * B) // U + B
                    if z_num % A != 0:
                        continue
                    z = z_num // A
                    
                    if z <= y:
                        continue
                    if z > LIMIT:
                        continue
                    if z == x or z == y:
                        continue
                    
                    # Verify
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers
