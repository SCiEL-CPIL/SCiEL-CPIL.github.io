import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Sieve for primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def factorize(num):
        factors = {}
        temp = num
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors[p] = count
        if temp > 1:
            if temp <= limit:
                factors[temp] = 1
            else:
                factors[temp] = 1
        return factors

    def get_divisors(factors):
        divisors = [1]
        for p, exp in factors.items():
            new_divs = []
            p_power = 1
            for _ in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return divisors

    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        # Try different strategies for finding solutions
        # Strategy 1: Fix z (largest denominator) and solve for x, y
        # 1/x + 1/y = 4/n - 1/z = (4z - n)/(nz)
        # Let D = 4z - n, C = nz
        # Then 1/x + 1/y = D/C => C(x+y) = Dxy => Dxy - Cx - Cy = 0
        # Multiply by D: D²xy - DCx - DCy = 0
        # (Dx - C)(Dy - C) = C²
        # So we need divisors u, v of C² such that u*v = C², 
        # x = (u+C)/D, y = (v+C)/D, both integers and positive
        
        # z must satisfy: 1/z < 4/n => z > n/4
        # Also 1/x + 1/y > 4/n - 1/z, and since x,y >= 2, 1/x+1/y <= 1
        # So 4/n - 1/z <= 1 => 1/z >= 4/n - 1 => z <= n/(n-4) for n>4
        # For n=3: z <= 3/(-1) invalid, handle separately
        # Actually, for fixed n, z ranges from floor(n/4)+1 to some upper bound
        
        # Upper bound on z: since x,y >= 2 (and distinct from z), 1/x+1/y >= 1/2 + 1/3 = 5/6
        # So 4/n - 1/z >= 5/6 => 1/z <= 4/n - 5/6 = (24-5n)/(6n)
        # This requires 24-5n >= 0 => n <= 4.8, so only n=3,4
        # For n>=5, we need a different bound
        # Since x,y > z (we'll enforce ordering), 1/x+1/y < 2/z
        # So 4/n - 1/z < 2/z => 4/n < 3/z => z < 3n/4
        # Wait, that's not right. If x,y < z, then 1/x,1/y > 1/z
        # Let me just try z from floor(n/4)+1 upward with a reasonable cap
        
        max_z = 1000000000000  # Upper bound from problem
        
        # For efficiency, cap z at a reasonable value
        # From 1/x + 1/y = (4z-n)/(nz), with x,y >= 2:
        # (4z-n)/(nz) <= 1 => 4z-n <= nz => z(4-n) <= n
        # For n>4: z >= n/(n-4)
        # For n=4: 4z-4 <= 4z always true
        # For n=3: 4z-3 <= 3z => z <= 3
        
        # Better approach: iterate z in a smart range
        # z must be > n/4
        # Also, for the equation to have integer solutions, we need good number theory properties
        
        # Try small z values first (more likely to work)
        z_start = n // 4 + 1
        z_end = min(max_z, n * 1000)  # Cap at 1000*n for efficiency
        
        for z in range(z_start, z_end + 1):
            if time.time() > deadline:
                break
            
            D = 4 * z - n
            if D <= 0:
                continue
            
            C = n * z
            
            # Check if D divides C² in the right way
            # (Dx - C)(Dy - C) = C²
            # Let u = Dx - C, v = Dy - C, so u*v = C²
            # x = (u+C)/D, y = (v+C)/D
            # Need u+C ≡ 0 (mod D) and v+C ≡ 0 (mod D)
            # Since u*v = C², and u ≡ -C (mod D), v ≡ -C (mod D)
            # So C² ≡ C² (mod D), which is always true
            # But we need u ≡ -C (mod D)
            
            # Factor C²
            c_factors = factorize(C)
            c2_factors = {p: 2 * e for p, e in c_factors.items()}
            
            divs = get_divisors(c2_factors)
            
            for u in divs:
                if u * u > C * C:  # Only check half
                    continue
                v = (C * C) // u
                if u * v != C * C:
                    continue
                
                if (u + C) % D != 0:
                    continue
                if (v + C) % D != 0:
                    continue
                
                x = (u + C) // D
                y = (v + C) // D
                
                if x <= 0 or y <= 0:
                    continue
                if x > 1000000000000 or y > 1000000000000:
                    continue
                
                # Ensure distinctness and ordering
                denominators = sorted([x, y, z])
                if denominators[0] == denominators[1] or denominators[1] == denominators[2]:
                    continue
                
                # Verify the solution
                a, b, c = denominators
                if a * b * c == 4 * n * b * c + 4 * n * a * c + 4 * n * a * b:
                    found = [a, b, c]
                    break
            
            if found:
                break
        
        # If not found, try a different strategy
        if not found:
            # Try fixing x and searching for y, z
            min_x = n // 4 + 1
            max_x = (3 * n) // 4 + 10  # Slightly expanded range
            
            for x in range(min_x, max_x + 1):
                if time.time() > deadline:
                    break
                
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                
                # 1/y + 1/z = (A)/(B) => A(y+z) = Byz => Byz - Ay - Az = 0
                # (By - A)(Bz - A) = A²
                a2 = A * A
                
                # Factor A²
                a_factors = factorize(A)
                a2_factors = {p: 2 * e for p, e in a_factors.items()}
                divs = get_divisors(a2_factors)
                
                for u in divs:
                    if u * u > a2:
                        continue
                    v = a2 // u
                    if u * v != a2:
                        continue
                    
                    if (u + A) % B != 0:
                        continue
                    if (v + A) % B != 0:
                        continue
                    
                    y = (u + A) // B
                    z = (v + A) // B
                    
                    if y <= 0 or z <= 0:
                        continue
                    if y > 1000000000000 or z > 1000000000000:
                        continue
                    
                    denominators = sorted([x, y, z])
                    if denominators[0] == denominators[1] or denominators[1] == denominators[2]:
                        continue
                    
                    a, b, c = denominators
                    if a * b * c == 4 * n * b * c + 4 * n * a * c + 4 * n * a * b:
                        found = [a, b, c]
                        break
                
                if found:
                    break
        
        answers.append(found)
        
    return answers
