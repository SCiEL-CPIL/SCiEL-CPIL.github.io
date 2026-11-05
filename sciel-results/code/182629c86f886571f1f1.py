import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    # Precompute smallest prime factor up to some limit for fast factorization
    # B = n*x, n<=256, x can be up to ~3n/4 + some. Let's say x up to 10000.
    # So B up to 256*10000 = 2.56e6.
    # Let's set limit to 5_000_000 to be safe.
    LIMIT = 5_000_000
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # prime
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(n):
        factors = {}
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            factors[p] = cnt
        return factors
    
    def generate_divisors(factors):
        divs = [1]
        for p, exp in factors.items():
            new_divs = []
            for d in divs:
                val = 1
                for _ in range(exp + 1):
                    new_divs.append(d * val)
                    val *= p
            divs = new_divs
        return divs

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        
        # x must satisfy: 1/x < 4/n => x > n/4.
        # Also, for distinct x,y,z with x<y<z, we have 1/x > 1/y > 1/z.
        # 3/x > 1/x + 1/y + 1/z = 4/n => x < 3n/4.
        # So x is in [floor(n/4)+1, floor(3n/4)].
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # To find a solution quickly, we can iterate x.
        # For each x, we solve 1/y + 1/z = (4x - n) / (nx).
        # Let A = 4x - n, B = nx.
        # (Ay - B)(Az - B) = B^2.
        # Let d = Ay - B. Then d divides B^2.
        # y = (d + B) / A.
        # z = (B^2/d + B) / A.
        # Conditions:
        # 1. y, z are integers.
        # 2. y > x, z > y (implies distinctness and order).
        # 3. y, z <= 10^12 (usually satisfied if n is small).
        
        # We want to minimize z or just find any valid triple.
        # To make y > x:
        # (d + B) / A > x  => d + B > Ax => d > Ax - B.
        # Ax - B = x(4x - n) - nx = 4x^2 - 2nx.
        # Let D_min = 4x^2 - 2nx + 1.
        
        # To make z > y:
        # Since y < z, we have Ay - B < Az - B => d < B^2/d => d^2 < B^2 => d < B.
        # So we need d in [D_min, B-1] such that d | B^2.
        
        # If we find such a d, we have a solution.
        # To maximize the chance of finding a solution quickly, we should pick x such that
        # the range [D_min, B-1] contains a divisor of B^2.
        
        # Note: d = B is a divisor, but d < B is required for z > y.
        # If d = B, then y = (2B)/A = 2nx/A = 2nx/(4x-n).
        # And z = (B+B)/A = y. So y=z, not distinct.
        # So we strictly need d < B.
        
        # Let's iterate x from x_min to x_max.
        # For each x, factorize B^2, generate divisors, and check.
        # To speed up, we can stop early if we find a solution.
        
        # Optimization: Instead of generating ALL divisors of B^2,
        # we can just iterate divisors of B^2 that are < B.
        # Or even better, iterate divisors d of B^2.
        # Since B is small (<= 256 * 192 ~ 50000), B^2 is ~ 2.5e9.
        # Number of divisors is small (usually < 1000).
        
        # However, factorizing B^2 for every x might be slow if done naively.
        # But we have spf, so factorization is O(log n).
        
        # Let's try iterating x.
        # We can also try to be smarter:
        # If A = 4x - n is small, then y and z are large.
        # If A is large, y and z are small.
        
        # Let's just iterate x.
        # To avoid TLE, we limit the number of x's we try per n?
        # The task says "Maximize fraction solved".
        # It is known that for any n, a solution exists.
        # In fact, for 4/n, solutions are abundant.
        # Let's iterate x until found.
        
        # To be safe against worst-case n, we should ensure we find a solution quickly.
        # Usually x = n//4 + 1 works or is close.
        
        # Let's iterate x from x_min to x_max.
        # If we don't find a solution in a reasonable number of steps, we might need a fallback.
        # But given the math, a solution should be found.
        
        # One potential issue: B^2 might have many divisors.
        # But checking each divisor is O(1).
        
        # Let's optimize the inner loop:
        # 1. Compute A, B.
        # 2. Compute D_min.
        # 3. If D_min >= B, no solution for this x (since d < B).
        #    Wait, d can be equal to B? No, d < B for z > y.
        #    If D_min >= B, skip.
        # 4. Factorize B. Get factors of B^2.
        # 5. Generate divisors of B^2.
        # 6. Filter divisors d such that D_min <= d < B.
        # 7. For each such d, check if (d+B) % A == 0 and (B^2/d + B) % A == 0.
        #    If yes, compute y, z. Check y > x (guaranteed by d >= D_min) and z > y (guaranteed by d < B).
        #    Also check y, z are positive integers.
        
        # To speed up step 5-7, we can just iterate d from D_min to B-1 and check if B^2 % d == 0.
        # Range size: B - D_min.
        # B = nx. D_min = 4x^2 - 2nx + 1.
        # B - D_min = nx - (4x^2 - 2nx + 1) = 3nx - 4x^2 - 1.
        # For x close to n/4:
        # x = n/4 => 3n(n/4) - 4(n/4)^2 = 3n^2/4 - n^2/4 = n^2/2.
        # For n=256, n^2/2 = 32768.
        # Iterating 32k times is fast.
        # For x close to 3n/4:
        # x = 3n/4 => 3n(3n/4) - 4(9n^2/16) = 9n^2/4 - 9n^2/4 = 0.
        # So the range shrinks as x increases.
        
        # So iterating d is feasible.
        # Total work per n: sum over x of (B - D_min).
        # This is roughly integral of (3nx - 4x^2) dx from n/4 to 3n/4.
        # ~ O(n^3).
        # For n=256, 256^3 = 16 million.
        # 254 n's * 16 million = 4 billion. Too slow in Python.
        
        # So we must use divisor generation, not linear scan of d.
        
        # Revised plan:
        # For each x:
        #   1. Compute A, B, D_min.
        #   2. If D_min >= B, continue.
        #   3. Factorize B.
        #   4. Generate divisors of B^2.
        #   5. For each divisor d of B^2:
        #        if D_min <= d < B:
        #            check integrality of y, z.
        
        # Generating divisors of B^2:
        # B has at most ~20 divisors (for small B). B^2 has at most ~400 divisors.
        # This is very fast.
        
        # Let's implement this.
        
        # To further speed up, we can sort divisors and break early?
        # We just need ANY valid d.
        
        # Let's try.
        
        # We also need to handle the case where no solution is found for a specific x.
        # We continue to next x.
        
        # If we run out of x's, we return None.
        # But we should find one.
        
        # Let's add a timeout check inside the x loop.
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            D_min = 4 * x * x - 2 * n * x + 1
            
            if D_min >= B:
                continue
                
            # Factorize B
            if B > LIMIT:
                # Fallback for large B (unlikely given bounds)
                # Just do linear scan of d?
                # Or skip?
                # For n=256, x=192, B=49152.
                # LIMIT is 5e6. So B is always < LIMIT.
                pass
            
            factors_B = factorize(B)
            
            # Generate divisors of B^2
            # If B = prod p^e, B^2 = prod p^(2e).
            factors_B_sq = {p: 2*e for p, e in factors_B.items()}
            divs = generate_divisors(factors_B_sq)
            
            # Check divisors
            for d in divs:
                if d < D_min or d >= B:
                    continue
                    
                # d is a valid candidate
                # y = (d + B) / A
                # z = (B^2/d + B) / A
                
                num_y = d + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                d_prime = (B * B) // d
                num_z = d_prime + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                # Verify
                if y > x and z > y:
                    # Check bound
                    if y <= 10**12 and z <= 10**12:
                        found = [x, y, z]
                        break
                        
            if found:
                break
            
        answers[idx] = found

    return answers
