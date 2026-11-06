import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, 10s margin
    
    # 1. Generate primes p == 1 mod 4 up to a reasonable limit
    # We need primes to build N. 
    # The optimal N will likely use small primes.
    limit_primes = 500
    is_prime = [True] * (limit_primes + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit_primes)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_primes + 1, i):
                is_prime[j] = False
                
    primes_1_mod_4 = [p for p in range(2, limit_primes + 1) if is_prime[p] and p % 4 == 1]
    
    # 2. Find optimal N = prod p_i^e_i to maximize sum of squares representations
    # Maximize prod (e_i + 1) subject to N <= 1e18
    MAX_N = 10**18
    
    def dfs(idx, current_N, current_score):
        if time.time() > deadline - 10:
            return current_N, current_score
        if idx == len(primes_1_mod_4):
            return current_N, current_score
        
        p = primes_1_mod_4[idx]
        best_N = current_N
        best_score = current_score
        
        # Try exponents e from 0 upwards
        e = 0
        val = current_N
        while val <= MAX_N:
            new_score = current_score * (e + 1)
            # Pruning: if even with e=0 for rest, we can't beat best? 
            # Hard to prune simply, but depth is small.
            
            # Recurse
            sub_N, sub_score = dfs(idx + 1, val, new_score)
            
            if sub_score > best_score:
                best_score = sub_score
                best_N = sub_N
            
            e += 1
            if val > MAX_N // p:
                break
            val *= p
            
        return best_N, best_score

    # Start DFS
    # To speed up, we can limit the number of primes considered if N grows too fast
    # But DFS handles it.
    optimal_N, _ = dfs(0, 1, 1)
    
    # Ensure optimal_N is valid
    if optimal_N < 2:
        optimal_N = 5 # Fallback
    
    # 3. Generate all pairs (a, b) such that a+b=optimal_N and a,b are powerful
    # We focus on a=x^2, b=y^2 (squares are powerful)
    # Find all x, y such that x^2 + y^2 = optimal_N
    
    # Factorize optimal_N
    def factorize(n):
        factors = {}
        temp_n = n
        for p in primes_1_mod_4:
            if p * p > temp_n:
                break
            while temp_n % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp_n //= p
        # Check remaining
        if temp_n > 1:
            # It might be a prime > limit_primes or 1 mod 4
            # If it's 3 mod 4, it must have even exponent (which is 1? No, if 1 mod 4 logic holds, 
            # our N only has 1 mod 4 primes. 
            # Wait, did we include 2 or 3 mod 4 primes?
            # No, primes_1_mod_4 only.
            # So temp_n should be 1 or a prime 1 mod 4.
            if temp_n % 4 == 1:
                factors[temp_n] = factors.get(temp_n, 0) + 1
            else:
                # This shouldn't happen if we only used 1 mod 4 primes
                # But just in case, if it's a prime 3 mod 4, it contributes 0 to sum of squares reps?
                # No, if N has a factor q^k with q=3 mod 4 and k odd, r_2(N)=0.
                # Our construction ensures only 1 mod 4 primes, so r_2(N) > 0.
                pass
        return factors

    factors = factorize(optimal_N)
    
    # Helper: find a^2 + b^2 = p for prime p == 1 mod 4
    def find_sum_sq_prime(p):
        # Cornacchia's algorithm or simple search for small p
        # For p < 500, simple search is fine
        for a in range(1, int(math.sqrt(p)) + 1):
            b2 = p - a * a
            b = int(math.isqrt(b2))
            if b * b == b2:
                return a, b
        return 1, 0 # Fallback, shouldn't happen

    # Get base representations for each prime power
    # We need representations for p^e
    # If p = a^2 + b^2, then p^e has e+1 representations (up to sym)
    # We generate them using powers of (a+bi)
    
    base_reps = {}
    for p, e in factors.items():
        a, b = find_sum_sq_prime(p)
        reps = [(1, 0)]
        for _ in range(e):
            new_reps = []
            for x, y in reps:
                # (x+iy)(a+ib) = (xa-yb) + i(xb+ya)
                # (x-iy)(a+ib) = (xa+yb) + i(ya-xb) -- wait, conjugate
                # We want all integer solutions.
                # Multiplication in Gaussian integers:
                # (x+iy)(a+ib) = (xa-yb) + i(xb+ya)
                # (x+iy)(a-ib) = (xa+yb) + i(ya-xb)
                # Both give valid sums of squares.
                r1 = (abs(x*a - y*b), abs(x*b + y*a))
                r2 = (abs(x*a + y*b), abs(y*a - x*b))
                new_reps.append(r1)
                new_reps.append(r2)
            # Remove duplicates? 
            # For p=5, a=1, b=2.
            # e=1: (1,0)-> (1,2), (1,2). Duplicates?
            # (1*1-0*2, 1*2+0*1) = (1,2)
            # (1*1+0*2, 0*1-1*2) = (1,-2) -> (1,2)
            # So yes, duplicates.
            reps = list(set(new_reps))
        base_reps[p] = reps

    # Combine representations for all prime factors
    # N = prod p_i^e_i
    # Reps of N are convolutions of reps of p_i^e_i
    
    all_reps = [(1, 0)]
    for p, e in factors.items():
        new_all_reps = []
        for x, y in all_reps:
            for u, v in base_reps[p]:
                # (x+iy)(u+iv)
                r1 = (abs(x*u - y*v), abs(x*v + y*u))
                r2 = (abs(x*u + y*v), abs(y*u - x*v))
                new_all_reps.append(r1)
                new_all_reps.append(r2)
        all_reps = list(set(new_all_reps))
        
    # Filter and format pairs
    pairs = []
    seen = set()
    for x, y in all_reps:
        a = x * x
        b = y * y
        if a + b != optimal_N:
            continue
        if a < b:
            a, b = b, a
        # Ensure a, b are powerful (squares are powerful)
        if a <= b and (a, b) not in seen:
            seen.add((a, b))
            pairs.append([a, b])
            
    # Sort pairs for consistency
    pairs.sort()
    
    return {'N': optimal_N, 'pairs': pairs}
