import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 60s limit
    
    N = 210
    MAX_VAL = 10210
    
    # Precompute squarefree kernel for all numbers up to MAX_VAL
    # sf[x] = squarefree part of x (product of primes with odd exponent)
    sf = list(range(MAX_VAL + 1))
    for i in range(2, MAX_VAL + 1):
        if sf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                sf[j] //= i
                while sf[j] % i == 0:
                    sf[j] //= i
    # Now sf[x] is the squarefree kernel of x
    # Actually, the above loop is a bit off. Let me redo it properly.
    
    # Better approach: compute sf using sieve-like method
    sf = list(range(MAX_VAL + 1))
    for p in range(2, MAX_VAL + 1):
        if sf[p] == p:  # p is prime (not yet divided)
            p2 = p * p
            for j in range(p2, MAX_VAL + 1, p2):
                sf[j] //= p2
            # Also handle p itself if it appears once
            # Actually, we need to divide by p^2 for each square factor
            # Let me use a different approach
    
    # Clean approach: for each number, compute its squarefree kernel
    sf = [0] * (MAX_VAL + 1)
    sf[1] = 1
    for i in range(2, MAX_VAL + 1):
        sf[i] = i
    for p in range(2, int(MAX_VAL**0.5) + 1):
        p2 = p * p
        for j in range(p2, MAX_VAL + 1, p2):
            while sf[j] % p2 == 0:
                sf[j] //= p2
    
    # Verify: sf[x] should be the product of primes p where x has odd exponent of p
    # Let's verify with a few examples
    # sf[12] = sf[4*3] = 3 (since 4=2^2 is a square, 3 remains)
    # sf[18] = sf[9*2] = 2
    # sf[8] = sf[4*2] = 2
    # sf[16] = 1
    # sf[1] = 1
    # sf[2] = 2
    # sf[3] = 3
    # sf[6] = 6
    # sf[30] = 30
    # sf[210] = 210
    
    # The condition: product of selected numbers * 210 must be a perfect square
    # This means: sf(product of selected) * sf(210) must be a perfect square
    # Since sf(210) = 210 (all primes have exponent 1), we need:
    # sf(product of selected) * 210 is a perfect square
    # Which means: sf(product of selected) = 210 (since 210 * 210 = 210^2 is a square)
    # Wait, that's not quite right. Let me think again.
    # 
    # If P = product of selected numbers, we need P * 210 to be a perfect square.
    # P * 210 is a perfect square iff sf(P * 210) = 1.
    # sf(P * 210) = sf(P) * sf(210) / gcd(sf(P), sf(210))^2 ... no, that's not how it works.
    # 
    # Actually, sf(a*b) = sf(a) * sf(b) / (sf(gcd(a,b)))^2 is not correct either.
    # 
    # The squarefree kernel is multiplicative in the sense that:
    # sf(a*b) = sf(a) * sf(b) if gcd(a,b) is a square, but in general:
    # sf(a*b) = sf(a) * sf(b) * sf(gcd(a,b)) / (something)
    # 
    # Let me think differently. The condition P * 210 is a perfect square means:
    # For every prime p, the exponent of p in P * 210 is even.
    # The exponent of p in P is the sum of exponents of p in each selected number.
    # The exponent of p in 210 is 1 if p in {2,3,5,7}, else 0.
    # So we need: (sum of exponents of p in selected numbers) + (1 if p in {2,3,5,7} else 0) is even.
    # 
    # This is equivalent to: the XOR (in GF(2)) of the prime exponent parity vectors
    # of all selected numbers and 210 must be zero.
    # 
    # So if we represent each number by a vector in GF(2)^k where k is the number of primes
    # up to MAX_VAL, and the i-th component is 1 if the exponent of the i-th prime is odd,
    # then we need the XOR of all selected vectors and the vector for 210 to be zero.
    # 
    # This is equivalent to: the XOR of all selected vectors equals the vector for 210.
    
    # Let's compute the vector for 210
    # Primes up to MAX_VAL
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(MAX_VAL)
    D = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Compute vector for 210
    target = 0
    n = N
    for i, p in enumerate(primes):
        if p > n:
            break
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            if exp % 2 == 1:
                target |= (1 << i)
    
    # Now we need to find a subset S of {211, 212, ..., M} such that:
    # XOR of vectors of elements in S = target
    # And we want to minimize M - 210.
    # 
    # We can use BFS over XOR states. Start with state 0 (empty set).
    # For each M from 211 upward, we can either include M or not.
    # But this is exponential. Instead, we use the fact that we want the minimum M.
    # 
    # Alternative: for each M, check if target is in the span of vectors {v(211), ..., v(M)}.
    # If yes, find a small subset.
    # 
    # But finding a small subset is the hard part. Let's use a different approach.
    # 
    # Key insight: we can use the null space to reduce the size of the solution.
    # 
    # Let's build the basis incrementally and track the null space.
    
    # Basis for the span of vectors v(211), ..., v(M)
    # basis[i] is a vector with highest set bit at position i
    # origin[i] is the set of values that make up basis[i]
    
    basis = [0] * (D + 1)
    origin = [set() for _ in range(D + 1)]
    null_space = []
    
    # Function to reduce a vector by the current basis
    def reduce_vec(v, track_origin=False):
        temp = v
        temp_origin = set() if track_origin else None
        for i in range(D - 1, -1, -1):
            if (temp >> i) & 1:
                if basis[i] != 0:
                    temp ^= basis[i]
                    if track_origin:
                        temp_origin ^= origin[i]
                else:
                    break
        if track_origin:
            return temp, temp_origin
        return temp
    
    # Iterate M from 211 to MAX_VAL
    best_solution = None
    best_max = MAX_VAL + 1
    
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
        
        # Compute vector for M
        v = 0
        x = M
        for i, p in enumerate(primes):
            if p * p > x:
                break
            if x % p == 0:
                exp = 0
                while x % p == 0:
                    x //= p
                    exp += 1
                if exp % 2 == 1:
                    v |= (1 << i)
        if x > 1:
            if x in prime_to_idx:
                v |= (1 << prime_to_idx[x])
        
        # Try to add v to basis
        temp = v
        temp_origin = {M}
        
        for i in range(D - 1, -1, -1):
            if (temp >> i) & 1:
                if basis[i] == 0:
                    basis[i] = temp
                    origin[i] = temp_origin
                    break
                else:
                    temp ^= basis[i]
                    temp_origin ^= origin[i]
        
        if temp == 0:
            # v is in the span, so temp_origin is a null space vector
            null_space.append(temp_origin)
        
        # Check if target is in the span
        temp_t = target
        indices = set()
        in_span = True
        for i in range(D - 1, -1, -1):
            if (temp_t >> i) & 1:
                if basis[i] != 0:
                    temp_t ^= basis[i]
                    indices ^= origin[i]
                else:
                    in_span = False
                    break
        
        if in_span and temp_t == 0:
            # Found a solution
            # Try to reduce size using null space
            if len(indices) > 100:
                changed = True
                while changed and len(indices) > 100:
                    changed = False
                    for ns in null_space:
                        if not ns:
                            continue
                        if ns.issubset(indices):
                            indices -= ns
                            changed = True
                            break
                        else:
                            inter = len(ns & indices)
                            if 2 * inter > len(ns):
                                indices ^= ns
                                changed = True
                                break
            
            if 1 <= len(indices) <= 100:
                # Verify the solution
                prod = 1
                for x in indices:
                    prod *= x
                total = prod * N
                # Check if total is a perfect square
                s = math.isqrt(total)
                if s * s == total:
                    max_val = max(indices)
                    if max_val < best_max:
                        best_max = max_val
                        best_solution = list(indices)
                    # We can break early since we're iterating M in increasing order
                    # and we want the minimum M
                    return best_solution
            # If we can't reduce to <= 100, continue to next M
    
    # If we found a solution, return it
    if best_solution is not None:
        return best_solution
    
    # Fallback: try to find a 1-element solution
    # We need x such that x * 210 is a perfect square
    # x * 210 = k^2
    # x = k^2 / 210
    # x must be an integer > 210 and <= 10210
    for k in range(math.isqrt(211 * 210) + 1, math.isqrt(10210 * 210) + 1):
        if k * k % 210 == 0:
            x = k * k // 210
            if 211 <= x <= 10210:
                # Verify
                s = math.isqrt(x * 210)
                if s * s == x * 210:
                    return [x]
    
    # Fallback: return a trivial solution if possible
    # 210 * 840 = 210 * 4 * 210 = 4 * 210^2 = (2*210)^2 = 420^2
    # So 840 works as a 1-element solution
    if 211 <= 840 <= 10210:
        s = math.isqrt(840 * 210)
        if s * s == 840 * 210:
            return [840]
    
    # Last resort: return empty list (shouldn't happen)
    return []
