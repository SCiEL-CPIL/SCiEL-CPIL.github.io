import math

def find_gaussian_prime(p):
    # Find a, b such that a^2 + b^2 = p
    # p is prime, p % 4 == 1
    a = 1
    while a * a < p:
        b2 = p - a * a
        b = int(math.isqrt(b2))
        if b * b == b2:
            return (a, b)
        a += 1
    return (1, 1) # Should not happen

def construct():
    # 1. Generate primes p % 4 == 1
    primes = []
    candidate = 5
    while len(primes) < 50: # Enough primes
        # Simple primality test
        is_prime = True
        for i in range(2, int(math.isqrt(candidate)) + 1):
            if candidate % i == 0:
                is_prime = False
                break
        if is_prime and candidate % 4 == 1:
            primes.append(candidate)
        candidate += 1
    
    # 2. Optimize exponents
    # Budget L = ln(10^18)
    L = 18 * math.log(10)
    
    # We want to maximize sum(log(e_i + 1)) subject to sum(e_i * log(p_i)) <= L
    # Use a priority queue of next possible exponent increments
    # Item: (efficiency, prime_index, current_exponent)
    # Efficiency of adding 1 to e_i (going from k to k+1) is log((k+2)/(k+1)) / log(p_i)
    
    import heapq
    
    # Initialize
    exponents = {p: 0 for p in primes}
    heap = []
    for p in primes:
        # First increment: 0 -> 1. Gain log(2/1)/log(p)
        gain = math.log(2) / math.log(p)
        # We want to pick highest gain. Min-heap, so push -gain
        heapq.heappush(heap, (-gain, p, 0))
        
    current_cost = 0
    # We will add increments until we run out of budget
    # But we need to be careful: we can't just pick the best marginal gain if it exceeds remaining budget?
    # Actually, since we can't split, we should just pick the best available increment
    # if it fits in the remaining budget.
    # However, the "efficiency" drops.
    # A better way: just simulate.
    
    # Actually, the greedy approach of picking the single best marginal gain at each step
    # works for concave functions if we check budget.
    
    while heap:
        neg_gain, p, k = heap[0]
        gain = -neg_gain
        cost = math.log(p)
        if current_cost + cost > L:
            break
        
        # Accept this increment
        current_cost += cost
        exponents[p] = k + 1
        
        # Push next increment for this prime
        # Next k is k+1. Going to k+2.
        # Gain is log((k+3)/(k+2)) / log(p)
        next_k = k + 1
        next_gain = math.log((next_k + 2) / (next_k + 1)) / math.log(p)
        heapq.heappush(heap, (-next_gain, p, next_k))
        
        # Pop the current one
        heapq.heappop(heap)
        
    # 3. Calculate N
    N = 1
    for p, e in exponents.items():
        if e > 0:
            N *= p ** e
            
    if N > 10**18:
        # Adjust if needed (shouldn't happen with strict check, but safe)
        # Reduce largest exponent
        # For simplicity, just return a small N if this fails, but it shouldn't
        pass
        
    # 4. Generate pairs
    # Factor primes into Gaussian primes
    gaussian_primes = {}
    for p in primes:
        if exponents[p] > 0:
            a, b = find_gaussian_prime(p)
            # pi = a + bi
            gaussian_primes[p] = (a, b)
            
    pairs = set()
    
    # We need to generate all z = prod pi_i^a_i * conj(pi_i)^(e_i - a_i)
    # This is a recursive generation or iterative product.
    
    # To avoid huge recursion, we can iterate over all combinations of a_i.
    # Number of combinations is prod(e_i + 1).
    
    # Get list of active primes
    active_primes = [p for p in primes if exponents[p] > 0]
    
    # If no active primes, N=1, but N>=2.
    
    # We will generate the real and imaginary parts.
    # Start with z = 1 + 0i
    # For each prime, we multiply by pi^a * conj(pi)^(e-a)
    # This is equivalent to summing over all choices.
    
    # Since the number of combinations might be large (thousands), we can just loop.
    
    # Precompute powers?
    # For each p, we have pi = a+bi.
    # We need pi^k for k=0..e.
    # And conj(pi)^k for k=0..e.
    
    # Let's just compute the product for each combination.
    
    # To be efficient, we can precompute the list of factors for each prime.
    # factors[p] = [ pi^0 * conj(pi)^e, pi^1 * conj(pi)^(e-1), ..., pi^e * conj(pi)^0 ]
    # Note: pi^k * conj(pi)^(e-k) has modulus p^e.
    
    factor_lists = []
    for p in active_primes:
        a, b = gaussian_primes[p]
        e = exponents[p]
        pi = complex(a, b)
        conj_pi = complex(a, -b)
        
        # Precompute powers of pi and conj_pi
        pi_pows = [1]
        for _ in range(e):
            pi_pows.append(pi_pows[-1] * pi)
            
        conj_pi_pows = [1]
        for _ in range(e):
            conj_pi_pows.append(conj_pi_pows[-1] * conj_pi)
            
        # Construct the list of factors
        # For a_i in 0..e:
        # factor = pi^a_i * conj_pi^(e-a_i)
        current_factors = []
        for ai in range(e + 1):
            f = pi_pows[ai] * conj_pi_pows[e - ai]
            current_factors.append(f)
            
        factor_lists.append(current_factors)
        
    # Now generate all combinations
    # This is a Cartesian product of factor_lists
    # We can use itertools.product
    
    import itertools
    
    # To save memory, we can do it recursively or just use product if count is small.
    # Max count is maybe 10^4 or 10^5.
    
    for combo in itertools.product(*factor_lists):
        z = 1
        for f in combo:
            z *= f
        x = int(round(z.real))
        y = int(round(z.imag))
        
        # Check if x^2 + y^2 == N
        # Due to floating point or integer rounding, we should verify.
        # But z is constructed exactly from integer Gaussian primes, so x,y are integers.
        # However, complex multiplication in Python uses float? No, complex(a,b) with ints stays as floats?
        # Python complex numbers are float64.
        # 10^18 is too big for float64 precision (53 bits ~ 15-16 digits).
        # We MUST use integer arithmetic for Gaussian integers.
        
    # REDO: Use tuple (real, imag) for Gaussian integers.
    
    def gmul(z1, z2):
        a, b = z1
        c, d = z2
        return (a*c - b*d, a*d + b*c)
        
    def gpow(z, k):
        if k == 0:
            return (1, 0)
        result = (1, 0)
        base = z
        while k > 0:
            if k % 2 == 1:
                result = gmul(result, base)
            base = gmul(base, base)
            k //= 2
        return result
        
    # Rebuild factor lists with tuples
    factor_lists = []
    for p in active_primes:
        a, b = gaussian_primes[p]
        e = exponents[p]
        pi = (a, b)
        conj_pi = (a, -b)
        
        pi_pows = [gpow(pi, i) for i in range(e + 1)]
        conj_pi_pows = [gpow(conj_pi, i) for i in range(e + 1)]
        
        current_factors = []
        for ai in range(e + 1):
            f = gmul(pi_pows[ai], conj_pi_pows[e - ai])
            current_factors.append(f)
            
        factor_lists.append(current_factors)
        
    for combo in itertools.product(*factor_lists):
        z = (1, 0)
        for f in combo:
            z = gmul(z, f)
        x, y = z
        x = abs(x)
        y = abs(y)
        
        if x == 0 or y == 0:
            continue
            
        # Ensure a <= b
        if x > y:
            x, y = y, x
            
        # Check if x^2 + y^2 == N
        if x*x + y*y == N:
            pairs.add((x, y))
            
    # Convert to list of lists
    pair_list = [[x, y] for x, y in pairs]
    
    return {'N': N, 'pairs': pair_list}
