import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def time_left():
        return deadline - time.time()
        
    def find_sum_sq_rep(p):
        """Find a² + b² = p for prime p ≡ 1 (mod 4)"""
        limit = int(math.isqrt(p)) + 1
        for a in range(1, limit):
            b2 = p - a*a
            if b2 < 0:
                break
            b = math.isqrt(b2)
            if b*b == b2:
                return (a, b)
        raise ValueError(f"No sum of squares found for {p}")
        
    # Use smallest primes 1 mod 4 to maximize the number of factors
    # N = product of these primes
    # Number of representations as sum of two squares = 2^k where k is number of distinct primes 1 mod 4
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193]
    
    # Check which subset fits in 10^18
    N = 1
    selected_primes = []
    for p in primes_1mod4:
        if N * p <= 10**18:
            N *= p
            selected_primes.append(p)
        else:
            break
    
    # Find sum of squares representation for each prime
    reps = {}
    for p in selected_primes:
        reps[p] = find_sum_sq_rep(p)
        
    # Generate all 2^k Gaussian integer combinations
    # For each prime p with p = a² + b² = (a+bi)(a-bi), we choose either (a+bi) or (a-bi)
    # The product gives a Gaussian integer z = x + yi where x² + y² = N
    # Each such z gives a representation N = x² + y²
    # Note: x² and y² are always powerful (perfect squares)
    
    # Start with [(1, 0)]
    states = [(1, 0)]
    
    for p in selected_primes:
        a, b = reps[p]
        g_plus = (a, b)
        g_minus = (a, -b)
        
        new_states = []
        for (r, i) in states:
            # Multiply by g_plus: (r+ii)(a+bi) = (ra - ib) + (rb + ia)i
            nr1 = r*a - i*b
            ni1 = r*b + i*a
            new_states.append((nr1, ni1))
            
            # Multiply by g_minus: (r+ii)(a-bi) = (ra + ib) + (ib - ia)i
            nr2 = r*a + i*b
            ni2 = r*b - i*a
            new_states.append((nr2, ni2))
        
        states = new_states
        
        if time_left() < 5:
            break
            
    # Extract pairs from Gaussian integers
    # N = x² + y² where x, y are integers
    # a = x², b = y², both are powerful (perfect squares)
    seen_pairs = set()
    pairs = []
    
    for (x, y) in states:
        a = x * x
        b = y * y
        if a > b:
            a, b = b, a
        if a + b != N:
            continue
        if (a, b) not in seen_pairs:
            seen_pairs.add((a, b))
            pairs.append([a, b])
    
    # Sort for consistency
    pairs.sort()
    
    # Truncate to 1000 if needed
    if len(pairs) > 1000:
        pairs = pairs[:1000]
        
    if not pairs:
        # Fallback: N=5, 5=1+4, both powerful
        return {'N': 5, 'pairs': [[1, 4]]}
        
    return {'N': N, 'pairs': pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
