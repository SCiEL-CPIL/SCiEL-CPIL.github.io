import math
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 50.0  # Safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Step 1: Generate all powerful numbers up to N_MAX
    powerful_numbers = set()
    
    # m = a^2 * b^3
    # b^3 <= N_MAX => b <= 100
    # a^2 <= N_MAX / b^3 => a <= sqrt(N_MAX / b^3)
    
    for b in range(1, 101):
        b3 = b ** 3
        if b3 > N_MAX:
            break
        max_a = int(math.isqrt(N_MAX // b3))
        for a in range(1, max_a + 1):
            m = (a * a) * b3
            if m <= N_MAX:
                powerful_numbers.add(m)
    
    # Step 2: Find the largest n in [N_MIN, N_MAX-1] such that n and n+1 are powerful
    best_n = -1
    # We want to maximize n(n+1), which is increasing with n.
    # So we just need the largest valid n.
    for n in range(N_MAX - 1, N_MIN - 1, -1):
        if n in powerful_numbers and (n + 1) in powerful_numbers:
            best_n = n
            break
        
        # Check deadline
        if time.time() > DEADLINE:
            break
            
    if best_n != -1:
        return best_n
        
    # Fallback: If no consecutive powerful numbers found (unlikely), 
    # we perform a broader search. 
    # We compute P(x) for all x and find max P(x)P(x+1).
    # This is slower, so we only do it if necessary.
    
    if time.time() > DEADLINE - 5.0:
        # Not enough time for full computation, return a heuristic guess
        # or the best powerful number found?
        # Actually, if no consecutive powerful numbers, we can't just return -1.
        # We need to return a valid n.
        # Let's return the largest powerful number n such that n < N_MAX.
        # This ensures P(n) is large, even if P(n+1) is small.
        # But we need to maximize the product.
        # A safer fallback: return N_MIN.
        return N_MIN

    # Efficient computation of Powerful Part P(x)
    # P(x) = product of p^e where e >= 2 in factorization of x.
    
    limit = N_MAX + 1
    P = [1] * limit
    
    # Sieve of Eratosthenes for primes
    is_prime = [True] * limit
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit, i):
                is_prime[j] = False
    
    primes = [i for i, p in enumerate(is_prime) if p]
    
    for p in primes:
        p2 = p * p
        if p2 >= limit:
            break
        # For each multiple of p2, determine the exact power of p
        # We iterate through powers p^k
        power = p2
        while power < limit:
            # Multiples of 'power' that are NOT multiples of 'power*p'
            # have exactly exponent k for prime p.
            next_power = power * p
            for m in range(power, limit, power):
                if m % next_power != 0:
                    P[m] *= power
            power = next_power
            
    # Find max P[n] * P[n+1]
    best_n = N_MIN
    best_val = P[N_MIN] * P[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
        if time.time() > DEADLINE:
            break
            
    return best_n
