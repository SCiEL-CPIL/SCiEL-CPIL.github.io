import time
import math
import random

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = 128
        g = r = q = 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g != 1:
                    break
        if g != n:
            return g

def _factor(n, result):
    if n <= 1:
        return
    if _is_prime_mr(n):
        result.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, result)
    _factor(n // d, result)

# ---------- Sieve for small primes ----------
def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Precompute small primes for trial division in the segmented window
    # We need primes up to a reasonable limit to quickly reduce numbers
    # Let's use primes up to 100,000 for the initial reduction
    SIEVE_LIMIT = 100000
    small_primes = _sieve(SIEVE_LIMIT)
    
    # We will search for primes p.
    # For each p, we check if p^2+1 ... p^2+k are p-smooth.
    # A number n is p-smooth if all its prime factors are <= p.
    
    # Strategy:
    # 1. Iterate through primes p.
    # 2. For a given p, create an array of values V[i] = p^2 + i for i=1..200.
    # 3. Divide V[i] by all small primes q <= p.
    # 4. If V[i] becomes 1, it is p-smooth.
    # 5. If V[i] > 1, it must be checked further. Since we divided by all q <= p,
    #    any remaining factor must be > p. 
    #    Wait, the condition is "all prime factors at most p".
    #    So if after dividing by all q <= p, the remainder is 1, it's good.
    #    If the remainder is > 1, it has a prime factor > p (since all factors <= p were removed).
    #    Therefore, it is NOT p-smooth.
    #    
    #    This implies we don't even need Pollard-Brent for the check!
    #    We just need to trial divide by all primes q <= p.
    #    
    #    However, trial dividing 200 numbers of size ~p^2 by all primes up to p 
    #    is O(200 * pi(p)) operations.
    #    For p = 10^6, pi(p) ~ 78,000. 200 * 78,000 = 15,600,000 divisions.
    #    This is feasible in Python? 15M divisions might take ~1-2 seconds.
    #    We can check ~100 primes per second? No, 15M ops is likely > 1s in pure Python loops.
    #    
    #    Optimization:
    #    We can use a segmented sieve approach for the window [p^2+1, p^2+200].
    #    But the divisors change with p.
    #    
    #    Alternative Optimization:
    #    Notice that if p is small, pi(p) is small.
    #    If p is large, the probability that p^2+i is p-smooth drops?
    #    Actually, for large p, p^2+i is often prime or has large factors.
    #    The condition "all factors <= p" is very restrictive for large p.
    #    For p=3, 3^2+1=10 (2,5) -> 5>3, fail. 3^2+2=11 (prime) -> 11>3, fail.
    #    Wait, 10 = 2*5. 5 > 3. So k=0?
    #    Let's check p=2. 2^2+1=5 (prime, 5>2) fail.
    #    p=5. 25+1=26 (2,13) 13>5 fail.
    #    p=7. 49+1=50 (2,5) ok. 49+2=51 (3,17) 17>7 fail. k=1.
    #    p=11. 121+1=122 (2,61) 61>11 fail.
    #    p=13. 169+1=170 (2,5,17) 17>13 fail.
    #    p=17. 289+1=290 (2,5,29) 29>17 fail.
    #    p=19. 361+1=362 (2,181) fail.
    #    p=23. 529+1=530 (2,5,53) fail.
    #    p=29. 841+1=842 (2,421) fail.
    #    p=31. 961+1=962 (2,13,37) 37>31 fail.
    #    p=37. 1369+1=1370 (2,5,137) fail.
    #    p=41. 1681+1=1682 (2,29,29) 29<=41. OK.
    #         1682 = 2 * 29^2. Factors: 2, 29. All <= 41.
    #         1681+2=1683 = 3 * 561 = 3 * 3 * 187 = 9 * 11 * 17. Factors: 3,11,17. All <= 41. OK.
    #         1681+3=1684 = 4 * 421. 421 is prime. 421 > 41. Fail.
    #         So k=2 for p=41.
    #
    #    So we are looking for primes p where the first few integers after p^2 are p-smooth.
    #    
    #    Key Insight:
    #    We can precompute the "bad" primes for a given p? No.
    #    
    #    Let's stick to the trial division method but optimize the loop.
    #    We can break early if the remainder becomes 1.
    #    Also, we only need to check primes q <= p.
    #    
    #    To speed up, we can use NumPy for the vectorized trial division?
    #    NumPy is available.
    #    
    #    Let's try a hybrid:
    #    1. Generate primes p up to some limit (e.g., 10^5 or 10^6).
    #    2. For each p, check k.
    #    
    #    Using NumPy:
    #    Create array of 200 values.
    #    For each prime q <= p:
    #       mask = (values % q == 0)
    #       values[mask] //= q (repeatedly?)
    #       Actually, we need to divide out ALL powers of q.
    #       This is hard to vectorize efficiently for "divide until not divisible".
    #       But we can do:
    #       while any(values % q == 0):
    #           mask = values % q == 0
    #           values[mask] //= q
    #       This inner loop runs at most log_q(max_val) times.
    #    
    #    Let's implement this with NumPy.

    # Generate primes up to 1,000,000 for the search space of p
    # We'll search p up to 1,000,000 first. If time permits, go higher.
    MAX_P_SEARCH = 1000000
    primes_list = _sieve(MAX_P_SEARCH)
    
    best_k = 0
    best_p = 2
    random.seed(42)
    
    # Precompute small primes for the reduction step (primes up to MAX_P_SEARCH)
    # We already have primes_list.
    
    # We need to check p^2 + i for i=1..200.
    # For each p, we need to divide by all primes q <= p.
    
    # Optimization: 
    # If p is small, the number of divisors is small.
    # If p is large, the number of divisors is large.
    # 
    # Let's process primes in increasing order.
    
    for p in primes_list:
        if time.time() > deadline:
            break
        
        p_sq = p * p
        # Create NumPy array of p^2 + 1 ... p^2 + 200
        # Using int64 to avoid overflow? p^2 can be up to 10^12, fits in int64.
        # Python's numpy int64 is signed, max ~9.2e18. Safe.
        
        # We only need to check up to k=200.
        # But we can stop early if we find a failure.
        
        # Initialize the remainders
        # We will check sequentially i=1 to 200.
        # If we use NumPy, we can check all 200 at once?
        # But we need the FIRST failure to determine k.
        # So we need to find the smallest i such that p^2+i is NOT p-smooth.
        
        # Let's create an array of the 200 numbers.
        # vals = p^2 + [1..200]
        # For each prime q <= p:
        #    Divide vals by q as much as possible.
        #    If any val becomes 1, it's good.
        #    If any val > 1 remains after processing ALL primes q <= p, it's bad.
        #    Wait, if a val > 1 remains, it has a factor > p.
        #    So it is NOT p-smooth.
        #
        # So the algorithm is:
        # 1. vals = p^2 + arange(1, 201)
        # 2. For each prime q in primes_list where q <= p:
        #      while any(vals % q == 0):
        #          mask = vals % q == 0
        #          vals[mask] //= q
        # 3. After processing all q <= p:
        #      Check which vals are 1.
        #      The first index i (1-based) where vals[i-1] != 1 is the first failure.
        #      So k is the count of leading 1s.
        
        # This is O(200 * pi(p)) operations in the inner loop, but vectorized.
        # For p=1000, pi(p)=168. 168 * (few vector ops). Very fast.
        # For p=100000, pi(p)=9592. 9592 * (few vector ops).
        # NumPy overhead is ~1us per op. 9592 * 10 ops = 100k ops = 0.1s?
        # Maybe too slow for many large p.
        
        # We can optimize by skipping primes q where q^2 > max(vals)?
        # No, we need to divide by ALL primes q <= p.
        # Even if q > sqrt(val), if val is divisible by q, we must divide it.
        # Example: val = 2 * p. q=p. val/p = 2.
        # So we must check all q <= p.
        
        # However, we can break early in the prime loop if all vals are 1.
        
        import numpy as np
        vals = p_sq + np.arange(1, 201, dtype=np.int64)
        
        # Iterate through primes q <= p
        # We can use bisect to find the index of the last prime <= p
        # But primes_list is sorted.
        # Let's just loop and break if q > p.
        
        # Optimization: Only iterate primes up to p.
        # We can precompute indices?
        # For now, simple loop.
        
        all_one = False
        for q in primes_list:
            if q > p:
                break
            
            # Check if any val is divisible by q
            # We need to divide out ALL powers.
            # 
            # Efficient vectorized division:
            # while True:
            #    mask = vals % q == 0
            #    if not mask.any(): break
            #    vals[mask] //= q
            
            # To avoid repeated .any() calls, we can do:
            # mask = vals % q == 0
            # if mask.any():
            #    while mask.any():
            #       vals[mask] //= q
            #       mask = vals % q == 0
            
            # This is still potentially slow if many primes don't divide any val.
            # But for small q, many vals are divisible.
            # For large q, few vals are divisible.
            
            mask = vals % q == 0
            if not mask.any():
                continue
            
            while mask.any():
                vals[mask] //= q
                mask = vals % q == 0
                
            # Check if all are 1
            # We can check this periodically to break early
            # if np.all(vals == 1):
            #    all_one = True
            #    break
            
        # After processing all primes q <= p:
        # Find the first index where val != 1
        # 
        # Note: If a val is still > 1, it has a prime factor > p.
        # So it is NOT p-smooth.
        
        # Find k
        # k is the number of leading 1s.
        # 
        # We can use:
        # non_one = np.nonzero(vals != 1)[0]
        # if len(non_one) == 0: k = 200
        # else: k = non_one[0]  # 0-based index, so k is the count of 1s before it
        
        non_one = np.nonzero(vals != 1)[0]
        if len(non_one) == 0:
            k = 200
        else:
            k = non_one[0]
            
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                return {'p': best_p, 'k': best_k}
                
    # If we ran out of time or primes, return best found
    return {'p': best_p, 'k': best_k}
