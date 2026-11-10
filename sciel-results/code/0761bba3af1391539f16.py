import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def score(n):
    # Get divisors
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    count = 0
    for j in range(len(divs) - 1):
        if gcd(divs[j], divs[j+1]) == 1:
            count += 1
    return count

def construct():
    # Greedy search for primes
    # Start with 2
    primes = [2]
    # Try to add primes
    # We need to check if adding a prime improves the score
    # Limit primes to < 100 for speed, or just try next few primes
    
    # Actually, let's just try a few specific configurations
    # 1. Primorial P15
    p15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n15 = 1
    for p in p15:
        n15 *= p
    
    # 2. P14 (drop 47)
    p14 = p15[:14]
    n14 = 1
    for p in p14:
        n14 *= p
        
    # 3. P15 with 47 replaced by 53? No, 53 is bigger.
    # 4. P15 with 43, 47 replaced by larger primes?
    
    # Let's try to find a better set via simple search
    # Start with P10
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    n_base = 1
    for p in base_primes:
        n_base *= p
        
    best_n = n15
    best_s = score(n15)
    
    # Try replacing the largest prime with a larger one
    # Or adding a prime if space allows (not possible for P15)
    
    # Try P14 + large prime?
    # P14 = 13082761331670030
    # Max prime = 10^18 / P14 ~ 76437
    # Try primes near 76437
    # But score(P14) is likely lower than score(P15).
    
    # Let's just return the best of a few candidates
    candidates = [n15]
    
    # Try P13 + 2 large primes?
    p13 = p15[:13]
    n13 = 1
    for p in p13:
        n13 *= p
    # Remaining budget 10^18 / n13
    # n13 = 223092870
    # Budget ~ 4.5 * 10^12
    # Can we fit 2 primes?
    # p_a * p_b < 4.5 * 10^12
    # If p_a ~ 10^6, p_b ~ 10^6.
    # Try 1000003 * 1000033 ~ 10^12.
    n_test = n13 * 1000003 * 1000033
    if n_test <= 10**18:
        candidates.append(n_test)
        
    # Try P12 + 3 large primes?
    p12 = p15[:12]
    n12 = 1
    for p in p12:
        n12 *= p
    # n12 = 7420738134810
    # Budget ~ 1.3 * 10^11
    # 3 primes?
    # 10^4 * 10^4 * 10^3 = 10^11.
    # Try 10007 * 10009 * 10037
    n_test2 = n12 * 10007 * 10009 * 10037
    if n_test2 <= 10**18:
        candidates.append(n_test2)

    best = max(candidates, key=score)
    return best
