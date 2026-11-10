import math

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def construct():
    # Generate primes in [2, 128]
    primes = [x for x in range(2, 129) if is_prime(x)]
    
    # Generate squares of primes in [2, 128]
    squares = []
    for p in primes:
        sq = p * p
        if sq <= 128:
            squares.append(sq)
            
    # Combine and sort
    candidates = sorted(set(primes + squares))
    
    return candidates
