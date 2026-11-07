"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _squarefree_part(n):
    """Return the squarefree part of n and the multiplier k such that n = k^2 * sf."""
    sf = 1
    k = 1
    for p in range(2, int(math.isqrt(n)) + 1):
        while n % (p * p) == 0:
            n //= (p * p)
            k *= p
    # remaining n is squarefree part contribution
    sf = n
    return sf, k

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    n = P['n']  # 210
    
    # Strategy: search for pairs (a, b) with 210 < a <= b <= M
    # such that a*b*n is a perfect square.
    # For a pair to work: squarefree(a*b*n) == 1
    # i.e., squarefree(a) * squarefree(b) * squarefree(n) must be a perfect square
    # Since n=210 is squarefree, we need squarefree(a*b) = squarefree(n) = 210
    # More precisely: the product a*b*n must have all even exponents.
    
    # For a pair (a,b): a*b*210 must be a perfect square.
    # Let's just check directly.
    
    best_raw = None
    best_pair = None
    
    # Search pairs from smallest possible max upward
    # max value M means both a and b are in (210, M]
    # We want to minimize M = max(a,b) = b (since a <= b)
    
    # For each candidate max b from 211 upward, check if there exists a in (210, b]
    # such that a*b*210 is a perfect square.
    
    # Precompute squarefree parts for efficiency
    # Actually, let's just iterate and check.
    
    # Start with known good answer as fallback
    fallback = [224, 240]
    
    # Try to find a pair with max < 240
    for b in range(211, 240):
        if time.time() > deadline:
            break
        for a in range(211, b + 1):
            prod = a * b * n
            if _is_perfect_square(prod):
                best_raw = b - n
                best_pair = [a, b]
                break
        if best_pair is not None:
            break
    
    # If we found something better, use it
    if best_pair is not None:
        return list(best_pair)
    
    # Otherwise, try slightly larger ranges
    # Check if 224, 240 still works
    if _is_perfect_square(224 * 240 * n):
        return [224, 240]
    
    # Broader search for pairs up to some limit
    for b in range(241, min(300, P['maximum_window'] + 1)):
        if time.time() > deadline:
            break
        for a in range(211, b + 1):
            prod = a * b * n
            if _is_perfect_square(prod):
                return [a, b]
    
    # Fallback: use single element 4*210 = 840 (product = 840*210 = 176400 = 420^2)
    # Actually 4*n * n = 4*n^2 = (2n)^2, so [4*n] works
    return [4 * n]
# EVOLVE-BLOCK-END
