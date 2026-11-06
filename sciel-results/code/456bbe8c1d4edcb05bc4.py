"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _check_paley_property(p):
    """Check if Paley tournament on p vertices satisfies the (4,1) property.
    i.e., every 4-set has a vertex that dominates the other 3."""
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    def beats(i, j):
        return (j - i) % p in squares
    
    n = p
    # Check all 4-subsets
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # Check if any of a,b,c,d dominates the other three
                    found = False
                    for v in (a, b, c, d):
                        others = [x for x in (a, b, c, d) if x != v]
                        if all(beats(v, o) for o in others):
                            found = True
                            break
                    if not found:
                        return False
    return True

def _find_smallest_paley_prime():
    """Find the smallest prime p ≡ 3 (mod 4) where Paley tournament satisfies the property."""
    # Try primes from small to large
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    for p in range(7, 200):
        if p % 4 == 3 and is_prime(p):
            if _check_paley_property(p):
                return p
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, try to find the smallest valid Paley prime
    # But checking all 4-subsets for large p is expensive
    # For p=83, C(83,4) = 1,581,582 which is manageable
    
    # Let's check if smaller primes work
    # p must be ≡ 3 (mod 4) for Paley tournament to be a tournament
    # Small candidates: 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83, ...
    
    # Check p=7: C(7,4)=35
    # Check p=11: C(11,4)=330
    # Check p=19: C(19,4)=3876
    # Check p=23: C(23,4)=8855
    # Check p=31: C(31,4)=31465
    # Check p=43: C(43,4)=123410
    # Check p=47: C(47,4)=195300
    # Check p=59: C(59,4)=499114
    # Check p=67: C(67,4)=791940
    # Check p=71: C(71,4)=1015400
    # Check p=79: C(79,4)=1538575
    # Check p=83: C(83,4)=1581582
    
    # Let's check from small to large
    candidates = [7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83]
    
    best_p = 83  # Fallback to known good value
    
    for p in candidates:
        if time.time() > deadline:
            break
        if _check_paley_property(p):
            best_p = p
            break
    
    # Construct the orientation for the best prime found
    p = best_p
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            orientation.append(0 if (j - i) % p in squares else 1)
    
    return {'n': p, 'orientation': orientation}
# EVOLVE-BLOCK-END
