"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
N = 210
MAX_VAL = 10210
MAX_SIZE = 100
DEADLINE = 90.0
SAFETY_MARGIN = 2.0

def squarefree_kernel(n):
    """Compute the squarefree kernel of n."""
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count % 2 == 1:
                result *= d
        d += 1
    if temp > 1:
        result *= temp
    return result

def is_perfect_square(n):
    """Check if n is a perfect square using integer square root."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # The optimal solution is a single number a such that 210*a is a perfect square.
    # Since 210 = 2*3*5*7 is squarefree, a must be 210 * k^2 for some integer k >= 2.
    # The smallest such a > 210 is 210 * 4 = 840, giving max - 210 = 630.
    
    # Verify that 210 * 840 is a perfect square
    a = 840
    product = N * a
    if not is_perfect_square(product):
        # Fallback: search for the smallest valid single number
        best_a = None
        best_val = MAX_VAL + 1
        for k in range(2, 200):
            candidate = N * k * k
            if candidate > MAX_VAL:
                break
            if candidate > N:
                prod = N * candidate
                if is_perfect_square(prod):
                    if candidate < best_val:
                        best_val = candidate
                        best_a = candidate
                    break
        if best_a is None:
            # Absolute fallback
            return [840]
        return [best_a]
    
    # Double-check: could a pair give a smaller max?
    # For pair (a,b) with a <= b, we need 210*a*b = square, so a*b = 210*k^2.
    # Both a,b > 210, so a*b > 44100, thus 210*k^2 > 44100, k^2 > 210, k >= 15.
    # For k=15: P = 210*225 = 47250. Need a|47250, 210 < a <= sqrt(47250)~217.
    # Check if any such a exists with b=47250/a > 210.
    # If found, max = b. Compare with 840.
    
    # Let's do a quick check for small k values to see if any pair beats 840
    found_better_pair = False
    for k in range(15, 50):
        if time.time() > deadline:
            break
        P = N * k * k
        if P > MAX_VAL * MAX_VAL:
            break
        # a must divide P, 210 < a <= sqrt(P), and b = P/a > 210
        sqrt_P = math.isqrt(P)
        if sqrt_P < 211:
            continue
        # Check divisors of P in range (210, sqrt_P]
        # For efficiency, only check if P has small divisors
        # Actually, let's just check a few values
        for a in range(211, min(sqrt_P + 1, 300)):  # Limit search for speed
            if P % a == 0:
                b = P // a
                if b > N and b <= MAX_VAL and b != a:
                    if b < 840:
                        # Found a better pair
                        return [a, b]
                    found_better_pair = True
                    break
    
    # The single number 840 gives raw value 630
    return [a]
# EVOLVE-BLOCK-END
