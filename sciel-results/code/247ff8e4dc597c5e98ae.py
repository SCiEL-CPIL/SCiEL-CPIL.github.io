"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import defaultdict

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
    
    # Precompute squarefree kernels for all candidates in range (211, 10210]
    # We need distinct integers > 210 and <= 10210
    candidates = range(211, MAX_VAL + 1)
    
    # Group candidates by their squarefree kernel
    # For the product of selected numbers * 210 to be a perfect square,
    # the squarefree kernel of (product * 210) must be 1.
    # sf(product * 210) = sf(product) * sf(210) / gcd(sf(product), sf(210))^2 ... 
    # Actually, it's easier to think in terms of prime factorization parity.
    # Let's use the property: sf(a*b) = sf(a)*sf(b) / (gcd(sf(a), sf(b))^2) is wrong.
    # Correct property: sf(a*b) = sf(sf(a) * sf(b)).
    # So we need sf(210 * prod(selected)) == 1.
    
    # Let's precompute sf for 210
    sf_210 = squarefree_kernel(210)
    
    # Strategy:
    # We want to minimize max(selected) - 210.
    # This means we want the selected numbers to be as small as possible (close to 210).
    # But we need their product * 210 to be a square.
    
    # Let's try to find a small set of numbers whose product has the right squarefree kernel.
    # If we pick numbers a1, a2, ..., ak, then sf(210 * a1 * ... * ak) must be 1.
    # This is equivalent to saying that in the prime factorization of 210 * a1 * ... * ak,
    # every prime has an even exponent.
    
    # Let's use a greedy approach:
    # 1. Start with the smallest candidates.
    # 2. Maintain the current "deficit" (the squarefree kernel of the product so far).
    # 3. Try to add a number that reduces the deficit or helps complete it.
    
    # Alternative approach:
    # Since we want to minimize the max value, let's try increasing the max value M from 211 upwards.
    # For each M, check if there exists a subset of {211, ..., M} of size <= 100 
    # such that the product * 210 is a perfect square.
    # But this is a subset product problem, which is hard.
    
    # Better approach:
    # Let's look for pairs (a, b) such that sf(210 * a * b) == 1.
    # This means sf(210 * a) == sf(b) (since sf(x*y) = 1 iff sf(x) == sf(y) for squarefree x,y).
    # Wait, sf(x*y) = 1 implies x*y is a square. This means x and y have the same squarefree part.
    # So we need sf(210 * a) == sf(b).
    
    # Let's precompute sf for all numbers in range.
    sf_cache = {}
    for i in candidates:
        sf_cache[i] = squarefree_kernel(i)
    
    # Group numbers by their squarefree kernel
    kernel_to_numbers = defaultdict(list)
    for i in candidates:
        kernel_to_numbers[sf_cache[i]].append(i)
    
    # For each number a, we need a number b such that sf(b) == sf(210 * a).
    # Let's compute target_kernel = sf(210 * a) for each a.
    # Then find the smallest b > 210 with sf(b) == target_kernel.
    # We want to minimize max(a, b).
    
    best_max = MAX_VAL
    best_pair = None
    
    # Sort candidates by value
    sorted_candidates = sorted(candidates)
    
    # For each a, compute target kernel and find smallest b
    for a in sorted_candidates:
        if time.time() > deadline:
            break
        target = squarefree_kernel(210 * a)
        if target in kernel_to_numbers:
            # Find the smallest b in kernel_to_numbers[target] that is different from a
            for b in kernel_to_numbers[target]:
                if b != a:
                    current_max = max(a, b)
                    if current_max < best_max:
                        best_max = current_max
                        best_pair = (a, b)
                    break  # We only need the smallest b for this a
    
    # If we found a pair, use it. Otherwise, try to find a single number a
    # such that 210 * a is a perfect square.
    # This means a must be of the form k^2 * sf(210) for some k.
    # sf(210) = sf(2 * 3 * 5 * 7) = 2 * 3 * 5 * 7 = 210.
    # So a = k^2 * 210.
    # The smallest such a > 210 is when k=2: a = 4 * 210 = 840.
    # Let's check if 840 is in range. Yes, 840 <= 10210.
    
    if best_pair is None:
        # Fallback: use a single number
        a = 4 * 210  # 840
        if a <= MAX_VAL:
            best_max = a
            best_pair = (a,)
        else:
            # This shouldn't happen, but just in case
            best_pair = (4 * 210,)
            best_max = 4 * 210
    
    # Verify the solution
    if len(best_pair) == 2:
        a, b = best_pair
        product = 210 * a * b
        if not is_perfect_square(product):
            # Fallback to single number
            a = 4 * 210
            best_pair = (a,)
            best_max = a
    else:
        a = best_pair[0]
        product = 210 * a
        if not is_perfect_square(product):
            # This should not happen for a = 4 * 210
            pass
    
    # Return the list of selected integers
    return list(best_pair)
# EVOLVE-BLOCK-END
