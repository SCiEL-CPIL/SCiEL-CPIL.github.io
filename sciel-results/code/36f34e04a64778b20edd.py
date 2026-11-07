import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_valid_set(nums):
    """Check if all 2^k subset products are distinct."""
    products = {1}
    for n in nums:
        new_products = {x * n for x in products}
        if new_products & products:
            return False
        products |= new_products
    return True

def _can_add(current_products, candidate):
    """Check if adding candidate to the set preserves distinct subset products."""
    new_products = {x * candidate for x in current_products}
    return not (new_products & current_products)

def _build_base_set():
    """
    Build a strong base set using prime powers with binary-weighted exponents.
    
    Key insight: For each prime p, we can include p^e1, p^e2, ... as long as
    the exponent vectors remain linearly independent. Using exponents that are
    powers of 2 (1, 2, 4, 8, ...) ensures subset sums of exponents are all distinct.
    
    For primes in [2, 40]:
    - Prime 2: 2^1=2, 2^2=4, 2^4=16 (exponents 1,2,4; 2^8=256>40)
    - Prime 3: 3^1=3, 3^2=9 (exponents 1,2; 3^4=81>40)  
    - Prime 5: 5^1=5, 5^2=25 (exponents 1,2; 5^4=625>40)
    - Primes 7,11,13,17,19,23,29,31,37: just the prime itself
    
    Total: 3 + 2 + 2 + 9 = 16 numbers
    """
    base_set = []
    
    # Prime 2: exponents 1, 2, 4 -> 2, 4, 16
    base_set.extend([2, 4, 16])
    
    # Prime 3: exponents 1, 2 -> 3, 9
    base_set.extend([3, 9])
    
    # Prime 5: exponents 1, 2 -> 5, 25
    base_set.extend([5, 25])
    
    # Primes 7, 11, 13, 17, 19, 23, 29, 31, 37
    base_set.extend([7, 11, 13, 17, 19, 23, 29, 31, 37])
    
    return base_set

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    
    Uses a multi-strategy approach:
    1. Build a strong base set using prime powers with binary-weighted exponents.
    2. Try to add more numbers by checking if they maintain the property.
    3. Use local search with swaps to potentially improve further.
    """
    deadline = time.time() + 75  # Safety margin before 90s limit
    
    # All candidate numbers 2..40
    candidates = list(range(2, 41))
    
    # Build base set using prime powers strategy
    base_set = _build_base_set()
    
    # Verify base_set is valid
    if not _is_valid_set(base_set):
        # Fallback to just primes
        base_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_set = list(base_set)
    best_products = {1}
    for n in best_set:
        best_products = {x * n for x in best_products}
    
    best_size = len(best_set)
    
    # Try to add more numbers from remaining candidates
    remaining = [x for x in candidates if x not in best_set]
    
    for cand in remaining:
        if time.time() > deadline:
            break
        if _can_add(best_products, cand):
            best_set.append(cand)
            best_products = {x * cand for x in best_products}
            best_size = len(best_set)
    
    # Try local search: swap one element for another to potentially add more
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(best_set)
        for i in range(current_size):
            if time.time() > deadline:
                break
            for cand in remaining:
                if time.time() > deadline:
                    break
                # Try replacing best_set[i] with cand
                trial_set = best_set[:i] + [cand] + best_set[i+1:]
                if _is_valid_set(trial_set):
                    # Check if we can then add more
                    trial_products = {1}
                    for n in trial_set:
                        trial_products = {x * n for x in trial_products}
                    # Try to add any remaining
                    added_any = False
                    for extra in remaining:
                        if extra not in trial_set and _can_add(trial_products, extra):
                            trial_set.append(extra)
                            trial_products = {x * extra for x in trial_products}
                            added_any = True
                    if len(trial_set) > current_size:
                        best_set = trial_set
                        best_products = trial_products
                        remaining = [x for x in candidates if x not in best_set]
                        best_size = len(best_set)
                        improved = True
                        break
            if improved:
                break
    
    # Final verification
    if not _is_valid_set(best_set):
        # Fallback to known valid set
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure all are Python ints
    result = [int(x) for x in best_set]
    
    return result

# EVOLVE-BLOCK-END
