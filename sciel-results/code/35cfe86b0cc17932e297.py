"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def _can_add(prods, x):
    """
    Check if adding x to a set with existing subset products 'prods'
    keeps all subset products distinct.
    
    The new products will be {p * x for p in prods}.
    For the new set to be valid, none of these new products can collide
    with each other (guaranteed if x != 1 and prods distinct? No, p1*x == p2*x implies p1==p2, so yes)
    AND none of them can collide with existing products in 'prods'.
    
    So we just need: for all p in prods, p * x not in prods.
    """
    for p in prods:
        if p * x in prods:
            return False
    return True

def _backtrack(prods, candidates, current_set, best_set):
    """
    Recursive backtracking to find the largest subset.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return
    
    # Update best if we have a larger set
    if len(current_set) > len(best_set):
        best_set[:] = current_set
    
    if not candidates:
        return
    
    # Heuristic: try candidates that are "easier" to add first? 
    # Or just iterate. To maximize chance of finding a large set,
    # we should explore adding elements first.
    
    # Optimization: If we can't beat best_set with remaining candidates, prune
    if len(current_set) + len(candidates) <= len(best_set):
        return

    for i, x in enumerate(candidates):
        # Check if we can add x
        if _can_add(prods, x):
            # Generate new products
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            
            # Recurse
            current_set.append(x)
            _backtrack(new_prods, candidates[i+1:], current_set, best_set)
            current_set.pop()
            
            # Check deadline again
            if time.time() > _DEADLINE:
                return
        else:
            # Cannot add x, skip it (implicit in loop structure, we just don't recurse)
            pass

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    """
    # Candidates are 2..40
    candidates = list(range(2, 41))
    
    # Initial state: empty set, product {1}
    initial_prods = {1}
    current_set = []
    best_set = []
    
    # We want to maximize size. 
    # A simple greedy might work well as a baseline, but backtracking allows us to fix greedy mistakes.
    # However, full backtracking on 39 items is 2^39 which is too big.
    # We need a strong heuristic or pruning.
    
    # Pruning idea: If we sort candidates, maybe we should prefer primes first? 
    # Primes are "safe" to add unless they create a collision with a composite already added?
    # Actually, if we have a set of primes P, any composite C = p1*p2... 
    # If C is added, does it collide?
    # Products are unique if the set is multiplicatively independent.
    # {a1, ..., ak} is multiplicatively independent if no product of a subset equals product of another disjoint subset.
    # This is equivalent to: log(a_i) are linearly independent over rationals? No, that's for real numbers.
    # For integers, it's about prime factorization vectors being linearly independent over Z (or Q).
    
    # Let's use a hybrid:
    # 1. Try to find a large set via backtracking with a good ordering.
    # Ordering: Primes first are usually safe. Composites are risky.
    # Let's sort candidates such that "safe" ones come first.
    # A number x is "safe" to add to a set S if x is not divisible by any product of a subset of S?
    # Actually, the condition `p * x in prods` is the exact check.
    
    # To make backtracking efficient, let's order candidates by "difficulty".
    # Primes are generally easier to add than composites because they introduce a new prime factor dimension.
    # Composites reuse existing dimensions.
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
        
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    # Order: Primes first, then composites.
    # Within composites, maybe smaller ones first? Or ones with fewer prime factors?
    # Let's just put primes first.
    ordered_candidates = primes + composites
    
    # Run backtracking
    _backtrack(initial_prods, ordered_candidates, current_set, best_set)
    
    # If backtracking didn't finish or found nothing, ensure we return a valid set.
    # best_set is a list of ints.
    if not best_set:
        # Fallback to just primes if something went wrong (shouldn't happen)
        best_set = primes
        
    # Ensure types are correct
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
