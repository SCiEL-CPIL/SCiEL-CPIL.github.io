"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _check_can_add_fast(x, selected):
    """Fast check if x can be added to selected without violating condition."""
    n = len(selected)
    if n < 3:
        return True
    
    sel = np.array(selected)
    
    # Check 1: Does x divide sum of any 3 elements in selected?
    if n >= 3:
        # Generate all triples from selected
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    s = selected[i] + selected[j] + selected[k]
                    if s % x == 0:
                        return False
    
    # Check 2: Does any element a in selected divide sum of x and 2 other elements?
    if n >= 2:
        for i, a in enumerate(selected):
            # Need to pick 2 other elements from selected (excluding index i)
            other_indices = [j for j in range(n) if j != i]
            for p in range(len(other_indices) - 1):
                for q in range(p + 1, len(other_indices)):
                    j = other_indices[p]
                    k = other_indices[q]
                    s = x + selected[j] + selected[k]
                    if s % a == 0:
                        return False
    
    return True

def _check_validity(selected):
    """Check if no element divides the sum of any 3 distinct other selected elements."""
    n = len(selected)
    if n < 4:
        return True
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        if len(others) < 3:
            continue
        for t in combinations(others, 3):
            if (t[0] + t[1] + t[2]) % a == 0:
                return False
    return True

def _score_candidate(x, selected):
    """Score how 'safe' it is to add x to selected (higher = safer)."""
    # Elements with more divisors in the range are riskier
    # Large elements are generally safer as divisors
    score = x  # Prefer larger elements
    
    # Penalize if x has many small factors that could be in selected
    if selected:
        for a in selected:
            if a < x and x % a == 0:
                score -= 10  # x is divisible by a, risky
            if a > x and a % x == 0:
                score -= 5  # a is divisible by x, also risky
    
    return score

def _greedy_construct(order, deadline, check_func=None):
    """Greedy construction: add elements in given order if valid."""
    if check_func is None:
        check_func = _check_can_add_fast
    
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _check_can_add_fast(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Improve set by trying to add elements or swap elements."""
    best = list(selected)
    best_set = set(best)
    improved = True
    no_improve_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count = 0
        
        # Strategy 1: Try adding any element (prefer larger ones)
        for x in range(192, 0, -1):
            if time.time() > deadline:
                break
            if x in best_set:
                continue
            if _check_can_add_fast(x, best):
                best.append(x)
                best_set.add(x)
                improved = True
                no_improve_count = 0
                break  # Restart with new state
        
        if time.time() > deadline:
            break
        
        # Strategy 2: Try single swaps (remove one, add one or more)
        if len(best) >= 1:
            # Try removing each element and see if we can add more
            for remove_idx in range(len(best)):
                if time.time() > deadline:
                    break
                
                temp_best = [e for i, e in enumerate(best) if i != remove_idx]
                temp_set = set(temp_best)
                
                # Try to add elements to temp_best
                added_any = False
                for x in range(192, 0, -1):
                    if time.time() > deadline:
                        break
                    if x in temp_set:
                        continue
                    if _check_can_add_fast(x, temp_best):
                        temp_best.append(x)
                        temp_set.add(x)
                        added_any = True
                
                # Check if this swap improved things
                if len(temp_best) > len(best):
                    best = temp_best
                    best_set = temp_set
                    improved = True
                    no_improve_count = 0
                    break
                elif len(temp_best) == len(best) and added_any:
                    # Same size but different composition - might help future additions
                    # Only accept if it allows adding more in subsequent rounds
                    pass
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
    
    return best

def _multi_start_search(deadline):
    """Run multiple greedy starts with different orderings."""
    best = []
    
    # Start 1: Descending order
    order1 = list(range(192, 0, -1))
    candidate1 = _greedy_construct(order1, deadline)
    if len(candidate1) > len(best):
        best = candidate1
    
    # Start 2: Only large numbers first (150-192)
    order2 = list(range(192, 149, -1)) + list(range(149, 0, -1))
    candidate2 = _greedy_construct(order2, deadline)
    if len(candidate2) > len(best):
        best = candidate2
    
    # Start 3: Numbers with few divisors (primes and near-primes)
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in range(192, 0, -1) if is_prime(x)]
    non_primes = [x for x in range(192, 0, -1) if not is_prime(x)]
    order3 = primes + non_primes
    candidate3 = _greedy_construct(order3, deadline)
    if len(candidate3) > len(best):
        best = candidate3
    
    # Start 4: Randomized with bias toward larger numbers
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        order = list(range(192, 0, -1))
        # Partial shuffle: keep large numbers relatively ordered
        for i in range(150, 192):
            j = random.randint(i, 191)
            order[i], order[j] = order[j], order[i]
        candidate = _greedy_construct(order, deadline)
        if len(candidate) > len(best):
            best = candidate
    
    return best

def construct():
    start_time = time.time()
    # Be conservative with deadline to ensure we return before 90s
    deadline = start_time + 80.0
    
    # Multi-start greedy
    best = _multi_start_search(deadline)
    
    # Local search on the best found
    if time.time() < deadline - 5.0:
        improved = _local_search(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # If we still have time, try local search from a few other good candidates
    if time.time() < deadline - 10.0 and len(best) < 48:
        # Try local search from descending order result
        order_desc = list(range(192, 0, -1))
        cand_desc = _greedy_construct(order_desc, deadline)
        if len(cand_desc) >= len(best) - 2:
            improved2 = _local_search(cand_desc, deadline)
            if len(improved2) > len(best):
                best = improved2
    
    # Final fallback
    if len(best) == 0:
        best = [192]
    
    # Ensure we don't exceed maximum of 48
    if len(best) > 48:
        best = best[:48]
    
    return best
# EVOLVE-BLOCK-END
