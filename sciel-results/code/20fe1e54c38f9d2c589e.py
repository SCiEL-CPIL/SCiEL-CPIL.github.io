"""Rewritten program for reciprocal dissociated subset search."""
import time
import random
import math
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _all_subset_sums_fractions(elements):
    """Return the set of all subset sums (as Fractions) including 0."""
    n = len(elements)
    sums = [Fraction(0)]
    for x in elements:
        new_sums = [s + x for s in sums]
        sums.extend(new_sums)
    return set(sums)

def _is_dissociated(elements):
    """Check if all subset sums are distinct."""
    sums = _all_subset_sums_fractions(elements)
    return len(sums) == (1 << len(elements))

def _min_pairwise_diff(elements):
    """
    Compute the minimum absolute difference between any two distinct subset sums.
    Returns 0 if there's a collision, otherwise the minimum positive difference.
    For performance, we work with scaled integers.
    """
    n = len(elements)
    if n == 0:
        return 1
    # Compute LCM of denominators
    lcm_val = 1
    for i in elements:
        g = math.gcd(lcm_val, i)
        lcm_val = lcm_val * i // g
    
    # Convert to integer weights
    weights = [lcm_val // i for i in elements]
    
    # Generate all subset sums as integers
    sums = [0]
    for w in weights:
        sums = sums + [s + w for s in sums]
    
    if len(set(sums)) != len(sums):
        return 0
    
    # Find minimum pairwise difference
    sums_sorted = sorted(sums)
    min_diff = min(b - a for a, b in zip(sums_sorted, sums_sorted[1:]))
    return min_diff

def _is_dissociated_fast(elements):
    """Fast check using integer scaling."""
    if not elements:
        return True
    n = len(elements)
    if n > 16:  # Too many subsets for fast check
        return _is_dissociated(elements)
    
    lcm_val = 1
    for i in elements:
        g = math.gcd(lcm_val, i)
        lcm_val = lcm_val * i // g
    
    weights = [lcm_val // i for i in elements]
    
    sums = [0]
    for w in weights:
        sums = sums + [s + w for s in sums]
    
    return len(set(sums)) == len(sums)

def construct():
    """
    Find the largest subset of [1,22] such that all reciprocal subset sums are distinct.
    Uses simulated annealing / local search to maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    all_elements = list(range(1, 23))
    
    # Start with a known good set
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Try to greedily build a larger set
    # Sort elements by some heuristic
    random.seed(42)
    
    # Try multiple random restarts with local search
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random initial set of size between 10 and 15
        target_size = random.randint(10, min(15, len(all_elements)))
        current_set = set(random.sample(all_elements, target_size))
        
        # Ensure it's dissociated, if not, remove elements
        while not _is_dissociated_fast(list(current_set)):
            if not current_set:
                break
            # Remove a random element
            current_set.remove(random.choice(list(current_set)))
        
        current_size = len(current_set)
        
        if current_size > best_size:
            best_set = list(current_set)
            best_size = current_size
        
        # Local search: try to add elements or swap
        for step in range(500):
            if time.time() > deadline:
                break
            
            current_list = list(current_set)
            
            # Try adding a random element not in the set
            available = [x for x in all_elements if x not in current_set]
            if available:
                add_elem = random.choice(available)
                new_set = set(current_set)
                new_set.add(add_elem)
                if _is_dissociated_fast(list(new_set)):
                    current_set = new_set
                    current_size += 1
                    if current_size > best_size:
                        best_set = list(current_set)
                        best_size = current_size
            
            # Try swapping: remove one, add another
            if current_set and available:
                remove_elem = random.choice(list(current_set))
                add_elem = random.choice(available)
                if remove_elem != add_elem:
                    new_set = set(current_set)
                    new_set.remove(remove_elem)
                    new_set.add(add_elem)
                    if _is_dissociated_fast(list(new_set)):
                        current_set = new_set
                        if current_size > best_size:
                            best_set = list(current_set)
                            best_size = current_size
            
            # Try removing an element and adding two (net +1)
            if current_size < 20 and len(available) >= 2:
                remove_elem = random.choice(list(current_set))
                add_elems = random.sample(available, 2)
                new_set = set(current_set)
                new_set.remove(remove_elem)
                new_set.update(add_elems)
                if _is_dissociated_fast(list(new_set)):
                    current_set = new_set
                    current_size += 1
                    if current_size > best_size:
                        best_set = list(current_set)
                        best_size = current_size
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Final verification and return
    result = sorted(best_set)
    if not result:
        result = [1]
    return result

# EVOLVE-BLOCK-END
