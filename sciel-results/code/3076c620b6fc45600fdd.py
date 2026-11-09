"""Improved program: exact cardinality-indexed subset-sum DP with greedy + swap search."""
import time
import random

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin before 60s limit

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def compute_sums_bitset(elements, max_sum):
    """
    Compute subset-sum bitsets indexed by cardinality.
    sums[k] is a Python int where bit s is set if sum s is achievable with exactly k elements.
    """
    n = len(elements)
    # sums[k] as a bitmask: bit i set means sum i is achievable with k elements
    sums = [0] * (n + 1)
    sums[0] = 1  # sum 0 with 0 elements
    
    for elem in elements:
        # Process in reverse to avoid using the same element twice
        for k in range(n, 0, -1):
            # sums[k] |= (sums[k-1] << elem)
            shifted = sums[k-1] << elem
            sums[k] |= shifted
    
    # Mask to max_sum bits (sums beyond max_sum don't matter for collisions within range)
    mask = (1 << (max_sum + 1)) - 1
    for k in range(n + 1):
        sums[k] &= mask
    
    return sums

def check_validity_bitset(sums, n):
    """Check that sums[i] & sums[j] == 0 for all i != j."""
    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            if sums[i] & sums[j]:
                return False
    return True

def check_validity(elements):
    """Full validity check: distinct ints in 1..256, cardinality-separated subset sums."""
    n = len(elements)
    if n == 0:
        return True
    # Check distinct and in range
    if len(set(elements)) != n:
        return False
    if any(e < 1 or e > 256 for e in elements):
        return False
    
    max_sum = sum(elements)
    sums = compute_sums_bitset(elements, max_sum)
    return check_validity_bitset(sums, n)

def greedy_construct():
    """Greedy construction: add elements one by one if validity is preserved."""
    elements = []
    # Try adding elements in increasing order
    for candidate in range(1, 257):
        if is_done():
            break
        test = elements + [candidate]
        if check_validity(test):
            elements.append(candidate)
        if is_done():
            break
    return elements

def swap_search(elements, max_iterations=50000):
    """
    Try to improve the set by swapping one element out and one in.
    Also try adding new elements.
    """
    current = list(elements)
    n = len(current)
    current_sums = compute_sums_bitset(current, sum(current))
    current_valid = check_validity_bitset(current_sums, n)
    
    if not current_valid:
        return current
    
    used = set(current)
    improved = True
    iterations = 0
    
    while improved and not is_done() and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try adding a new element
        for candidate in range(1, 257):
            if candidate in used:
                continue
            test = current + [candidate]
            if check_validity(test):
                current = test
                used.add(candidate)
                current_sums = compute_sums_bitset(current, sum(current))
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one element, add another
        for i in range(len(current)):
            if is_done():
                break
            removed = current[i]
            without = current[:i] + current[i+1:]
            
            # Check if removing makes it valid (it should, since subset of valid set)
            # Actually removing an element from a valid set keeps it valid
            for candidate in range(1, 257):
                if candidate in used or candidate == removed:
                    continue
                test = without + [candidate]
                if check_validity(test):
                    current = test
                    used.discard(removed)
                    used.add(candidate)
                    current_sums = compute_sums_bitset(current, sum(current))
                    improved = True
                    break
            
            if improved:
                break
    
    return current

def randomized_restart(elements, num_restarts=5):
    """Try randomized perturbations to escape local optima."""
    best = list(elements)
    best_n = len(best)
    
    for _ in range(num_restarts):
        if is_done():
            break
        
        # Perturb: remove 1-3 random elements, then try to rebuild
        perturbed = list(best)
        num_remove = min(len(perturbed), random.randint(1, 3))
        for _ in range(num_remove):
            if perturbed:
                idx = random.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
        
        # Rebuild greedily
        used = set(perturbed)
        for candidate in range(1, 257):
            if is_done():
                break
            if candidate in used:
                continue
            test = perturbed + [candidate]
            if check_validity(test):
                perturbed.append(candidate)
                used.add(candidate)
        
        # Apply swap search
        perturbed = swap_search(perturbed, max_iterations=10000)
        
        if len(perturbed) > best_n:
            best = perturbed
            best_n = len(perturbed)
    
    return best

def construct():
    random.seed(42)
    
    # Start with greedy
    elements = greedy_construct()
    
    if is_done():
        return [int(x) for x in elements]
    
    # Apply swap search
    elements = swap_search(elements, max_iterations=50000)
    
    if is_done():
        return [int(x) for x in elements]
    
    # Try randomized restarts with remaining time
    remaining = DEADLINE - elapsed()
    if remaining > 5.0:
        num_restarts = min(10, int(remaining / 2))
        elements = randomized_restart(elements, num_restarts)
    
    # Final check
    if not check_validity(elements):
        # Fallback to a known valid small set
        elements = list(range(128, 136))
    
    return [int(x) for x in elements]
# EVOLVE-BLOCK-END
