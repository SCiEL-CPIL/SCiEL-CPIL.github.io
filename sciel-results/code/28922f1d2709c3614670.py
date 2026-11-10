"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import random

def construct():
    """
    Find a nonempty subset of [1, 22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum = 0) are distinct.
    
    Uses optimized greedy construction with multiple random orderings.
    """
    candidates = list(range(1, 23))  # [1, 2, ..., 22]
    best_set = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Generate a variety of orderings
    # 1. Descending (large elements first)
    # 2. Ascending
    # 3. Random shuffles
    # 4. "Weighted" shuffles favoring larger elements
    
    orderings = [sorted(candidates, reverse=True), sorted(candidates)]
    
    # Add 200 random orderings
    for _ in range(200):
        order = candidates[:]
        rng.shuffle(order)
        orderings.append(order)
        
    # Add 50 orderings that prioritize larger numbers
    for _ in range(50):
        order = candidates[:]
        # Sort by 1/x (descending) with some noise
        # Or just pick the largest available with some probability
        remaining = candidates[:]
        constructed_order = []
        while remaining:
            # Pick a random element, but bias towards larger indices
            # Simple: shuffle, then move larger elements to front with prob 0.5?
            # Let's just use a random shuffle but sort by value descending with 50% chance
            if rng.random() < 0.5:
                remaining.sort(reverse=True)
            else:
                rng.shuffle(remaining)
            # Pick the first one
            constructed_order.append(remaining.pop(0))
        orderings.append(constructed_order)

    for order in orderings:
        current = []
        current_sums = {Fraction(0, 1)}
        
        for candidate in order:
            r = Fraction(1, candidate)
            # Check if adding candidate preserves distinctness
            # Collision if (s + r) in current_sums for any s in current_sums
            valid = True
            temp_new_sums = set()
            
            # Optimization: iterate and check
            for existing in current_sums:
                val = existing + r
                if val in current_sums:
                    valid = False
                    break
                temp_new_sums.add(val)
            
            if valid:
                current.append(candidate)
                # Update current_sums
                # current_sums = current_sums.union(temp_new_sums)
                # In-place update is not possible for set in this context easily without creating new set
                # But we can just do:
                current_sums = current_sums | temp_new_sums
                
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
    # If no set was found (unlikely), return a valid single element
    if not best_set:
        return [1]
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
