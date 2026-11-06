# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce
import random

def construct():
    """
    Find a nonempty subset of [1,22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum 0) are distinct.
    """
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Compute LCM of 1..22 to convert reciprocal sums to integer sums
    L = reduce(lcm, range(1, 23))
    
    # For each element a in [1,22], its contribution to the integer sum is L // a
    vals = {a: L // a for a in range(1, 23)}
    
    # Sort elements by their value (L//a) in descending order for greedy
    elements_desc = sorted(range(1, 23), key=lambda a: vals[a], reverse=True)
    elements_asc = sorted(range(1, 23), key=lambda a: vals[a])
    
    def is_dissociated(subset):
        """Check if all subset sums of the given subset (using integer values) are distinct."""
        if not subset:
            return True
        
        # Generate all subset sums
        sums = {0}
        for a in subset:
            v = vals[a]
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    best_subset = []
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    orderings = [
        elements_desc,
        elements_asc,
        list(range(1, 23)), # Natural order
        list(range(22, 0, -1)) # Reverse natural
    ]
    
    # Add some random orderings
    random.seed(42)
    for _ in range(20):
        shuffled = list(range(1, 23))
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    for ordering in orderings:
        if time.time() > deadline - 5:
            break
        current = []
        for a in ordering:
            test = current + [a]
            if is_dissociated(test):
                current = test
        if len(current) > best_size:
            best_size = len(current)
            best_subset = list(current)
    
    # Strategy 2: Local Search / Hill Climbing
    # Try to improve the best_subset by adding elements or swapping
    # We define a "neighbor" as a set obtained by adding one element (if valid)
    # or swapping one element with another, or removing one and adding two, etc.
    
    def try_improve(subset):
        """Try to find a larger dissociated set starting from subset."""
        current = list(subset)
        current_set = set(current)
        improved = False
        
        # Iterate until no improvement
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add any element not in current
            for a in range(1, 23):
                if a in current_set:
                    continue
                test = current + [a]
                if is_dissociated(test):
                    current = test
                    current_set.add(a)
                    improved = True
                    break # Restart loop after improvement
            
            # 2. If no direct addition, try swapping 1 element out, 1 in
            if not improved:
                for a_out in current:
                    for a_in in range(1, 23):
                        if a_in in current_set:
                            continue
                        # Remove a_out, add a_in
                        test = [x for x in current if x != a_out] + [a_in]
                        if is_dissociated(test):
                            current = test
                            current_set.remove(a_out)
                            current_set.add(a_in)
                            improved = True
                            break
                    if improved:
                        break
            
            # 3. If no 1-for-1 swap, try removing 1, adding 2 (net +1)
            if not improved:
                # This is more expensive, O(N^2 * check)
                # Only try if we have time
                if time.time() < deadline - 10:
                    for a_out in current:
                        for a_in1 in range(1, 23):
                            if a_in1 in current_set or a_in1 == a_out:
                                continue
                            for a_in2 in range(1, 23):
                                if a_in2 in current_set or a_in2 == a_out or a_in2 == a_in1:
                                    continue
                                test = [x for x in current if x != a_out] + [a_in1, a_in2]
                                if is_dissociated(test):
                                    current = test
                                    current_set.remove(a_out)
                                    current_set.add(a_in1)
                                    current_set.add(a_in2)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
    
        return current

    # Run local search from the best greedy result
    if best_subset:
        improved_subset = try_improve(best_subset)
        if len(improved_subset) > best_size:
            best_size = len(improved_subset)
            best_subset = improved_subset

    # Strategy 3: Randomized Local Search
    # Start from random subsets and try to improve
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        
        # Start with a random subset of size ~ best_size - 2
        start_size = max(1, best_size - 2)
        random_subset = random.sample(range(1, 23), start_size)
        
        # Ensure it's dissociated, otherwise reduce size or fix
        # Simple: just try to improve it. If it's not dissociated, try_improve might fail or return smaller.
        # Better: filter to dissociated first? No, just let try_improve handle it.
        # But try_improve assumes input is dissociated? No, it checks.
        # However, if input is NOT dissociated, "improving" is ill-defined.
        # Let's ensure start is dissociated.
        
        # Greedy build up to start_size?
        # Just use the greedy result as a base for random perturbations?
        
        # Let's try: Take best_subset, remove 2 random, add 2 random, then try_improve
        if len(best_subset) >= 4:
            to_remove = random.sample(best_subset, 2)
            to_add_candidates = [x for x in range(1, 23) if x not in best_subset]
            if len(to_add_candidates) >= 2:
                to_add = random.sample(to_add_candidates, 2)
                perturbed = [x for x in best_subset if x not in to_remove] + to_add
                improved = try_improve(perturbed)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_subset = improved

    # Ensure we have at least one element
    if not best_subset:
        best_subset = [1]
    
    return best_subset

# EVOLVE-BLOCK-END
