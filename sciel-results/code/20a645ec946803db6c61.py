"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

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
    
    # Sort elements by their value in descending order
    elements = sorted(range(1, 23), key=lambda a: vals[a], reverse=True)
    
    def is_dissociated(subset):
        """Check if all subset sums of the given subset (using integer values) are distinct."""
        if not subset:
            return True
        k = len(subset)
        
        # Generate all subset sums
        sums = {0}
        for a in subset:
            v = vals[a]
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True
    
    best_subset = []
    
    # Strategy 1: Multiple greedy passes with different orderings
    import random
    rng = random.Random(42)
    
    # Try several different orderings for greedy
    orderings = [
        elements,  # descending by value
        elements[::-1],  # ascending by value
    ]
    
    # Add some random orderings
    for _ in range(10):
        shuffled = elements[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    for ordering in orderings:
        if time.time() > deadline - 5:
            break
        current = []
        current_sums = {0}
        for a in ordering:
            v = vals[a]
            new_sums = set()
            conflict = False
            for s in current_sums:
                ns = s + v
                if ns in current_sums or ns in new_sums:
                    conflict = True
                    break
                new_sums.add(ns)
            if not conflict:
                current.append(a)
                current_sums = current_sums | new_sums
        if len(current) > len(best_subset):
            best_subset = list(current)
    
    greedy_size = len(best_subset)
    
    # Strategy 2: Backtracking with better pruning
    def backtrack_with_sums(start_idx, current_subset, current_sums, target_size):
        """Try to build a dissociated set of target_size using incremental subset sums."""
        if time.time() > deadline - 3:
            return None, None
        
        if len(current_subset) == target_size:
            return list(current_subset), None
        
        remaining_needed = target_size - len(current_subset)
        available = 22 - start_idx
        
        if available < remaining_needed:
            return None, None
        
        # Pruning: if we can't possibly reach target_size, return early
        # Also, if we have enough elements and they're all valid, we're done
        
        for i in range(start_idx, 22):
            if time.time() > deadline - 3:
                return None, None
            
            a = elements[i]
            v = vals[a]
            
            # Check if adding v creates any duplicate subset sums
            new_sums = set()
            conflict = False
            for s in current_sums:
                ns = s + v
                if ns in current_sums or ns in new_sums:
                    conflict = True
                    break
                new_sums.add(ns)
            
            if not conflict:
                merged_sums = current_sums | new_sums
                result, _ = backtrack_with_sums(i + 1, current_subset + [a], merged_sums, target_size)
                if result is not None:
                    return result, merged_sums
        
        return None, None
    
    # Try to find larger sets using backtracking with incremental sums
    # Start from a reasonable size to avoid excessive backtracking
    for target_size in range(22, max(greedy_size, 15), -1):
        if time.time() > deadline - 10:
            break
        
        result, _ = backtrack_with_sums(0, [], {0}, target_size)
        if result is not None:
            best_subset = result
            break
    
    # Strategy 3: Local search - try to improve by swapping elements
    if time.time() < deadline - 10 and len(best_subset) < 22:
        rng2 = random.Random(123)
        
        for trial in range(500):
            if time.time() > deadline - 3:
                break
            
            # Start from best_subset and try random swaps
            current = list(best_subset)
            current_sums = {0}
            for a in current:
                v = vals[a]
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s + v)
                current_sums = current_sums | new_sums
            
            # Try adding random elements not in current
            candidates = [a for a in range(1, 23) if a not in current]
            if not candidates:
                continue
            
            rng2.shuffle(candidates)
            improved = False
            for a in candidates:
                v = vals[a]
                new_sums = set()
                conflict = False
                for s in current_sums:
                    ns = s + v
                    if ns in current_sums or ns in new_sums:
                        conflict = True
                        break
                    new_sums.add(ns)
                if not conflict:
                    current.append(a)
                    current_sums = current_sums | new_sums
                    improved = True
            
            if improved and len(current) > len(best_subset):
                best_subset = list(current)
    
    # Ensure we have at least one element
    if not best_subset:
        best_subset = [1]
    
    return best_subset

# EVOLVE-BLOCK-END
