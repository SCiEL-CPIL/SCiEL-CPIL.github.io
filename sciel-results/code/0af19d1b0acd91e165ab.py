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
    
    elements = list(range(1, 23))
    elements_by_val_desc = sorted(elements, key=lambda a: vals[a], reverse=True)
    
    def check_add(current_sums, v):
        """Check if adding value v creates any duplicate subset sums."""
        new_sums = set()
        for s in current_sums:
            ns = s + v
            if ns in current_sums or ns in new_sums:
                return False, None
            new_sums.add(ns)
        return True, new_sums
    
    def get_sums(subset):
        """Compute all subset sums for a given subset."""
        sums = {0}
        for a in subset:
            v = vals[a]
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums = sums | new_sums
        return sums
    
    def greedy_build(order):
        """Greedy build with a given element order."""
        current = []
        current_sums = {0}
        for a in order:
            v = vals[a]
            ok, new_sums = check_add(current_sums, v)
            if ok:
                current.append(a)
                current_sums = current_sums | new_sums
        return current
    
    best_subset = []
    
    # Strategy 1: Try multiple greedy orderings
    orderings = [
        elements_by_val_desc,
        sorted(elements, key=lambda a: vals[a]),
        elements,
        sorted(elements, key=lambda a: (a % 5, a)),
    ]
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        result = greedy_build(order)
        if len(result) > len(best_subset):
            best_subset = result
    
    greedy_size = len(best_subset)
    
    # Strategy 2: Backtracking with incremental sums
    def backtrack(start_idx, current_subset, current_sums, target_size, order):
        """Try to build a dissociated set of target_size using incremental subset sums."""
        if time.time() > deadline - 3:
            return None
        
        if len(current_subset) == target_size:
            return list(current_subset)
        
        remaining_needed = target_size - len(current_subset)
        available = len(order) - start_idx
        
        if available < remaining_needed:
            return None
        
        for i in range(start_idx, len(order)):
            if time.time() > deadline - 3:
                return None
            
            a = order[i]
            v = vals[a]
            ok, new_sums = check_add(current_sums, v)
            
            if ok:
                merged_sums = current_sums | new_sums
                result = backtrack(i + 1, current_subset + [a], merged_sums, target_size, order)
                if result is not None:
                    return result
        
        return None
    
    # Try to find larger sets using backtracking
    for target_size in range(22, greedy_size, -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        for order in [elements_by_val_desc, sorted(elements, key=lambda a: vals[a])]:
            if time.time() > deadline - 3:
                break
            result = backtrack(0, [], {0}, target_size, order)
            if result is not None:
                best_subset = result
                found = True
                break
        
        if found:
            break
    
    # Strategy 3: Local search with swaps (remove 1, add up to 2)
    if time.time() < deadline - 10 and len(best_subset) < 22:
        best = list(best_subset)
        improved = True
        while improved:
            improved = False
            # First try adding any missing element
            best_sums = get_sums(best)
            for candidate in range(1, 23):
                if time.time() > deadline - 3:
                    break
                if candidate in best:
                    continue
                v = vals[candidate]
                ok, _ = check_add(best_sums, v)
                if ok:
                    best.append(candidate)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one element, see if we can add two
            for i in range(len(best)):
                if time.time() > deadline - 3:
                    break
                removed = best[i]
                remaining = best[:i] + best[i+1:]
                rem_sums = get_sums(remaining)
                
                added = []
                for j in range(1, 23):
                    if time.time() > deadline - 3:
                        break
                    if j == removed or j in remaining:
                        continue
                    v = vals[j]
                    ok, new_sums = check_add(rem_sums, v)
                    if ok:
                        added.append(j)
                        rem_sums = rem_sums | new_sums
                    if len(added) >= 2:
                        break
                
                if len(added) >= 2:
                    best = remaining + added
                    improved = True
                    break
        
        if len(best) > len(best_subset):
            best_subset = best
    
    # Strategy 4: Random restarts
    if time.time() < deadline - 10 and len(best_subset) < 22:
        import random
        random.seed(42)
        
        for trial in range(500):
            if time.time() > deadline - 3:
                break
            
            shuffled = elements_by_val_desc[:]
            random.shuffle(shuffled)
            result = greedy_build(shuffled)
            if len(result) > len(best_subset):
                best_subset = result
    
    # Ensure we have at least one element
    if not best_subset:
        best_subset = [1]
    
    return best_subset

# EVOLVE-BLOCK-END
