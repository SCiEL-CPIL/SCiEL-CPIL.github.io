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
    
    # Sort elements by their value (L//a) in descending order
    elements = sorted(range(1, 23), key=lambda a: vals[a], reverse=True)
    
    def check_dissociated_incremental(existing_sums, new_value):
        """
        Check if adding new_value to a set with existing subset sums keeps them distinct.
        existing_sums: set of all current subset sums
        Returns: (True, new_sums_set) if still dissociated, (False, None) otherwise
        """
        new_sums = set()
        for s in existing_sums:
            ns = s + new_value
            if ns in existing_sums or ns in new_sums:
                return (False, None)
            new_sums.add(ns)
        return (True, existing_sums | new_sums)
    
    def is_dissociated_full(subset):
        """Check if all subset sums of the given subset (using integer values) are distinct."""
        if not subset:
            return True
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
    best_sums = {0}
    
    # Strategy 1: Greedy - add elements in order of decreasing value
    current = []
    current_sums = {0}
    for a in elements:
        if time.time() > deadline - 5:
            break
        ok, new_sums = check_dissociated_incremental(current_sums, vals[a])
        if ok:
            current.append(a)
            current_sums = new_sums
            best_subset = list(current)
            best_sums = new_sums
    
    greedy_size = len(best_subset)
    
    # Strategy 2: Meet-in-the-middle style block combination
    # Partition elements into two halves, precompute valid subsets for each half
    # then combine them optimally
    
    def precompute_half_valid_subsets(half_elements, max_size):
        """
        Precompute all dissociated subsets of half_elements up to max_size.
        Returns a list of (subset, sums_set) pairs sorted by size descending.
        """
        if time.time() > deadline - 3:
            return []
        
        valid = []
        n = len(half_elements)
        
        # Use bit masks for subsets
        for mask in range(1 << n):
            if time.time() > deadline - 3:
                return valid
            
            # Count bits
            bit_count = bin(mask).count('1')
            if bit_count > max_size:
                continue
            
            subset = []
            for i in range(n):
                if mask & (1 << i):
                    subset.append(half_elements[i])
            
            if not subset:
                continue
            
            if is_dissociated_full(subset):
                # Compute sums
                sums = {0}
                for a in subset:
                    v = vals[a]
                    new_sums = set()
                    for s in sums:
                        new_sums.add(s + v)
                    sums.update(new_sums)
                valid.append((subset, sums))
        
        # Sort by size descending
        valid.sort(key=lambda x: len(x[0]), reverse=True)
        return valid
    
    # Split elements into two halves
    mid = len(elements) // 2
    first_half = elements[:mid]
    second_half = elements[mid:]
    
    # Precompute valid subsets for each half
    max_half_size = len(best_subset)
    if time.time() < deadline - 10:
        valid_first = precompute_half_valid_subsets(first_half, max_half_size)
        valid_second = precompute_half_valid_subsets(second_half, max_half_size)
        
        # Try to combine: pick one subset from each half and check if combined is dissociated
        # We want to maximize total cardinality
        best_combined_size = len(best_subset)
        
        for sub1, sums1 in valid_first:
            if time.time() > deadline - 5:
                break
            for sub2, sums2 in valid_second:
                if time.time() > deadline - 5:
                    break
                
                total_size = len(sub1) + len(sub2)
                if total_size <= best_combined_size:
                    continue
                
                # Check if combined set is dissociated
                # The combined sums are {s1 + s2 for s1 in sums1 for s2 in sums2}
                # But we need to check if any two different pairs give same sum
                # Actually, we need to check that the union set is dissociated
                # This is equivalent to: no s1a + s2a == s1b + s2b for distinct pairs
                
                # More efficiently: check if sums1 and sums2 are "shift-disjoint"
                # i.e., (sums1 - sums2) doesn't contain 0 except when s1=s1' and s2=s2'
                # Actually simpler: just build the combined sums set and check size
                
                if len(sub1) + len(sub2) > best_combined_size:
                    combined = list(sub1) + list(sub2)
                    if is_dissociated_full(combined):
                        # Compute combined sums
                        combined_sums = {0}
                        for a in combined:
                            v = vals[a]
                            new_sums = set()
                            for s in combined_sums:
                                new_sums.add(s + v)
                            combined_sums.update(new_sums)
                        best_subset = combined
                        best_sums = combined_sums
                        best_combined_size = total_size
        
        # After combination, try to add more elements from the remaining pool
        if best_combined_size > greedy_size:
            greedy_size = best_combined_size
    
    # Strategy 3: Backtracking with incremental sum tracking
    # Try to find a larger set by backtracking
    
    def backtrack(start_idx, current_subset, current_sums, target_size):
        """Try to build a dissociated set of target_size starting from start_idx."""
        if time.time() > deadline - 3:
            return None, None
        
        if len(current_subset) == target_size:
            return list(current_subset), set(current_sums)
        
        remaining_needed = target_size - len(current_subset)
        available = 22 - start_idx
        
        if available < remaining_needed:
            return None, None
        
        for i in range(start_idx, 22):
            if time.time() > deadline - 3:
                return None, None
            
            a = elements[i]
            v = vals[a]
            
            # Incremental check: can we add this element?
            ok, new_sums = check_dissociated_incremental(current_sums, v)
            if ok:
                current_subset.append(a)
                result, result_sums = backtrack(i + 1, current_subset, new_sums, target_size)
                current_subset.pop()
                if result is not None:
                    return result, result_sums
        
        return None, None
    
    # Try to find larger sets
    for target_size in range(22, greedy_size, -1):
        if time.time() > deadline - 5:
            break
        
        result, result_sums = backtrack(0, [], {0}, target_size)
        if result is not None:
            best_subset = result
            best_sums = result_sums
            break
    
    # Strategy 4: If we still have time, try random restarts with different orderings
    if time.time() < deadline - 10 and len(best_subset) < 22:
        import random
        random.seed(42)
        
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            
            # Shuffle elements
            shuffled = elements[:]
            random.shuffle(shuffled)
            
            current = []
            current_sums = {0}
            for a in shuffled:
                if time.time() > deadline - 3:
                    break
                ok, new_sums = check_dissociated_incremental(current_sums, vals[a])
                if ok:
                    current.append(a)
                    current_sums = new_sums
            
            if len(current) > len(best_subset):
                best_subset = current
                best_sums = current_sums
    
    # Ensure we have at least one element
    if not best_subset:
        best_subset = [1]
    
    return best_subset

# EVOLVE-BLOCK-END
