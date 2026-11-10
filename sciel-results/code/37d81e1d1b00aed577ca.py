"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import lcm
    from fractions import Fraction
    
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin before 90s limit
    
    # Precompute LCM matrix and conflict bitmasks
    n = 256
    # conflict_mask[x] is a bitmask where bit y is set if lcm(x,y) <= 256
    # We use 0-based indexing for numbers 2..256, so index i corresponds to number i+2
    conflict_mask = [0] * (n - 1)  # For numbers 2..256
    numbers = list(range(2, 257))
    
    for i, x in enumerate(numbers):
        mask = 0
        for j, y in enumerate(numbers):
            if i != j and lcm(x, y) <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Initialize with greedy construction ordered by 1/x (largest reciprocal first)
    # But we want to maximize sum, so we prefer small numbers that don't conflict
    # Let's start with an empty set and add elements in order of decreasing 1/x (i.e., increasing x)
    # Actually, to maximize reciprocal sum, we want small numbers. But small numbers conflict more.
    # Greedy: sort by 1/x descending (x ascending), add if no conflict.
    
    selected_indices = []
    selected_set = set()
    
    for i in range(len(numbers)):
        x = numbers[i]
        # Check if x conflicts with any in selected_set
        can_add = True
        for j in selected_indices:
            if (conflict_mask[i] >> j) & 1:
                can_add = False
                break
        if can_add:
            selected_indices.append(i)
            selected_set.add(i)
    
    def current_score(indices):
        if not indices:
            return Fraction(0, 1)
        total = Fraction(0, 1)
        for i in indices:
            total += Fraction(1, numbers[i])
        return total
    
    best_indices = selected_indices[:]
    best_score = current_score(best_indices)
    
    def is_valid_add(i, current_set):
        """Check if index i can be added to current_set"""
        for j in current_set:
            if (conflict_mask[i] >> j) & 1:
                return False
        return True
    
    def local_search(indices, time_limit):
        nonlocal best_indices, best_score
        
        indices = indices[:]
        current_score_val = current_score(indices)
        
        # Precompute set for fast lookup
        idx_set = set(indices)
        
        no_improvement = 0
        max_no_improvement = 5000
        
        while time.time() < time_limit:
            improved = False
            
            # Try adding elements
            for i in range(len(numbers)):
                if i in idx_set:
                    continue
                if is_valid_add(i, idx_set):
                    new_indices = indices + [i]
                    new_score = current_score_val + Fraction(1, numbers[i])
                    if new_score > current_score_val:
                        indices = new_indices
                        idx_set.add(i)
                        current_score_val = new_score
                        improved = True
                        no_improvement = 0
                        if current_score_val > best_score:
                            best_score = current_score_val
                            best_indices = indices[:]
                        break  # Restart loop after improvement
            
            if improved:
                continue
            
            # Try 1-swap: remove one, add another
            if no_improvement > 100:
                # Try 2-swap or just break if stalled
                pass
                
            # Try removing an element and adding a better one
            for i in range(len(indices)):
                remove_idx = indices[i]
                remaining = [x for x in indices if x != remove_idx]
                rem_set = idx_set - {remove_idx}
                
                # Try adding each element not in set
                for j in range(len(numbers)):
                    if j in rem_set:
                        continue
                    if is_valid_add(j, rem_set):
                        # Check if j conflicts with remove_idx? No, we're swapping.
                        # But j must not conflict with any in rem_set.
                        # Also, we need to check if removing remove_idx allows j.
                        # The new set is rem_set + {j}
                        new_indices = remaining + [j]
                        new_score = current_score_val - Fraction(1, numbers[remove_idx]) + Fraction(1, numbers[j])
                        if new_score > current_score_val:
                            indices = new_indices
                            idx_set = rem_set | {j}
                            current_score_val = new_score
                            improved = True
                            no_improvement = 0
                            if current_score_val > best_score:
                                best_score = current_score_val
                                best_indices = indices[:]
                            break
                if improved:
                    break
            
            if not improved:
                no_improvement += 1
                if no_improvement > max_no_improvement:
                    break
        
        return best_indices, best_score
    
    # Run local search
    best_indices, best_score = local_search(best_indices, deadline)
    
    # Convert back to numbers
    result = [numbers[i] for i in best_indices]
    
    return result
# EVOLVE-BLOCK-END
