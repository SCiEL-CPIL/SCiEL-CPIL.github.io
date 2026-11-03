import math
import random

def construct():
    # Fast score calculation using bitsets
    def calc_score(u):
        if not u:
            return 0
        max_u = u[-1]
        if max_u == 0:
            return 0
            
        sum_mask = 0
        diff_mask = 0
        
        # Use integers as bitsets
        for i in u:
            for j in u:
                sum_mask |= (1 << (i + j))
                diff_mask |= (1 << (i - j + max_u))
                
        num_sums = bin(sum_mask).count('1')
        num_diffs = bin(diff_mask).count('1')
        
        if num_sums == 0:
            return 0
            
        ratio = num_diffs / num_sums
        if ratio <= 1:
            return 0
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 0
            
        return 1 + math.log2(ratio) / denom

    # Candidate sets
    # Known good structures:
    # 1. Powers of 2 minus 1
    # 2. Small sets with high diff/sum ratio
    
    candidates = []
    
    # Generate powers of 2 minus 1 sets
    for k in range(2, 10):
        u = [2**i - 1 for i in range(k)]
        if u[0] != 0:
             u[0] = 0 # Ensure 0 is in the set, though 2^0-1=0
        candidates.append(sorted(set(u)))
        
    # Generate some other sparse sets
    # e.g. {0, 1, 2, 4, 8}
    candidates.append([0, 1, 2, 4, 8, 16])
    candidates.append([0, 1, 2, 5, 10, 20])
    candidates.append([0, 1, 3, 4, 8, 16])
    
    # Add the specific set from the previous analysis
    candidates.append([0, 1, 3, 7, 15])
    candidates.append([0, 1, 3, 7, 15, 31])
    candidates.append([0, 1, 3, 7, 15, 31, 63])
    
    best_score = 0
    best_set = [0]
    
    for u in candidates:
        s = calc_score(u)
        if s > best_score:
            best_score = s
            best_set = u

    # Local search / Hill Climbing
    # We try to improve the best set found so far
    # We limit the search to keep it within time limits
    max_iters = 5000
    current_set = best_set[:]
    current_score = best_score
    
    for _ in range(max_iters):
        improved = False
        # Try adding a new element
        max_u = current_set[-1]
        # Try elements up to 2 * max_u + 10
        upper_bound = min(2 * max_u + 10, 200) 
        # To avoid too many iterations, we sample or use a small range
        # We focus on small sets first
        
        # Strategy: 
        # 1. Try adding small numbers
        # 2. Try replacing the largest number with a slightly larger one
        
        # Add
        for new_elem in range(1, upper_bound + 1):
            if new_elem in current_set:
                continue
            new_set = sorted(current_set + [new_elem])
            s = calc_score(new_set)
            if s > current_score + 1e-6:
                current_set = new_set
                current_score = s
                best_set = new_set
                best_score = s
                improved = True
                break
        
        if improved:
            continue
            
        # Remove
        if len(current_set) > 2:
            for i in range(len(current_set)):
                if i == 0: # Keep 0
                    continue
                new_set = current_set[:i] + current_set[i+1:]
                s = calc_score(new_set)
                if s > current_score + 1e-6:
                    current_set = new_set
                    current_score = s
                    best_set = new_set
                    best_score = s
                    improved = True
                    break
            
        if improved:
            continue
        
        # Swap
        # Replace the largest element with another
        max_elem = current_set[-1]
        for new_max in range(max_elem - 5, max_elem + 10):
            if new_max in current_set:
                continue
            if new_max <= 0:
                continue
            new_set = current_set[:-1] + [new_max]
            new_set = sorted(set(new_set))
            s = calc_score(new_set)
            if s > current_score + 1e-6:
                current_set = new_set
                current_score = s
                best_set = new_set
                best_score = s
                improved = True
                break
                
        if not improved:
            # If no improvement, try a random restart or break
            # To ensure we finish, we break if stuck for a while
            # For simplicity, we just break if no improvement in this iteration
            # But we might miss better local maxima. 
            # Given the small search space of "good" sets, this is likely sufficient.
            break

    return best_set
