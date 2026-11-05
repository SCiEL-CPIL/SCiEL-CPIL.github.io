import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 58.0
    
    # Precompute reciprocals
    recip = [0.0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i

    # Precompute conflict bitmasks: conflict[x] is a bitmask where bit y is set
    # if x and y cannot both be in a 3-AP-free set together
    # i.e., there exists z such that {x,y,z} forms a 3-AP
    # We use Python integers as arbitrary-precision bitmasks
    # Bit position (i-1) represents element i
    
    conflict = [0] * (n + 1)
    for x in range(1, n + 1):
        mask = 0
        for y in range(1, n + 1):
            if x == y:
                continue
            # Check if x and y can form a 3-AP with some z
            # Case 1: x, y, z form AP: 2y = x + z -> z = 2y - x
            z1 = 2 * y - x
            if z1 != x and z1 != y and 1 <= z1 <= n:
                mask |= (1 << (y - 1))
                continue
            # Case 2: y, x, z form AP: 2x = y + z -> z = 2x - y
            z2 = 2 * x - y
            if z2 != x and z2 != y and 1 <= z2 <= n:
                mask |= (1 << (y - 1))
                continue
            # Case 3: x, z, y form AP: 2z = x + y -> z = (x+y)/2
            if (x + y) % 2 == 0:
                z3 = (x + y) // 2
                if z3 != x and z3 != y and 1 <= z3 <= n:
                    mask |= (1 << (y - 1))
        
        conflict[x] = mask
    
    # Fast feasibility check using bitmask
    # s_mask is a Python int where bit (i-1) is set if element i is in the set
    def can_add_fast(x, s_mask):
        # Check if adding x creates a conflict
        # x conflicts with any y in s_mask if bit (y-1) is set in conflict[x]
        # So we check: (s_mask & conflict[x]) != 0
        return (s_mask & conflict[x]) == 0
    
    def get_sum_from_mask(s_mask):
        total = 0.0
        for i in range(1, n + 1):
            if (s_mask >> (i - 1)) & 1:
                total += recip[i]
        return total
    
    def greedy_fill_from_mask(s_mask):
        # Add elements in order of decreasing reciprocal (increasing value)
        # i.e., from smallest to largest index
        for x in range(1, n + 1):
            if not ((s_mask >> (x - 1)) & 1):
                if can_add_fast(x, s_mask):
                    s_mask |= (1 << (x - 1))
        return s_mask
    
    def local_search_2opt(s_mask):
        """2-opt local search: try removing one and adding one, or removing two and adding two"""
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_sum = get_sum_from_mask(s_mask)
            
            # Try 1-remove, 1-add moves
            for rem in range(1, n + 1):
                if not ((s_mask >> (rem - 1)) & 1):
                    continue
                # Remove rem
                trial_mask = s_mask & ~(1 << (rem - 1))
                
                # Try adding each non-member
                for add in range(1, n + 1):
                    if ((trial_mask >> (add - 1)) & 1):
                        continue
                    if can_add_fast(add, trial_mask):
                        new_mask = trial_mask | (1 << (add - 1))
                        new_sum = current_sum - recip[rem] + recip[add]
                        if new_sum > current_sum + 1e-12:
                            s_mask = new_mask
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
            
            # Try 2-remove, 2-add moves (more powerful)
            if not improved and time.time() < deadline - 2.0:
                for rem1 in range(1, n + 1):
                    if not ((s_mask >> (rem1 - 1)) & 1):
                        continue
                    for rem2 in range(rem1 + 1, n + 1):
                        if not ((s_mask >> (rem2 - 1)) & 1):
                            continue
                        trial_mask = s_mask & ~(1 << (rem1 - 1)) & ~(1 << (rem2 - 1))
                        
                        # Try adding pairs
                        for add1 in range(1, n + 1):
                            if ((trial_mask >> (add1 - 1)) & 1):
                                continue
                            if not can_add_fast(add1, trial_mask):
                                continue
                            for add2 in range(add1 + 1, n + 1):
                                if ((trial_mask >> (add2 - 1)) & 1):
                                    continue
                                if not can_add_fast(add2, trial_mask | (1 << (add1 - 1))):
                                    continue
                                new_mask = trial_mask | (1 << (add1 - 1)) | (1 << (add2 - 1))
                                new_sum = current_sum - recip[rem1] - recip[rem2] + recip[add1] + recip[add2]
                                if new_sum > current_sum + 1e-12:
                                    s_mask = new_mask
                                    current_sum = new_sum
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return s_mask
    
    # 1. Initial Greedy
    initial_mask = 0
    initial_mask = greedy_fill_from_mask(initial_mask)
    best_mask = initial_mask
    best_sum = get_sum_from_mask(best_mask)
    
    # 2. Local Search on Initial
    improved_mask = local_search_2opt(best_mask)
    if get_sum_from_mask(improved_mask) > best_sum:
        best_mask = improved_mask
        best_sum = get_sum_from_mask(best_mask)
    
    # 3. Randomized Restart / Perturbation
    random.seed(42)
    
    while time.time() < deadline - 3.0:
        # Perturb best_mask
        perturbed = best_mask
        
        # Remove 1 to 3 random elements
        num_removals = random.randint(1, 3)
        elements_in_set = [i for i in range(1, n + 1) if (perturbed >> (i - 1)) & 1]
        if not elements_in_set:
            break
            
        for _ in range(num_removals):
            if not elements_in_set:
                break
            idx = random.randint(0, len(elements_in_set) - 1)
            rem_val = elements_in_set[idx]
            perturbed &= ~(1 << (rem_val - 1))
            elements_in_set.pop(idx)
        
        # Refill
        perturbed = greedy_fill_from_mask(perturbed)
        
        # Local Search
        improved_mask = local_search_2opt(perturbed)
        
        new_sum = get_sum_from_mask(improved_mask)
        if new_sum > best_sum + 1e-12:
            best_sum = new_sum
            best_mask = improved_mask

    # Extract result
    result = [i for i in range(1, n + 1) if (best_mask >> (i - 1)) & 1]
    
    # Validation
    s_set = set(result)
    for i, a in enumerate(result):
        for b in result[i+1:]:
            if (a + b) % 2 == 0:
                mid = (a + b) // 2
                if mid != a and mid != b and mid in s_set:
                    raise AssertionError("3-AP found")
                    
    return result
