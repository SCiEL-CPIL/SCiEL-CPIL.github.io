"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # Numbers 2 to 256
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute weights (reciprocals as Fractions for exact comparison)
    weights = [Fraction(1, x) for x in nums]
    
    # Build conflict adjacency: conflict[i][j] = True if lcm(nums[i], nums[j]) <= 256
    # Use bitsets for efficiency
    # n = 255, so we can use Python integers as bitsets
    conflict = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nums[i], nums[j]
            g = gcd(a, b)
            lcm_val = a * b // g
            if lcm_val <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Sort by weight descending (larger reciprocal first) to get good initial solutions
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    # Remap: new_index -> old_index
    # We'll work in the sorted order for B&B
    # Let's create arrays in sorted order
    sorted_nums = [nums[i] for i in order]
    sorted_weights = [weights[i] for i in order]
    # Rebuild conflict in sorted order
    old_to_new = {}
    for new_idx, old_idx in enumerate(order):
        old_to_new[old_idx] = new_idx
    
    conflict_sorted = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if conflict[order[i]] & (1 << order[j]):
                conflict_sorted[i] |= (1 << j)
                conflict_sorted[j] |= (1 << i)
    
    # Branch and bound for Maximum Weight Independent Set
    # Using a recursive approach with coloring upper bound
    
    best_weight = Fraction(0, 1)
    best_set = set()
    
    # Greedy initial solution: pick highest weight nodes that don't conflict
    greedy_set = set()
    greedy_weight = Fraction(0, 1)
    for i in range(n):
        # Check if i conflicts with any in greedy_set
        can_add = True
        for j in greedy_set:
            if conflict_sorted[i] & (1 << j):
                can_add = False
                break
        if can_add:
            greedy_set.add(i)
            greedy_weight += sorted_weights[i]
    
    if greedy_weight > best_weight:
        best_weight = greedy_weight
        best_set = set(greedy_set)
    
    # B&B with upper bound using greedy coloring on the remaining independent set candidates
    # For MWIS, we use: upper_bound = sum of weights of nodes that can still be added
    # Better: use a fractional relaxation or a simple bound
    
    # Let's use a simple B&B: at each node, we decide to include or exclude the current node
    # Upper bound: current_weight + sum of weights of all remaining nodes (loose but fast)
    # Tighter: use the fact that in an independent set, we can't pick adjacent nodes
    
    # Implement a more sophisticated B&B using the "branch on a node" approach
    # with upper bound = current_weight + sum of weights of nodes in the candidate set
    # that are not adjacent to each other... this is still hard.
    
    # Use a simpler approach: branch and bound with the bound being
    # current_weight + sum of weights of all remaining candidates
    # This is loose but with good ordering it works
    
    # Actually, let's use a different strategy: since n=255 is moderate,
    # let's use a recursive B&B where at each step we pick a node and branch
    # on include/exclude, with the upper bound being the sum of weights
    # of all nodes in the current candidate set (which is an upper bound
    # on the additional weight we can get).
    
    # For better performance, use the following:
    # - Maintain a candidate set (nodes that can still be added)
    # - Upper bound = current_weight + sum(weights of candidate set)
    # - If upper bound <= best_weight, prune
    # - Branch on the highest-weight candidate node
    
    # This is essentially the same as the maximum clique B&B algorithm adapted for MWIS.
    
    # Let's implement it iteratively to avoid recursion depth issues
    # Use a stack-based approach
    
    # Actually, for n=255 with a loose bound, this might not be fast enough.
    # Let me try a different approach: use the structure of the problem.
    
    # Key observation: if a number x is in the set, then no multiple of x 
    # that is "too small" relative to x can be in the set.
    # Specifically, if x is in the set, then for any y in the set, 
    # lcm(x,y) > 256.
    
    # Let's try a smarter greedy + local search approach
    # 1. Start with a good greedy solution
    # 2. Try local search: swap operations
    
    # Local search: try to improve by adding, removing, or swapping
    current_set = set(greedy_set)
    current_weight = greedy_weight
    improved = True
    iterations = 0
    
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        if iterations > 10000:
            break
        
        # Try adding a node not in the set
        for i in range(n):
            if i in current_set:
                continue
            can_add = True
            for j in current_set:
                if conflict_sorted[i] & (1 << j):
                    can_add = False
                    break
            if can_add:
                new_set = current_set | {i}
                new_weight = current_weight + sorted_weights[i]
                if new_weight > best_weight:
                    best_weight = new_weight
                    best_set = set(new_set)
                current_set = new_set
                current_weight = new_weight
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one node, add another
            for i in list(current_set):
                for j in range(n):
                    if j in current_set:
                        continue
                    # Check if j can replace i
                    # j must not conflict with any node in current_set \ {i}
                    can_replace = True
                    for k in current_set:
                        if k == i:
                            continue
                        if conflict_sorted[j] & (1 << k):
                            can_replace = False
                            break
                    if can_replace:
                        new_weight = current_weight - sorted_weights[i] + sorted_weights[j]
                        if new_weight > current_weight:
                            current_set.remove(i)
                            current_set.add(j)
                            current_weight = new_weight
                            if current_weight > best_weight:
                                best_weight = current_weight
                                best_set = set(current_set)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Convert back to original numbers
    result = [sorted_nums[i] for i in best_set]
    
    # Verify the solution
    from math import lcm as lcm_func
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if lcm_func(result[i], result[j]) <= 256:
                # Invalid, fallback to greedy
                result = [sorted_nums[i] for i in greedy_set]
                break
    
    return sorted(result)

# EVOLVE-BLOCK-END
