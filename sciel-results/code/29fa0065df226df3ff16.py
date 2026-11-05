import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80.0  # 80s safety margin
    
    N = 256
    # Nodes are numbers 2..256, index 0..254
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflict: conflict[i][j] = True if lcm(nums[i], nums[j]) <= 256
    # For memory efficiency, use bitsets or just check on the fly with gcd
    # lcm(a,b) = a*b//gcd(a,b) <= 256 iff a*b <= 256*gcd(a,b)
    
    # Precompute gcd matrix or use math.gcd
    # Build adjacency as sets for fast lookup
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            a, b = nums[i], nums[j]
            g = math.gcd(a, b)
            if a * b <= 256 * g:  # lcm(a,b) <= 256
                adj[i].add(j)
                adj[j].add(i)
    
    # Weights: 1/x for each node
    weights = [1.0 / nums[i] for i in range(n)]
    
    # Branch and bound for maximum weight independent set
    # Order nodes by decreasing weight (smallest x first)
    order = sorted(range(n), key=lambda i: -weights[i])
    
    best_weight = 0.0
    best_set = set()
    
    # Use recursive branch and bound
    # State: current node index in order, current selected set, current weight
    # Upper bound: current_weight + sum of weights of all unprocessed nodes that are compatible with all selected nodes
    
    # For efficiency, track which nodes are "available" (not conflicting with any selected node)
    
    def branch_and_bound(pos, selected, current_weight, available_mask):
        nonlocal best_weight, best_set
        
        if time.time() > DEADLINE:
            return
        
        if pos == n:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(selected)
            return
        
        # Upper bound: current_weight + sum of weights of all available nodes from pos onward
        # Quick upper bound: current_weight + sum of weights[i] for i >= pos where i in available_mask
        ub = current_weight
        for k in range(pos, n):
            if k in available_mask:
                ub += weights[k]
        
        if ub <= best_weight:
            return
        
        idx = order[pos]
        
        # Option 1: Include idx (if available)
        if idx in available_mask:
            new_selected = selected + (idx,)
            new_available = set(available_mask)
            new_available.discard(idx)
            # Remove all neighbors of idx
            for nb in adj[idx]:
                if nb in new_available:
                    new_available.discard(nb)
            branch_and_bound(pos + 1, new_selected, current_weight + weights[idx], new_available)
        
        # Option 2: Exclude idx
        branch_and_bound(pos + 1, selected, current_weight, available_mask)
    
    # Initial available: all nodes
    initial_available = set(range(n))
    
    # To make this feasible, we need better pruning. Let's use a different approach.
    # The above is too slow for 255 nodes. Let me use a greedy + local improvement.
    
    # Better approach: Greedy construction + local search
    # Start with a greedy solution, then try to improve via swaps
    
    # Greedy: process nodes in order of decreasing weight, add if compatible
    def greedy_solve():
        selected = set()
        for i in order:
            if all(j not in selected for j in adj[i]):
                selected.add(i)
        return selected
    
    best_set = greedy_solve()
    best_weight = sum(weights[i] for i in best_set)
    
    # Local search: try to add nodes, remove nodes, swap
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        
        # Try adding each non-selected node
        for i in range(n):
            if i not in best_set:
                if all(j not in best_set for j in adj[i]):
                    new_weight = best_weight + weights[i]
                    if new_weight > best_weight:
                        best_set.add(i)
                        best_weight = new_weight
                        improved = True
        
        # Try removing a node and adding two or more nodes
        # This is more complex; try single removal + single addition (swap)
        for i in range(n):
            if i in best_set:
                # Try removing i and adding some j not in best_set
                for j in range(n):
                    if j not in best_set and j != i:
                        # Check if j is compatible with best_set - {i}
                        if all(k not in (best_set - {i}) for k in adj[j]):
                            new_weight = best_weight - weights[i] + weights[j]
                            if new_weight > best_weight:
                                best_set.discard(i)
                                best_set.add(j)
                                best_weight = new_weight
                                improved = True
        
        # Try removing a node and adding multiple nodes
        # This is expensive; limit to small sets
        if time.time() > DEADLINE - 10:
            break
    
    # Convert back to numbers
    result = [nums[i] for i in best_set]
    return result
