import time
from math import gcd, lcm
import sys
import functools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 256
    values = list(range(2, N + 1))
    n = len(values)
    
    # 1. Precompute Conflict Graph
    # conflict[i][j] is True if lcm(values[i], values[j]) <= 256
    # We want an independent set in this graph.
    # Note: values[i] = i + 2
    # Let's use 0-indexed for internal logic, mapping to 2..256 at the end.
    
    # Optimization: Precompute LCMs or just check on the fly? 
    # Precomputing a full 255x255 matrix is fine (65k entries).
    conflict = [[False] * n for _ in range(n)]
    neighbors = [set() for _ in range(n)]
    
    for i in range(n):
        vi = values[i]
        for j in range(i + 1, n):
            vj = values[j]
            # lcm(a,b) = a*b // gcd(a,b)
            # If lcm <= 256, they conflict.
            g = gcd(vi, vj)
            l = (vi * vj) // g
            if l <= 256:
                conflict[i][j] = True
                conflict[j][i] = True
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    weights = [1.0 / values[i] for i in range(n)]
    
    # Sort vertices by weight descending (i.e., value ascending)
    # This helps the greedy bound and branch-and-bound heuristic
    # We will reorder the vertices so that higher weights come first.
    # Let's create a permutation of indices sorted by weight desc (value asc)
    order = sorted(range(n), key=lambda x: -weights[x])
    
    # Re-map weights and neighbors to the new order?
    # Or just work with the original indices but iterate in 'order'.
    # Branch and bound is easier if we fix the order of decision.
    # Let's create arrays indexed by "decision position"
    # pos 0 is the highest weight vertex, pos 1 is next, etc.
    
    # perm[i] is the original index of the i-th vertex in decision order
    perm = order
    # new_weights[i] is weight of perm[i]
    new_weights = [weights[perm[i]] for i in range(n)]
    
    # new_neighbors[i] is the set of decision-positions j such that perm[i] and perm[j] conflict
    # We only care about j > i for forward checking? 
    # Actually, for independent set, if we pick i, we must exclude all neighbors.
    # Let's precompute for each position i, the set of positions j > i that conflict with i.
    # This allows us to quickly "mask out" future candidates.
    
    # Map original index to decision position
    pos_of_orig = [0] * n
    for i, orig_idx in enumerate(perm):
        pos_of_orig[orig_idx] = i
        
    # conflict_pos[i] = set of j > i such that perm[i] and perm[j] conflict
    conflict_pos = [set() for _ in range(n)]
    for i in range(n):
        orig_i = perm[i]
        for orig_j in neighbors[orig_i]:
            pos_j = pos_of_orig[orig_j]
            if pos_j > i:
                conflict_pos[i].add(pos_j)
                
    # 2. Branch and Bound
    # State: 
    # - current_pos: next vertex to consider (index in perm)
    # - selected: list of selected vertices (decision positions)
    # - current_score: sum of weights of selected
    # - excluded: set of vertices excluded due to conflicts with selected
    #   Actually, it's better to maintain a "available" set or bitmask.
    #   With n=255, a Python set or list of booleans is okay, but bitmask is faster if we use integers.
    #   255 bits fits in 4 Python integers (or one big int). Python big ints are efficient.
    
    # Let's use a bitmask for available vertices.
    # Bit i is 1 if vertex at decision position i is available.
    # Initially all 1s.
    full_mask = (1 << n) - 1
    
    best_score = 0.0
    best_selection = []
    
    # Greedy upper bound function
    # Given a mask of available vertices and current position,
    # estimate the max additional weight we can get.
    # Simple bound: sum of weights of all available vertices. (Too loose)
    # Better bound: Greedy independent set on the available subgraph.
    # Even better: Sum of weights of vertices that are not adjacent to any other available vertex 
    # plus a fraction of the rest? 
    # For now, a simple greedy bound is often sufficient for B&B on sparse graphs.
    # But our graph might be dense for small numbers.
    
    def greedy_bound(mask):
        # Greedy pick: pick highest weight available, remove neighbors, repeat.
        # This gives a feasible solution, which is a lower bound on the max, 
        # but we need an UPPER bound for pruning.
        # Wait, B&B prunes if current + UB < best.
        # So we need an Upper Bound.
        # A common UB for MWIS is the sum of weights of all available vertices.
        # This is very loose.
        # A tighter UB: For each connected component, the max weight independent set is at most 
        # the sum of weights of vertices in that component.
        # Actually, let's use the "Fractional Knapsack" style or just the sum of weights.
        # Given the time limit, let's try a simple recursive B&B with the sum-of-weights UB first.
        # If it's too slow, we'll need a tighter UB.
        
        # A decent UB: Sum of weights of available vertices.
        # We can precompute prefix sums of weights in decision order?
        # No, mask is arbitrary.
        # Let's just iterate through set bits.
        total = 0.0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            total += new_weights[idx]
            m ^= lsb
        return total

    # This UB is likely too loose. 
    # Alternative: Since weights are 1/v, and conflicts are based on LCM,
    # maybe we can use a property.
    # Let's stick to a standard B&B with a slightly tighter heuristic:
    # UB = sum(weights of available) is O(N) per node.
    # With 255 nodes, depth 255, this is expensive if the tree is wide.
    
    # Let's try a different strategy:
    # Since the graph structure is fixed, we can use a "Maximum Weight Independent Set" 
    # algorithm specifically designed for such graphs or use a library? No, std lib only.
    
    # Let's refine the B&B:
    # We process vertices in order of decreasing weight.
    # At each step, we either include the current vertex or exclude it.
    # If we include it, we remove all its neighbors from the available set.
    # UB: Sum of weights of all available vertices.
    
    # To make this faster, we can use a "branch on the most constrained vertex" or 
    # stick to the weight order which is good for finding high score solutions early.
    
    # Let's implement the B&B.
    
    # Optimization: If the remaining potential (UB) + current_score <= best_score, prune.
    
    # We need to be careful with recursion depth (255 is fine).
    
    def solve(pos, mask, current_score, current_sel):
        nonlocal best_score, best_selection
        
        if time.time() > deadline:
            return
            
        # Pruning: if even taking all remaining available vertices can't beat best
        # Calculate UB
        # Instead of iterating all bits every time, we can maintain a running sum?
        # Hard to maintain with mask updates.
        # Let's just calculate it. 255 iterations is fast.
        ub = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            ub += new_weights[idx]
            m ^= lsb
            
        if current_score + ub <= best_score:
            return
            
        # Base case: no more vertices to consider
        if mask == 0:
            if current_score > best_score:
                best_score = current_score
                best_selection = list(current_sel)
            return
            
        # Find the next available vertex in decision order
        # We want to process in order of pos? 
        # The mask tells us who is available. 
        # We should pick the lowest index in the mask that corresponds to our decision order?
        # No, the mask bits correspond to decision positions 0..n-1.
        # We should pick the vertex with the smallest decision position that is available?
        # Yes, because we defined the order.
        
        # Find lowest set bit in mask
        # This corresponds to the earliest available vertex in our weight-sorted order.
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        
        # Option 1: Exclude v
        # Just clear bit v
        mask_without_v = mask ^ lsb
        solve(v + 1, mask_without_v, current_score, current_sel)
        
        # Option 2: Include v
        # Remove v and all its neighbors (that are also in the mask)
        mask_with_v = mask_without_v
        # Remove neighbors of v
        # conflict_pos[v] contains indices j > v that conflict with v.
        # Since we are processing in order, we only need to remove those > v.
        # Wait, conflict_pos[v] is precomputed as j > v.
        # So we just need to clear those bits in mask_with_v.
        for nb in conflict_pos[v]:
            bit = 1 << nb
            if mask_with_v & bit:
                mask_with_v ^= bit
                
        current_sel.append(v)
        solve(v + 1, mask_with_v, current_score + new_weights[v], current_sel)
        current_sel.pop()

    # Initial call
    # We can get a good initial best_score using a greedy heuristic
    # Greedy: pick highest weight, remove neighbors, repeat.
    def greedy_initial():
        mask = full_mask
        sel = []
        score = 0.0
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            sel.append(v)
            score += new_weights[v]
            mask ^= lsb
            for nb in conflict_pos[v]:
                bit = 1 << nb
                if mask & bit:
                    mask ^= bit
        return score, sel

    init_score, init_sel = greedy_initial()
    best_score = init_score
    best_selection = init_sel
    
    # Run B&B
    # To avoid deep recursion overhead or stack overflow (unlikely for 255), 
    # we use recursion.
    
    # One issue: The UB is very loose. 
    # Let's add a tighter UB: 
    # The sum of weights of vertices in the mask is an upper bound.
    # Is there a better one?
    # For independent sets, the sum of weights is the best simple UB.
    # However, we can prune more aggressively if we detect that the remaining graph is empty.
    
    # Let's run it.
    solve(0, full_mask, 0.0, [])
    
    # Map back to original values
    # best_selection contains decision positions.
    # We need the original values.
    result_values = []
    for pos in best_selection:
        orig_idx = perm[pos]
        result_values.append(values[orig_idx])
        
    # Verify validity just in case (should be valid by construction)
    # But let's double check the time.
    if time.time() > deadline:
        # Return the greedy result if B&B didn't finish? 
        # The nonlocal best_score is updated only on valid full paths.
        # So best_selection is always valid.
        pass
        
    return result_values

# Test locally? No, just return the function.
# The function is construct.

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
