import math
import sys

# Increase recursion limit if necessary, though iterative or careful recursion is better
sys.setrecursionlimit(10000)

def construct():
    N = 511
    # Nodes are 1..511
    # Weights: 1 / (512 - a)
    weights = [0.0] * (N + 1)
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512 - a)
        
    # Build adjacency list (conflict graph)
    # adj[u] is a bitmask of nodes that conflict with u
    adj = [0] * (N + 1)
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if math.gcd(u, v) > 1:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    # Sort nodes by weight descending (largest a first)
    # We need to map sorted index to actual node value
    nodes = list(range(1, N + 1))
    nodes.sort(key=lambda x: weights[x], reverse=True)
    
    # Precompute weight of each node in the sorted order
    sorted_weights = [weights[n] for n in nodes]
    
    # Precompute neighbor masks in terms of the original node indices, 
    # but we will operate on the set of available original nodes.
    # It is more efficient to keep the mask of available original nodes.
    
    best_score = 0.0
    best_set = []
    
    # Helper to update best
    def update_best(score, mask):
        nonlocal best_score, best_set
        if score > best_score:
            best_score = score
            # Reconstruct set from mask
            s = []
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                idx = lsb.bit_length() - 1
                s.append(idx)
                m ^= lsb
            best_set = s
            
    # Initial mask: all nodes 1..511 are available
    # Bit i corresponds to node i. Note: bit 0 is unused or node 0?
    # Let's use bit i for node i. So bit 1 for node 1, ..., bit 511 for node 511.
    initial_mask = 0
    for i in range(1, N + 1):
        initial_mask |= (1 << i)
        
    # Precompute sum of weights for quick upper bound?
    # Sum of all weights is a constant.
    # Upper bound at any step = current_score + sum(weights of nodes in available_mask)
    
    # To speed up sum(weights of available_mask), we can precompute prefix sums?
    # No, the mask is arbitrary.
    # However, N=511 is small. Iterating through bits of the mask to sum weights is O(N).
    # In B&B, this is called many times.
    # We can optimize by passing the sum of available weights as a parameter, 
    # but updating it is tricky when removing neighbors.
    # Alternative: Just calculate sum(weights[i] for i in mask) each time.
    # 511 bits, so ~500 ops. If depth is ~50, and branching factor is small due to pruning, it should be fast.
    
    def get_sum_weights(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    # Branch and Bound
    # index: index in the sorted_nodes array
    # current_score: sum of weights of chosen nodes
    # available_mask: bitmask of nodes that can still be chosen
    
    def solve(idx, current_score, available_mask):
        nonlocal best_score, best_set
        
        if idx == N:
            if current_score > best_score:
                update_best(current_score, available_mask)
            return
        
        # Upper Bound: current_score + sum of weights of all available nodes
        # If this is <= best_score, prune
        # Optimization: If available_mask is 0, sum is 0
        if available_mask:
            ub = current_score + get_sum_weights(available_mask)
        else:
            ub = current_score
            
        if ub <= best_score:
            return
            
        u = nodes[idx]
        
        # Branch 1: Include u
        if (available_mask & (1 << u)) != 0:
            # New available mask: remove u and all its neighbors
            new_mask = available_mask & ~(adj[u] | (1 << u))
            solve(idx + 1, current_score + weights[u], new_mask)
            
        # Branch 2: Exclude u
        # New available mask: just remove u
        new_mask_excl = available_mask & ~(1 << u)
        solve(idx + 1, current_score, new_mask_excl)

    solve(0, 0.0, initial_mask)
    
    return best_set

