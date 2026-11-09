import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    N = 512
    nodes = list(range(1, N))
    n = len(nodes)
    
    # Precompute scores
    scores = [0.0] * n
    for i, a in enumerate(nodes):
        scores[i] = 1.0 / (N - a)
        
    # Sort nodes by score descending
    # We will re-index nodes to 0..n-1 based on this sort
    sorted_nodes = sorted(nodes, key=lambda x: -1.0 / (N - x))
    # Map original value to new index
    val_to_idx = {v: i for i, v in enumerate(sorted_nodes)}
    # Re-map scores to new indices
    sorted_scores = [1.0 / (N - v) for v in sorted_nodes]
    
    # Precompute coprime adjacency in the new indexing
    # adj[i][j] = True if sorted_nodes[i] and sorted_nodes[j] are coprime
    # We want Max Weight Clique on this graph.
    # So edges are coprime pairs.
    
    # For efficiency, use bitmasks for neighbors in the coprime graph
    # neighbors[i] is a bitmask of j such that gcd(sorted_nodes[i], sorted_nodes[j]) == 1
    neighbors = [0] * n
    for i in range(n):
        vi = sorted_nodes[i]
        mask = 0
        for j in range(n):
            if i == j: continue
            vj = sorted_nodes[j]
            if gcd(vi, vj) == 1:
                mask |= (1 << j)
        neighbors[i] = mask
        
    # Initial best solution
    # Greedy: pick highest score, then next highest coprime, etc.
    def greedy_init():
        chosen = []
        chosen_mask = 0 # mask of chosen indices
        # We need to track which are available
        # Since we pick highest first, we just iterate sorted_scores
        # But we must ensure pairwise coprime.
        # The "neighbors" mask tells us who is coprime.
        # If we pick i, we can only pick from neighbors[i].
        
        # Simple greedy:
        # 1. Pick 0 (highest score).
        # 2. Candidates = neighbors[0].
        # 3. Pick highest score in Candidates.
        # 4. Candidates = Candidates & neighbors[picked].
        
        # Let's implement this.
        current_cands = (1 << n) - 1
        best_w = 0.0
        chosen_indices = []
        
        while current_cands:
            # Find highest score index in current_cands
            # Since scores are sorted desc, we can just find the lowest set bit?
            # No, indices are 0..n-1, scores[0] >= scores[1]...
            # So the lowest set bit is the highest score.
            # Wait, bit 0 is index 0.
            # So we want the lowest set bit.
            # But we need to iterate?
            # Actually, just find the first set bit from left?
            # No, index 0 is highest score.
            # So we want the smallest index i such that (current_cands >> i) & 1 is 1.
            
            v = -1
            for i in range(n):
                if (current_cands >> i) & 1:
                    v = i
                    break
            
            if v == -1: break
            
            chosen_indices.append(v)
            best_w += sorted_scores[v]
            
            # Update candidates: must be coprime with v
            current_cands &= neighbors[v]
            
        return chosen_indices, best_w

    best_indices, best_weight = greedy_init()
    
    # Branch and Bound
    # State:
    # candidates: bitmask of available nodes
    # current_weight: sum of weights of chosen nodes
    # chosen: list of chosen indices
    
    import sys
    sys.setrecursionlimit(100000)
    
    def compute_bound(candidates_mask):
        # Greedy coloring of the subgraph induced by candidates_mask in the coprime graph
        # We process vertices in order of decreasing weight (which is increasing index in our sorted list? No, decreasing index).
        # Our sorted_nodes are sorted by weight DESC.
        # So index 0 is highest weight.
        # We iterate i from 0 to n-1.
        # If i is in candidates_mask, try to assign it to a color.
        
        # Colors: list of lists of vertices? Or just track the "max weight" which is the first vertex assigned.
        # Since we process in weight order, the first vertex assigned to a color is the max weight in that color.
        # So we just need to count the number of colors? No, we need the sum of the weights of the leaders.
        
        # Leader weights:
        # When we assign vertex i to a new color, the leader weight is sorted_scores[i].
        # When we assign vertex i to an existing color, the leader weight is already set (and is higher than sorted_scores[i]).
        # So the bound is the sum of sorted_scores[i] for all i that start a new color.
        
        # How to check if i can be assigned to color c?
        # i must NOT be adjacent to any vertex in color c in the coprime graph.
        # i.e., i must be NON-coprime with all vertices in color c.
        # i.e., for all u in color_c, (neighbors[i] & (1<<u)) == 0.
        # This is equivalent to: (neighbors[i] & color_mask_c) == 0.
        
        # We can maintain color_masks: list of bitmasks.
        # For each vertex i in candidates (in order of weight desc):
        #   For each color c:
        #       if (neighbors[i] & color_masks[c]) == 0:
        #           color_masks[c] |= (1 << i)
        #           break
        #   else:
        #       new_color = len(color_masks)
        #       color_masks.append(1 << i)
        #       bound += sorted_scores[i]
        
        # Optimization:
        # We only care about the bound value.
        # We don't need to store the actual colors if we just want the sum.
        # But we need to know if i fits in an existing color.
        
        color_masks = []
        bound = 0.0
        
        # Iterate candidates in order of weight desc
        # This is just iterating i from 0 to n-1 and checking if i is in candidates_mask
        for i in range(n):
            if not (candidates_mask >> i) & 1:
                continue
                
            # Try to fit i into existing colors
            placed = False
            for c in range(len(color_masks)):
                if (neighbors[i] & color_masks[c]) == 0:
                    color_masks[c] |= (1 << i)
                    placed = True
                    break
            
            if not placed:
                color_masks.append(1 << i)
                bound += sorted_scores[i]
                
        return bound

    def branch_and_bound(candidates_mask, current_weight, chosen_list):
        nonlocal best_weight, best_indices
        
        if time.time() > deadline:
            return
            
        if current_weight > best_weight:
            best_weight = current_weight
            best_indices = chosen_list[:]
            
        if candidates_mask == 0:
            return
            
        # Compute bound
        ub = compute_bound(candidates_mask)
        if current_weight + ub <= best_weight:
            return
            
        # Select pivot
        # Choose vertex with highest weight in candidates?
        # Or highest degree?
        # Tomita uses degree.
        # Let's try highest weight first, as it's simple and effective for weighted problems.
        v = -1
        for i in range(n):
            if (candidates_mask >> i) & 1:
                v = i
                break
                
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates: candidates_mask & neighbors[v]
        # Note: neighbors[v] contains v? No, we set mask for j != i.
        # So neighbors[v] does not contain v.
        new_cands = candidates_mask & neighbors[v]
        branch_and_bound(new_cands, current_weight + sorted_scores[v], chosen_list + [v])
        
        # Branch 2: Exclude v
        new_cands = candidates_mask & ~(1 << v)
        branch_and_bound(new_cands, current_weight, chosen_list)

    initial_mask = (1 << n) - 1
    branch_and_bound(initial_mask, 0.0, [])
    
    # Convert indices back to values
    result = [sorted_nodes[i] for i in best_indices]
    
    # Optional: Local search / Polish
    # The B&B should find the optimum if it finishes.
    # If it times out, we return the best found.
    
    return result
