def construct():
    from math import lcm
    from fractions import Fraction
    import sys

    N = 256
    nums = list(range(2, 257))
    n = len(nums)
    
    # Build Conflict Graph (comp_adj): edge if lcm <= 256
    comp_adj = [0] * n
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            bj = nums[j]
            if lcm(ai, bj) <= N:
                comp_adj[i] |= (1 << j)
                comp_adj[j] |= (1 << i)
                
    # Weights
    weights = [1.0 / nums[i] for i in range(n)]
    
    best_weight = 0.0
    best_clique = []
    
    # Helper: Get maximal matching bound
    def get_matching_bound(candidates):
        # candidates is a bitmask
        # Extract nodes
        cands = []
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            cands.append(idx)
            tmp ^= lsb
            
        if not cands:
            return 0.0
            
        # Maximal Matching
        # Simple greedy matching
        # Iterate nodes, if not matched and has unmatched neighbor, match them.
        matched = [False] * len(cands)
        # Map node -> index in cands
        node_to_idx = {node: i for i, node in enumerate(cands)}
        
        bound = 0.0
        unmatched_sum = 0.0
        
        for i, node in enumerate(cands):
            if matched[i]:
                continue
                
            # Find an unmatched neighbor
            neigh_mask = comp_adj[node] & candidates
            # Iterate neighbors
            nm = neigh_mask
            matched_neighbor = -1
            while nm:
                lsb = nm & (-nm)
                n_idx = lsb.bit_length() - 1
                # Check if this neighbor is in cands and not matched
                if n_idx in node_to_idx:
                    j = node_to_idx[n_idx]
                    if not matched[j]:
                        matched_neighbor = j
                        break
                nm ^= lsb
                
            if matched_neighbor != -1:
                # Match i and matched_neighbor
                matched[i] = True
                matched[matched_neighbor] = True
                bound += max(weights[cands[i]], weights[cands[matched_neighbor]])
            else:
                # No unmatched neighbor, so this node is unmatched in this matching
                # It contributes its full weight to the bound
                bound += weights[node]
                
        return bound

    def expand(candidates, current_weight, current_set):
        nonlocal best_weight, best_clique
        
        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = current_set[:]
            return
            
        # Bound
        bound = get_matching_bound(candidates)
        
        if current_weight + bound <= best_weight:
            return
            
        # Pick a node to branch on
        # Heuristic: Pick node with highest weight? Or highest degree?
        # Let's pick highest weight.
        cands = []
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            cands.append(idx)
            tmp ^= lsb
            
        # Sort by weight desc? 
        # cands is not sorted by weight.
        # Let's find max weight node.
        max_idx = -1
        max_w = -1.0
        for idx in cands:
            if weights[idx] > max_w:
                max_w = weights[idx]
                max_idx = idx
                
        v = max_idx
        
        # Branch 1: Include v
        # New candidates: remove v and its neighbors
        new_cands = candidates & ~(1 << v) & ~comp_adj[v]
        expand(new_cands, current_weight + weights[v], current_set + [v])
        
        # Branch 2: Exclude v
        new_cands = candidates & ~(1 << v)
        expand(new_cands, current_weight, current_set)

    all_nodes = (1 << n) - 1
    expand(all_nodes, 0.0, [])
    
    result = [nums[i] for i in best_clique]
    return result
