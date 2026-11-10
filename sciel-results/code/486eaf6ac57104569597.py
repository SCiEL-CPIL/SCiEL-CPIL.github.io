import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # 1. Compute forbidden differences (nonzero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency masks for the graph G where edges represent forbidden differences.
    # G[u] has bit v set if (u-v) % m is in forbidden.
    # We want Max Independent Set in G, which is Max Clique in G_complement.
    # Let H be G_complement. H[u] has bit v set if u != v and (u-v) % m is NOT in forbidden.
    
    adj_mask = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v and (u - v) % m in forbidden:
                mask |= (1 << v)
        adj_mask[u] = mask

    # Complement adjacency masks H
    full_mask = (1 << m) - 1
    comp_adj_mask = [0] * m
    for u in range(m):
        # H[u] = FullMask ^ (1<<u) ^ adj_mask[u]
        # But careful: adj_mask[u] does not contain u.
        # So H[u] = full_mask & ~(1 << u) & ~adj_mask[u]
        comp_adj_mask[u] = full_mask & ~(1 << u) & ~adj_mask[u]

    best_clique = []
    best_size = 0
    deadline = time.time() + 590.0 # Safety margin
    
    # Helper to check time
    def time_exceeded():
        return time.time() > deadline

    def expand(clique, candidates_mask):
        """
        Recursively expand the clique.
        clique: list of nodes in current clique
        candidates_mask: bitmask of nodes that can be added to clique (adjacent to all in clique in H)
        """
        nonlocal best_clique, best_size
        
        # Popcount of candidates
        num_cands = bin(candidates_mask).count('1')
        
        # Upper bound check
        if len(clique) + num_cands <= best_size:
            return
        
        if time_exceeded():
            return
            
        if candidates_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        # Greedy Coloring of the subgraph H induced by candidates_mask
        # We need to assign colors to nodes in candidates such that adjacent nodes in H have different colors.
        # The max color is an upper bound on the clique size in this subgraph.
        
        # To speed up, we can iterate through candidates.
        # We'll store color for each node in a dictionary or list.
        # Since m=205, a list of size m is efficient.
        color_of_node = [0] * m
        used_colors = [0] * m # Not strictly needed if we just track max, but helps for ordering?
        # Actually, we just need the color value for each node to order them and prune.
        
        # We process nodes in some order. A common heuristic is to process nodes with high degree first 
        # or just iterate through set bits. Let's iterate through set bits.
        # However, the order of coloring affects the tightness of the bound.
        # Standard Tomita uses a specific ordering. 
        # Let's try: Order by degree in H descending? 
        # Calculating degree for each node in candidates:
        # deg(v) = popcount(comp_adj_mask[v] & candidates_mask)
        
        # To avoid O(N^2) degree calc every time, we might just use the order of bits.
        # But let's try a simple greedy coloring where we pick the smallest available color.
        # To make it effective, we should process nodes in a way that conflicts are handled well.
        # Let's just iterate through the set bits of candidates_mask.
        
        # We need a list of nodes to iterate.
        cand_nodes = []
        temp_mask = candidates_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            cand_nodes.append(v)
            temp_mask ^= lsb
            
        # Sort by degree in H descending to improve coloring bound tightness?
        # This costs O(N log N). N is small.
        # deg(v) = bin(comp_adj_mask[v] & candidates_mask).count('1')
        # Let's compute degrees.
        degrees = {}
        for v in cand_nodes:
            degrees[v] = bin(comp_adj_mask[v] & candidates_mask).count('1')
        
        # Sort nodes by degree descending
        cand_nodes.sort(key=lambda x: degrees[x], reverse=True)
        
        max_color = 0
        for v in cand_nodes:
            # Find forbidden colors: colors of neighbors of v in H that are in candidates
            # Neighbors of v in H are bits in comp_adj_mask[v]
            # Intersection with candidates_mask gives neighbors in current subgraph
            neigh_mask = comp_adj_mask[v] & candidates_mask
            
            # We need to know which colors are used by these neighbors.
            # Since we are coloring sequentially, only previously colored nodes matter?
            # No, in greedy coloring for clique bound, we assign colors such that no two *adjacent* nodes have the same color.
            # If we process in any order, when we color v, we must check all already colored neighbors.
            # But wait, if we process in a fixed order, a neighbor might not be colored yet.
            # Standard greedy coloring for this bound:
            # "The color of a vertex is the smallest positive integer not used by any of its neighbors that have already been colored."
            # This works if we process in an order.
            
            forbidden_colors = set()
            # Iterate through neighbors of v in the candidate set
            # We can iterate through bits of neigh_mask
            n_mask = neigh_mask
            while n_mask:
                lsb = n_mask & -n_mask
                u = lsb.bit_length() - 1
                if color_of_node[u] > 0: # Already colored
                    forbidden_colors.add(color_of_node[u])
                n_mask ^= lsb
            
            # Find smallest available color
            c = 1
            while c in forbidden_colors:
                c += 1
            color_of_node[v] = c
            if c > max_color:
                max_color = c
            
        # Now, branch on nodes in decreasing order of color
        # We need to iterate through cand_nodes sorted by color descending
        # If multiple nodes have same color, order doesn't matter much for bound, but for search efficiency?
        # Let's sort by color descending.
        # We also need to maintain the candidate pool as we go.
        
        # Create list of (node, color)
        nodes_with_colors = [(v, color_of_node[v]) for v in cand_nodes]
        nodes_with_colors.sort(key=lambda x: x[1], reverse=True)
        
        current_pool_mask = candidates_mask
        
        for v, c in nodes_with_colors:
            if not (current_pool_mask & (1 << v)):
                continue # Node was removed from pool in a previous branch? 
                         # Actually, in this loop, we remove v after processing.
                         # But if we recurse, we don't modify current_pool_mask for the *next* iteration of this loop?
                         # Wait, the standard Tomita loop:
                         # for v in ordered_candidates:
                         #   if bound fails: break
                         #   recurse(clique + v, pool & neighbors(v))
                         #   pool = pool - v
                         #
                         # The `pool` is local to this level. We update it after recursion returns.
                         # So `current_pool_mask` should be updated.
                         
            # Bound check
            if len(clique) + c <= best_size:
                break
            
            # Recurse
            # New candidates: neighbors of v in H that are in current_pool_mask
            # Note: v is in current_pool_mask. neighbors(v) does not include v.
            new_cands = current_pool_mask & comp_adj_mask[v]
            
            expand(clique + [v], new_cands)
            
            # Remove v from pool
            current_pool_mask &= ~(1 << v)
            
            # Check time
            if time_exceeded():
                return

    # Initial call
    # Candidates are all nodes 0..204
    initial_candidates = (1 << m) - 1
    
    # Start time check
    start_time = time.time()
    
    expand([], initial_candidates)
    
    # If no clique found (should not happen, at least size 1 exists), return single element
    if not best_clique:
        return np.array([0], dtype=np.int64)
        
    return np.array(best_clique, dtype=np.int64)
