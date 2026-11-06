import time
from math import gcd

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b.
    
    Uses branch-and-bound maximum clique search on the graph where
    vertices are numbers in [1,255] sharing a factor with 255, and
    edges connect pairs with gcd > 1.
    """
    deadline = time.time() + 80  # 80s guard, 10s margin
    
    # Build candidate set: numbers in [1,255] that share a factor with 255
    # 255 = 3 * 5 * 17
    candidates = [x for x in range(1, 256) if gcd(x, 255) > 1]
    n = len(candidates)
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # 255 must be in the set. Find its index.
    idx_255 = val_to_idx[255]
    
    # Build adjacency: adj[i] is a bitset (int) of vertices adjacent to i
    # Two vertices are adjacent if gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n
    for i in range(n):
        ci = candidates[i]
        for j in range(i + 1, n):
            if gcd(ci, candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Since 255 is connected to all candidates (all share factor with 255),
    # the max clique containing 255 equals max clique in the full graph.
    # We'll find max clique in the induced subgraph on all n vertices.
    
    # Precompute: for each vertex, its neighbors as a set for quick access
    # We'll use bitset operations for speed
    
    # Branch and bound for maximum clique
    best_clique = set()
    best_size = 0
    
    # Greedy coloring for upper bound
    # Color the graph (complement or original?) - for max clique, we use
    # coloring on the subgraph to get upper bounds.
    
    # We'll use a recursive branch-and-bound with:
    # - Current clique
    # - Candidate set (vertices that can be added)
    # - Upper bound from greedy coloring of candidate subgraph
    
    # For efficiency, use bitsets
    def greedy_color_upper_bound(cand_mask):
        """
        Given a bitmask of candidate vertices, compute a greedy coloring
        of the induced subgraph and return the number of colors used
        (upper bound on clique size in that subgraph).
        """
        # Greedy coloring: process vertices in some order, assign smallest color
        # that doesn't conflict with already-colored neighbors in cand_mask
        # The number of colors is an upper bound on the max clique.
        
        # Extract vertices from mask
        vertices = []
        m = cand_mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            vertices.append(idx)
            m ^= lsb
        
        if not vertices:
            return 0
        
        # For better bounds, order vertices by degree in cand_mask (descending)
        # Compute degrees within cand_mask
        deg = {}
        for v in vertices:
            # Count neighbors of v that are in cand_mask
            neighbors = adj[v] & cand_mask
            deg[v] = bin(neighbors).count('1')
        
        # Sort by degree descending
        vertices.sort(key=lambda x: -deg[x])
        
        # Greedy coloring
        color = {}
        max_color = 0
        for v in vertices:
            # Find smallest color not used by v's colored neighbors in cand_mask
            neighbor_colors = set()
            for u in vertices:
                if u == v:
                    continue
                if u in color and (adj[v] >> u) & 1:
                    neighbor_colors.add(color[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            color[v] = c
            max_color = max(max_color, c)
        
        return max_color + 1  # colors are 0-indexed, so +1
    
    def branch_and_bound(clique, cand_mask, depth=0):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        # Upper bound: len(clique) + greedy_color_upper_bound(cand_mask)
        ub = len(clique) + greedy_color_upper_bound(cand_mask)
        if ub <= best_size:
            return
        
        # If no candidates, update best
        if cand_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = set(clique)
            return
        
        # Get vertices from cand_mask
        # Try to add vertices one by one
        # Order: try vertices with higher degree first
        
        m = cand_mask
        vertices = []
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            vertices.append(idx)
            m ^= lsb
        
        # Sort by degree in cand_mask descending
        vertices.sort(key=lambda x: -bin(adj[x] & cand_mask).count('1'))
        
        for v in vertices:
            if time.time() > deadline:
                return
            
            # Check if v is still in cand_mask
            if not (cand_mask >> v) & 1:
                continue
            
            new_clique = clique + [v]
            # New candidates: intersection of cand_mask and neighbors of v
            new_cand = cand_mask & adj[v]
            
            branch_and_bound(new_clique, new_cand, depth + 1)
            
            # Remove v from cand_mask for subsequent iterations
            cand_mask &= ~(1 << v)
            
            # Update upper bound check
            ub = len(clique) + greedy_color_upper_bound(cand_mask)
            if ub <= best_size:
                return
    
    # Initial call: clique starts empty, all candidates available
    # But we want cliques that include 255. Since 255 is connected to all,
    # we can just find max clique in the full graph.
    all_mask = (1 << n) - 1
    branch_and_bound([], all_mask)
    
    # Convert best_clique indices back to values
    if best_clique:
        result = [candidates[i] for i in best_clique]
        # Ensure 255 is in result
        if 255 not in result:
            result.append(255)
        return sorted(result)
    else:
        # Fallback: multiples of 3
        return list(range(3, 256, 3))
