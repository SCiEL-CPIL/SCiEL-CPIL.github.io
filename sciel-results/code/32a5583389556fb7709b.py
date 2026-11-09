import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # 1. Compute forbidden differences (quadratic residues)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency bitsets for G (edges where diff is in forbidden)
    # adj[i] has bit j set if (i-j) % m is in forbidden
    adj = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i != j and (i - j) % m in forbidden:
                mask |= (1 << j)
        adj[i] = mask
        
    # 3. Precompute a static ordering.
    # Vertices with higher degree in G are more constrained for MIS?
    # Actually, for MIS, we want to pick vertices that are "isolated" or have few neighbors?
    # In B&B for MIS, branching on a vertex with high degree in the *remaining* graph
    # often prunes the "Include" branch more aggressively (removes many candidates).
    # Let's sort by degree in G descending.
    degrees = [bin(adj[i]).count('1') for i in range(m)]
    order = list(range(m))
    order.sort(key=lambda x: degrees[x], reverse=True)
    
    # Map vertex to its index in order for faster lookup? 
    # Or just iterate order to find first available.
    
    best_size = 0
    best_set = 0
    deadline = time.time() + 590 # 10s margin
    
    # Greedy Coloring of the Complement Graph (Clique Cover of G)
    # Returns an upper bound on the size of the Maximum Independent Set in G[candidates]
    def greedy_color_upper_bound(candidates):
        if candidates == 0:
            return 0
            
        remaining = candidates
        colors = 0
        
        # We want to cover 'remaining' with cliques of G.
        # Each clique corresponds to a color in the complement graph.
        # The number of colors is an upper bound on the independence number.
        
        while remaining:
            colors += 1
            
            # Pick a vertex v from remaining.
            # Heuristic: Pick the one with highest degree in the induced subgraph?
            # For simplicity and speed, pick the first one in 'order'.
            v = -1
            for u in order:
                if (remaining >> u) & 1:
                    v = u
                    break
            
            if v == -1:
                break
                
            # Build a maximal clique containing v in G[remaining]
            # Start with v
            clique = 1 << v
            # Candidates to add to clique: neighbors of v in G, that are in remaining
            # Let's iterate through 'remaining' and add vertices that connect to the whole clique.
            
            # To make it fast, we can just scan 'order' again.
            # A vertex u can be added to clique if:
            # 1. u is in remaining
            # 2. u is connected to ALL vertices in clique in G
            
            # We can maintain a mask of "compatible" vertices.
            # Initially, compatible = remaining & adj[v]
            # When we add u to clique, compatible &= adj[u]
            
            compatible = remaining & adj[v]
            
            # Greedily add vertices from 'order' to the clique
            # Note: We must ensure we don't add v again (it's not in compatible usually, or we mask it out)
            # adj[v] does not contain v (no self-loops).
            
            while compatible:
                # Find next vertex in 'order' that is in compatible
                u = -1
                for w in order:
                    if (compatible >> w) & 1:
                        u = w
                        break
                
                if u == -1:
                    break
                    
                # Add u to clique
                clique |= (1 << u)
                # Update compatible: must be neighbor of u
                compatible &= adj[u]
                
            # Remove the entire clique from remaining
            remaining &= ~clique
            
        return colors

    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        # Pruning
        # Upper bound on how many more vertices we can add
        ub = greedy_color_upper_bound(candidates)
        if current_size + ub <= best_size:
            return
        
        # Select vertex to branch on
        # Heuristic: Pick vertex with max degree in induced subgraph?
        # Or just first in static order.
        # Let's try picking the vertex with the highest degree in the subgraph defined by 'candidates'.
        # This requires calculating degrees, which is O(N^2) or O(N * bit_count).
        # Given N=205, it's fine.
        
        # Alternative: Just use static order. It's faster.
        v = -1
        for u in order:
            if (candidates >> u) & 1:
                v = u
                break
                
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates: candidates minus v and its neighbors
        include_candidates = candidates & ~(adj[v] | (1 << v))
        branch_and_bound(include_candidates, current_set | (1 << v), current_size + 1)
        
        # Branch 2: Exclude v
        exclude_candidates = candidates & ~(1 << v)
        branch_and_bound(exclude_candidates, current_set, current_size)

    all_mask = (1 << m) - 1
    branch_and_bound(all_mask, 0, 0)
    
    # Extract result
    result = []
    for i in range(m):
        if (best_set >> i) & 1:
            result.append(i)
            
    if not result:
        result = [0]
        
    return np.array(result, dtype=np.int64)
