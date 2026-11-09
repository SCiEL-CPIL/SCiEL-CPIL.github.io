import numpy as np
import time
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Build adjacency bitsets
    # adj[i] is an integer where bit j is 1 if (i, j) is an edge in G
    # Edge exists if (i-j) % m is in forbidden
    adj = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            diff = (i - j) % m
            if diff in forbidden:
                mask |= (1 << j)
        adj[i] = mask
        
    # 3. Branch and Bound for Maximum Independent Set
    # We want to find a set of vertices with no edges between them.
    # This is equivalent to finding a clique in the complement graph, 
    # or just MIS in G.
    
    best_size = 0
    best_set = 0  # Bitset of the best independent set found
    
    # Precompute a static ordering of vertices by degree (descending) to help branching
    # Vertices with higher degree are more "constraining", so branching on them first 
    # often prunes the search space faster.
    degrees = [bin(adj[i]).count('1') for i in range(m)]
    order = list(range(m))
    order.sort(key=lambda x: degrees[x], reverse=True)
    
    # Heuristic upper bound: Chromatic number of complement graph 
    # approximated by greedy coloring of complement (or clique cover of G).
    # A simpler valid upper bound is the size of the candidate set, but 
    # greedy coloring is much tighter.
    # Let's implement a greedy coloring on the complement graph.
    # Complement edge (i, j) exists if (i, j) is NOT an edge in G.
    # Coloring the complement graph is equivalent to partitioning G into cliques.
    # The number of colors is an upper bound on the independence number of G.
    
    def greedy_color_upper_bound(candidates):
        # candidates is a bitset of available vertices
        if candidates == 0:
            return 0
        
        # Greedy coloring: 
        # 1. Pick a vertex v
        # 2. Assign it a color
        # 3. Remove v and its neighbors in the complement graph (which are non-neighbors in G)
        # Wait, standard greedy coloring:
        # While vertices remain:
        #   Pick a vertex v
        #   Create a color class C containing v
        #   Add any uncolored vertex u to C if u is NOT adjacent to v in the complement graph
        #   (i.e., u IS adjacent to v in G, or u==v)
        #   Actually, a clique in G is an independent set in complement.
        #   So we are partitioning into cliques of G.
        #   A clique in G: all pairs are connected in G.
        #   Greedy clique cover:
        #   While vertices remain:
        #      Pick v
        #      C = {v}
        #      For each uncolored u:
        #         If u is connected to ALL vertices in C in G, add u to C
        #      Remove C from candidates
        #      Count++
        
        remaining = candidates
        colors = 0
        while remaining:
            # Pick a vertex, e.g., the first one in 'order' that is in remaining
            v = None
            for u in order:
                if (remaining >> u) & 1:
                    v = u
                    break
            if v is None: break
            
            colors += 1
            
            # Build a clique containing v
            clique_mask = 1 << v
            # Potential additions must be in remaining and adjacent to v
            # And adjacent to all other members of the clique
            # We can iterate through remaining vertices and check if they connect to the whole clique
            # To be efficient, we can just check against the current clique_mask
            # A vertex u can be added to the clique if (adj[u] & clique_mask) == clique_mask
            # This means u is connected to every vertex in the clique.
            
            # We need to remove the clique from remaining
            # Let's find all vertices in remaining that can join the clique
            # Note: We must be careful. If we add u, we must ensure it connects to all.
            # Since we build the clique incrementally, this works.
            
            # Optimization: Just pick v, then find all u in remaining such that 
            # u is connected to v AND u is connected to all previously added nodes.
            # Since we add one by one, we just check against the current clique_mask.
            
            # However, a simpler greedy strategy for upper bound is just to count 
            # how many "rounds" it takes to cover all vertices with cliques.
            # Let's refine the clique selection: 
            # Start with v. 
            # Any u in remaining is a candidate if it is connected to v.
            # Then filter those candidates to be connected to the next added node, etc.
            
            # Actually, for an upper bound, we just need a valid clique cover.
            # A maximal clique containing v is a good choice.
            
            current_clique = 1 << v
            remaining_for_clique = remaining & adj[v] # Candidates must be neighbors of v
            
            # We want to add as many as possible to the clique to reduce 'remaining' quickly?
            # No, we want to cover 'remaining' with as FEW cliques as possible.
            # So we want LARGE cliques.
            # So we should greedily add vertices that are connected to the current clique.
            
            # Iterate through remaining candidates (excluding v)
            # We can use a while loop to add one by one
            added = True
            while added:
                added = False
                # Find a vertex in remaining_for_clique that is connected to current_clique
                # Note: remaining_for_clique contains vertices connected to v.
                # As we add to current_clique, we must intersect with adj of new vertex.
                
                # To keep it simple and fast, let's just take the maximal clique
                # formed by greedily adding vertices from 'order' that are in remaining
                # and connected to the current clique.
                
                # Reset candidate pool for this clique
                # Candidates are in 'remaining'.
                # We already picked v.
                # Let's try to add others.
                
                # We'll iterate through 'order' and check if they can be added.
                # If a vertex u is in remaining and (adj[u] & current_clique) == current_clique, 
                # we can add it.
                
                # To make it deterministic and fast, we just scan 'order'.
                for u in order:
                    if u == v: continue
                    if (remaining >> u) & 1:
                        if (adj[u] & current_clique) == current_clique:
                            current_clique |= (1 << u)
                            added = True # We added one, so we might be able to add more? 
                                        # Actually, adding u restricts future additions, 
                                        # but doesn't invalidate previous ones.
                                        # We can continue scanning.
            
            # Remove the clique from remaining
            remaining &= ~current_clique
            
        return colors

    # Initial upper bound
    all_mask = (1 << m) - 1
    current_upper_bound = greedy_color_upper_bound(all_mask)
    
    # If the bound is already 0 or 1, handle edge cases (though m=205, bound will be > 1)
    
    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_size, best_set, current_upper_bound
        
        if time.time() > deadline:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        # Pruning
        # Upper bound for this branch
        # The max size we can add is bounded by the clique cover of 'candidates'
        # But calculating the full clique cover is expensive.
        # A simpler bound: current_size + popcount(candidates)
        # A tighter bound: current_size + greedy_color_upper_bound(candidates)
        
        # Let's use a simple bound first: if current_size + popcount(candidates) <= best_size, prune
        if current_size + bin(candidates).count('1') <= best_size:
            return
            
        # Tighter bound:
        # ub = greedy_color_upper_bound(candidates)
        # if current_size + ub <= best_size:
        #     return
        
        # For speed, we might skip the tight bound in deep recursion or use a lighter heuristic.
        # Given m=205, the graph is not huge. Let's try the tight bound but with a check.
        # If the candidate set is small, the bound is cheap.
        
        # Heuristic: Pick a vertex to branch on
        # Choose vertex with max degree in the induced subgraph of candidates?
        # Or just the first one in 'order' that is in candidates.
        
        v = None
        for u in order:
            if (candidates >> u) & 1:
                v = u
                break
                
        if v is None:
            return
            
        # Branch 1: Include v
        # New candidates: candidates minus v and its neighbors
        # neighbors of v in candidates: candidates & adj[v]
        # new_candidates = candidates & ~(adj[v] | (1 << v))
        
        include_candidates = candidates & ~(adj[v] | (1 << v))
        branch_and_bound(include_candidates, current_set | (1 << v), current_size + 1)
        
        # Branch 2: Exclude v
        # New candidates: candidates minus v
        exclude_candidates = candidates & ~(1 << v)
        branch_and_bound(exclude_candidates, current_set, current_size)

    # Set deadline
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    # Initial call
    # We can start with all vertices
    branch_and_bound(all_mask, 0, 0)
    
    # Extract result
    result = []
    for i in range(m):
        if (best_set >> i) & 1:
            result.append(i)
            
    # If for some reason no set was found (shouldn't happen, at least size 1), return [0]
    if not result:
        result = [0]
        
    return np.array(result, dtype=np.int64)
