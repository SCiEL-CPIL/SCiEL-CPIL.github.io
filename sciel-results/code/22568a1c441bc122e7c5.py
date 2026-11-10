import time
import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def greedy_packing_partition(adj, n, deadline):
        """
        Greedy edge packing:
        1. Collect all edges.
        2. Sort edges by a heuristic (e.g., sum of degrees of endpoints) to prioritize "constrained" edges first.
        3. Maintain a list of cliques (as lists of vertices and bitmasks of vertices).
        4. For each edge (u, v):
           - Try to fit it into an existing clique C if (u in C or v in C is not required, just that C U {u,v} is a clique).
             Actually, the condition is: u and v must be connected to ALL vertices in C, and u-v is an edge (which it is).
           - If no existing clique fits, create a new clique [u, v].
        """
        # Collect edges
        edges = []
        for u in range(n):
            # Iterate over set bits in adj[u] > u
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    # Heuristic score: sum of degrees. Higher degree = more constrained = place earlier
                    score = adj[u].bit_count() + adj[v].bit_count()
                    edges.append((score, u, v))
                mask ^= lsb
        
        # Sort by score descending
        edges.sort(key=lambda x: -x[0])
        
        # List of cliques: each is a dict or tuple (vertex_list, vertex_mask, last_checked_index?)
        # To check if edge (u,v) fits in clique C:
        # We need (adj[u] & adj[v] & C_mask) == C_mask.
        # i.e., u and v are connected to all vertices in C.
        
        cliques = []
        
        # Optimization: 
        # Checking all cliques for every edge is O(E * K * n/word_size).
        # E ~ n^2/2. K ~ n^2/2 (worst case). n=64.
        # 2048 * 2048 * 1 op = 4M ops. Very fast.
        
        # However, as cliques grow, they become harder to fit new edges into.
        # We might want to try smaller cliques first? Or larger?
        # If we try larger cliques first, we might fail and create a new one, whereas a smaller one would have fit.
        # But if a large clique fits, it's a "good" fit because it reuses a "strong" clique.
        # Actually, if a clique fits, it doesn't matter which one we pick for validity, but it affects future packing.
        # Picking the "most constrained" existing clique (largest?) might be better?
        # Or picking the smallest?
        # Let's just iterate through existing cliques in order of creation (or maybe reverse order, newest first?).
        # Newest cliques are likely smaller and more flexible.
        # Let's try iterating from the most recently added clique backwards.
        
        for score, u, v in edges:
            # Check existing cliques
            placed = False
            # Iterate backwards to check newer (smaller) cliques first
            for i in range(len(cliques) - 1, -1, -1):
                clique_verts, clique_mask = cliques[i]
                
                # Check if u and v are connected to all vertices in clique_mask
                # u must connect to all in clique_mask
                # v must connect to all in clique_mask
                if (adj[u] & clique_mask) == clique_mask and (adj[v] & clique_mask) == clique_mask:
                    # It fits!
                    # Update clique
                    # We store as list and mask
                    # If u or v is already in clique, we don't add them again?
                    # The problem says "clique partition of edges". 
                    # "Every listed set must be a clique... every edge must lie in exactly one listed clique."
                    # It does NOT say vertices cannot be repeated in different cliques.
                    # It DOES say the set itself must be a clique.
                    # If u is already in clique, adding it again doesn't change the set.
                    # But we are adding the EDGE (u,v).
                    # If v is already in clique, then the edge (u,v) is covered by adding u to the clique?
                    # Wait. The "clique" is the set of vertices.
                    # If the set is {1, 2, 3}, the edges covered are (1,2), (1,3), (2,3).
                    # If we have clique {1, 2} and we want to add edge (2, 3), we add 3 to the clique -> {1, 2, 3}.
                    # Now edges (1,3) and (2,3) are covered.
                    # What if (1,3) was already covered by another clique? That's invalid.
                    # So we must ensure that NO edge in the new clique has been covered before.
                    # 
                    # My current logic: "assign each edge to the smallest-indexed clique".
                    # This implies edges are disjoint.
                    # If I add vertex 3 to clique {1,2}, I cover (2,3) and (1,3).
                    # I must ensure (1,3) was NOT already covered.
                    # 
                    # So the check "fits" is not enough.
                    # I need to ensure that all edges incident to u and v within the clique are FREE.
                    # 
                    # Let's maintain a global edge coverage mask? 
                    # n=64, edges ~ 2000. 
                    # We can maintain a set of covered edges.
                    # When adding (u,v) to clique C, we add all edges (x,y) where x,y in C U {u,v} that involve u or v?
                    # No, adding u and v to C covers all edges between {u,v} and C, and (u,v).
                    # We must verify that none of these edges are already covered.
                    
                    # This makes the "fits" check more expensive and stateful.
                    # Let's maintain `covered_edges` as a set of tuples or a bitmask of edges?
                    # Bitmask of edges is hard to index.
                    # Set of tuples is fine for 2000 edges.
                    
                    # Let's refine the loop.
                    pass
            
            # If not placed, create new clique
            if not placed:
                cliques.append(([u, v], (1 << u) | (1 << v)))
                # Cover edge (u,v)
                # We need to track covered edges to prevent double covering.
                # Let's restart the logic with edge tracking.
        
        # RESTART LOGIC WITH EDGE TRACKING
        # This is a crucial correction. The previous "fits" check was insufficient.
        
        cliques = []
        covered_edges = set()
        
        # Pre-compute edge indices or just use tuples
        # Sort edges again
        for score, u, v in edges:
            if (u, v) in covered_edges:
                continue
            
            placed = False
            # Try to add to existing cliques
            # We should try cliques where u or v is already present?
            # If u is in clique C, adding v covers (u,v) and (v, x) for all x in C.
            # If u is NOT in C, adding u and v covers (u,v), (u,x), (v,x) for all x in C.
            
            # Strategy: Try to add to cliques containing u or v first?
            # Or just iterate all?
            # Iterating all is O(K).
            
            for i in range(len(cliques) - 1, -1, -1):
                c_verts, c_mask = cliques[i]
                
                # Check if adding u and v to this clique is valid
                # 1. u and v must be connected to all vertices in c_mask
                if not ((adj[u] & c_mask) == c_mask and (adj[v] & c_mask) == c_mask):
                    continue
                
                # 2. Check if all new edges are uncovered
                # New edges are:
                # (u, v)
                # (u, x) for x in c_verts
                # (v, x) for x in c_verts
                
                # If u is in c_verts, (u, x) are already in the clique? 
                # No, the clique is a SET. If u is in c_verts, the edges (u, x) were already "covered" by the existence of u in the clique?
                # Wait. The "clique partition" defines the set of edges covered by the UNION of the cliques.
                # If u is in C, then all edges (u, x) for x in C are covered by C.
                # So if u is in C, we don't "add" those edges, they are already covered.
                # We only "add" edges involving the NEW vertices.
                
                new_vertices = []
                if u not in c_verts:
                    new_vertices.append(u)
                if v not in c_verts:
                    new_vertices.append(v)
                
                # Edges to check:
                # 1. (u, v) if both are new or one is new?
                #    If u in C and v not in C: Edge (u,v) is new. Edges (v, x) for x in C are new.
                #    If u not in C and v in C: Edge (u,v) is new. Edges (u, x) for x in C are new.
                #    If u not in C and v not in C: Edge (u,v) is new. Edges (u,x), (v,x) for x in C are new.
                
                # Let's just generate all pairs in (c_verts U {u, v}) that involve at least one of {u, v} 
                # AND were not previously in c_verts?
                # Actually, simpler:
                # The set of edges covered by C is E(C).
                # The set of edges covered by C' = C U {u, v} is E(C').
                # The new edges are E(C') \ E(C).
                # These are exactly the edges incident to u or v that connect to vertices in C, plus (u,v).
                
                # Check if any of these are in covered_edges.
                # Note: covered_edges contains edges from ALL cliques.
                # If an edge is in C, it is in covered_edges.
                
                valid = True
                # Check (u, v)
                if u != v and (min(u,v), max(u,v)) in covered_edges:
                    valid = False
                else:
                    # Check (u, x) for x in c_verts
                    if valid:
                        for x in c_verts:
                            if x == u: continue
                            if (min(u,x), max(u,x)) in covered_edges:
                                valid = False
                                break
                    # Check (v, x) for x in c_verts
                    if valid:
                        for x in c_verts:
                            if x == v: continue
                            if (min(v,x), max(v,x)) in covered_edges:
                                valid = False
                                break
                    
                if valid:
                    # Place it
                    # Update clique
                    new_verts = list(c_verts)
                    if u not in new_verts:
                        new_verts.append(u)
                    if v not in new_verts:
                        new_verts.append(v)
                    
                    new_mask = c_mask | (1 << u) | (1 << v)
                    cliques[i] = (new_verts, new_mask)
                    
                    # Update covered_edges
                    if u != v:
                        covered_edges.add((min(u,v), max(u,v)))
                    for x in c_verts:
                        if x == u: continue
                        covered_edges.add((min(u,x), max(u,x)))
                        if x == v: continue
                        covered_edges.add((min(v,x), max(v,x)))
                    
                    placed = True
                    break
            
            if not placed:
                # Create new clique
                new_verts = [u, v]
                new_mask = (1 << u) | (1 << v)
                cliques.append((new_verts, new_mask))
                if u != v:
                    covered_edges.add((min(u,v), max(u,v)))
                
            # Time check
            if time.time() > deadline:
                break
        
        # Safety: cover any remaining edges
        # This loop is O(n^2)
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    if (u, v) not in covered_edges:
                        # Add as a new clique or try to fit?
                        # Just add as new clique for safety/speed
                        cliques.append(([u, v], (1 << u) | (1 << v)))
                        covered_edges.add((u, v))
                mask ^= lsb
        
        # Return list of vertex lists
        return [c[0] for c in cliques]

    start_time = time.time()
    deadline = start_time + 140
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    cliques_1 = greedy_packing_partition(adj_1, 32, deadline)
    cliques_2 = greedy_packing_partition(adj_2, 64, deadline)
    
    return [cliques_1, cliques_2]

if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total: {len(res[0]) + len(res[1])}")
