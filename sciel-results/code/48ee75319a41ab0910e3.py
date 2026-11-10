import random

def construct():
    best_score = 18
    best_edges = None
    
    # Run several random trials
    for trial in range(50):
        random.seed(trial) # Deterministic for reproducibility, or use time? 
        # Let's use a fixed seed for the best known solution if we find one, 
        # but here we search.
        
        edges = set()
        n = 18
        vertices = list(range(n))
        
        # Generate a maximal K4-free graph
        # Order of adding edges matters. Random order.
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            # Check if adding (u,v) creates a K4
            is_k4 = False
            for w in vertices:
                if w == u or w == v: continue
                if (min(u,w), max(u,w)) in edges and (min(v,w), max(v,w)) in edges:
                    # u-v, u-w, v-w are edges.
                    # Check if any x forms K4 with u,v,w?
                    # No, K4 requires 4 vertices.
                    # If we add u-v, we need to check if there is a w such that 
                    # u-w, v-w are edges AND there is an x such that u-x, v-x, w-x are edges?
                    # No. A K4 is formed if there exist w, x such that 
                    # all edges among {u,v,w,x} are present.
                    # When adding u-v, we only need to check if there is a w 
                    # such that u-w and v-w are edges, AND for that w, 
                    # is there an x such that w-x, u-x, v-x are edges?
                    # Actually, if u-v is added, a K4 is formed iff there is a 
                    # triangle u-w-x in the existing graph such that v is connected to u, w, x?
                    # No.
                    # A K4 containing u,v is formed if there is a w such that 
                    # u-w, v-w are edges, and there is an x such that 
                    # u-x, v-x, w-x are edges?
                    # Simpler: 
                    # For each w connected to both u and v, check if there is an x 
                    # connected to u, v, and w.
                    # But we are adding u-v. So u and v are now connected.
                    # We need to check if there is a w such that u-w and v-w are edges.
                    # If such w exists, does it form a K4?
                    # Only if there is an x such that u-x, v-x, w-x are edges.
                    # But we don't know x yet.
                    # Actually, if u-w and v-w are edges, and we add u-v, 
                    # we have a triangle u-v-w.
                    # To form a K4, we need a 4th vertex x connected to u, v, w.
                    # So we need to check: 
                    # For all w in N(u) \cap N(v):
                    #    For all x in N(u) \cap N(v) \cap N(w):
                    #       if x != w, then K4 {u,v,w,x} is formed.
                    
                    # Optimization:
                    # Common neighbors of u and v in the current graph.
                    common = set()
                    for w in vertices:
                        if w == u or w == v: continue
                        if (min(u,w), max(u,w)) in edges and (min(v,w), max(v,w)) in edges:
                            common.add(w)
                            
                    # If common is empty, no triangle formed, so no K4.
                    if not common:
                        continue
                        
                    # Check for K4
                    for w in common:
                        # Check if there is x in common \ {w} such that w-x is an edge
                        for x in common:
                            if x == w: continue
                            if (min(w,x), max(w,x)) in edges:
                                is_k4 = True
                                break
                        if is_k4: break
            if not is_k4:
                edges.add((min(u,v), max(u,v)))
                
        # Calculate score
        # Find largest triangle-free induced subgraph
        # This is NP-hard, but for n=18 we can use branch and bound or simple search.
        # Since n is small, we can check all subsets? 2^18 = 262k. Feasible.
        # But we need to check if induced subgraph is triangle-free.
        
        max_tf_size = 0
        # To speed up, we can just check subsets of size 12, 11, etc.
        # If we find a triangle-free set of size 12, score is at least 12.
        # We want to minimize.
        
        # Let's just compute the exact max size using a simple recursive search or 
        # checking all subsets of size k from 18 down to 1.
        
        # Precompute adjacency matrix for speed
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            
        def is_triangle_free(subset):
            # subset is a list of vertices
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    if adj[subset[i]][subset[j]]:
                        for k in range(j+1, len(subset)):
                            if adj[subset[i]][subset[k]] and adj[subset[j]][subset[k]]:
                                return False
            return True
            
        # Check sizes from 12 down to 1
        found = False
        for size in range(12, 0, -1):
            # Generate combinations? 18C12 = 18564. 
            # For each trial, this is fast.
            from itertools import combinations
            for combo in combinations(vertices, size):
                if is_triangle_free(list(combo)):
                    max_tf_size = size
                    found = True
                    break
            if found:
                break
                
        if max_tf_size < best_score:
            best_score = max_tf_size
            best_edges = list(edges)
            
    return [[u, v] for u, v in best_edges]
