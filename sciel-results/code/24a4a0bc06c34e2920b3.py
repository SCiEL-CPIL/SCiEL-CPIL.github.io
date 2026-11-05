import random
import time

def construct():
    n = 20
    best_edges = 0
    best_graph = None
    
    # Precompute all triples and map them to an index for fast C[] access
    triples = []
    triple_index = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                idx = len(triples)
                triples.append((i, j, k))
                triple_index[(i, j, k)] = idx
    
    # C[t] will store the number of common neighbors for triple t
    # We need a fast way to check adjacency
    # adj[u] as a set of neighbors
    
    start_time = time.time()
    limit = 110.0 # Safety margin
    
    for attempt in range(1000):
        if time.time() - start_time > limit:
            break
            
        random.seed(attempt)
        adj = [set() for _ in range(n)]
        C = [0] * len(triples)
        
        # Generate random order of edges
        edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            # Check validity
            valid = True
            
            # Check triples containing u but not v
            # A = {u, a, b}
            # v must be connected to a and b
            # If so, C[A] increases by 1. If C[A] was >= 2, it becomes >= 3 -> Invalid
            
            # To optimize, we only check pairs (a,b) that are neighbors of v
            neighbors_v = adj[v]
            for a in neighbors_v:
                if a == u: continue
                for b in neighbors_v:
                    if b == a or b == u: continue
                    # Check if a, b are in V \ {u, v}
                    # a, b are neighbors of v. If a==u or b==u, skip.
                    # We need to find the triple index for (u, a, b)
                    # Sort u, a, b to find index
                    t = tuple(sorted((u, a, b)))
                    idx = triple_index[t]
                    if C[idx] >= 2:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
                
            # Check triples containing v but not u
            # A = {v, a, b}
            # u must be connected to a and b
            neighbors_u = adj[u]
            for a in neighbors_u:
                if a == v: continue
                for b in neighbors_u:
                    if b == a or b == v: continue
                    t = tuple(sorted((v, a, b)))
                    idx = triple_index[t]
                    if C[idx] >= 2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                
                # Update C
                # Update triples containing u (not v) where v connects to other two
                for a in neighbors_v:
                    if a == u: continue
                    for b in neighbors_v:
                        if b == a or b == u: continue
                        t = tuple(sorted((u, a, b)))
                        idx = triple_index[t]
                        C[idx] += 1
                        
                # Update triples containing v (not u) where u connects to other two
                for a in neighbors_u:
                    if a == v: continue
                    for b in neighbors_u:
                        if b == a or b == v: continue
                        t = tuple(sorted((v, a, b)))
                        idx = triple_index[t]
                        C[idx] += 1
                        
        # Count edges
        edge_count = sum(len(s) for s in adj) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = [[u, v] for u in range(n) for v in adj[u] if u < v]
            
    if best_graph is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_graph
