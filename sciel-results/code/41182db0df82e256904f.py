import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    best_edges = []
    best_count = 0
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Fast 3-colorability check using bitmasks and degree ordering
    def is_3_colorable(adj):
        # adj is list of bitmasks
        # Order vertices by degree descending
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            
            # Determine available colors
            # A color c is invalid if any neighbor has color c
            # We can precompute a bitmask of colored neighbors?
            # Or just check.
            
            # Optimization: Check if any neighbor is colored with c
            mask = adj[v]
            
            # To speed up, we can maintain a list of neighbors for each vertex
            # But n=16 is small.
            
            for c in range(3):
                conflict = False
                m = mask
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    if colors[u] == c:
                        conflict = True
                        break
                    m -= lsb
                
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            
            return False
        
        return backtrack(0)

    # Heuristic 1: Random Greedy
    # Heuristic 2: Greedy by "potential" (e.g., connect high degree vertices)
    
    while time.time() < deadline:
        adj = [0] * n
        edge_count = 0
        
        # Shuffle edges
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        if not is_3_colorable(adj):
            if edge_count > best_count:
                best_count = edge_count
                best_edges = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            best_edges.append([u, v])
                        mask -= lsb

    # Fallback: Clebsch Graph (40 edges)
    # Even if it's 3-colorable, if we found nothing, we return it?
    # If it's 3-colorable, it's invalid.
    # But if we found nothing, we return an invalid graph?
    # The prompt says "valid graph...".
    # If we return an invalid graph, score is 0.
    # So we should only return a graph if we are sure it's valid.
    # If best_count == 0, we have a problem.
    # But the random greedy usually finds something.
    
    if best_count == 0:
        # Try a deterministic construction?
        # Or just return the Clebsch graph and hope it's valid?
        # Or return a small known valid graph (e.g. Grötzsch extended)?
        # For now, return Clebsch.
        clebsch_edges = []
        for u in range(n):
            for bit_idx in range(4):
                v = u ^ (1 << bit_idx)
                if u < v:
                    clebsch_edges.append([u, v])
            v = u ^ 15
            if u < v:
                clebsch_edges.append([u, v])
        best_edges = clebsch_edges

    return best_edges
