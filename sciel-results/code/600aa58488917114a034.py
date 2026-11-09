import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Start with Paley graph on 17 vertices
    def make_paley(n):
        squares = {x * x % n for x in range(1, n) if n > 1}
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in squares:
                    edges.add((i, j))
        return edges

    edges = make_paley(17)
    n = 17
    
    # Precompute K3s and I5s for the current graph
    # Helper to get neighbors
    def get_neighbors(e):
        return set([v for (u, v) in edges if e in (u, v)])
    
    # We need to maintain lists of K3s and I5s to check constraints for new vertex
    # But updating them is complex.
    # Alternative: Just check the whole graph for K4 and I6 after each addition?
    # For n=17, checking is fast.
    # For n=35, checking is ~1.7M ops.
    # If we do this check, say, 50 times per added vertex, and add 18 vertices, total 900 checks.
    # 900 * 0.2s = 180s. Too slow.
    
    # Let's use the constraint satisfaction approach for the new vertex only.
    
    def find_k3s(edge_set, n):
        k3s = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in edge_set:
                    for k in range(j + 1, n):
                        if (i, k) in edge_set and (j, k) in edge_set:
                            k3s.append((i, j, k))
        return k3s

    def find_i5s(edge_set, n):
        i5s = []
        # I5 means no edges among the 5 vertices
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
                    for k in range(j + 1, n):
                        if (i, k) not in edge_set and (j, k) not in edge_set:
                            for l in range(k + 1, n):
                                if (i, l) not in edge_set and (j, l) not in edge_set and (k, l) not in edge_set:
                                    for m in range(l + 1, n):
                                        if (i, m) not in edge_set and (j, m) not in edge_set and (k, m) not in edge_set and (l, m) not in edge_set:
                                            i5s.append((i, j, k, l, m))
        return i5s

    # Initial check for n=17 (should be valid)
    # We assume Paley 17 is valid.
    
    k3s = find_k3s(edges, n)
    i5s = find_i5s(edges, n)
    
    best_n = n
    
    # Try to grow to n=35
    for target_n in range(n + 1, 36):
        if time.time() > deadline:
            break
        
        added = False
        max_attempts = 200  # Limit attempts per vertex to avoid getting stuck
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random connections for new vertex
            # Vertex index is target_n - 1
            new_v = target_n - 1
            new_edges = set()
            
            # Random probability
            p = 0.4
            for i in range(target_n - 1):
                if random.random() < p:
                    new_edges.add((i, new_v))
            
            # Check constraints against existing K3s and I5s
            valid = True
            
            # Check K3s: new_v must not be connected to all vertices of any K3
            for tri in k3s:
                u, v, w = tri
                if (u, new_v) in new_edges and (v, new_v) in new_edges and (w, new_v) in new_edges:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check I5s: new_v must be connected to at least one vertex of any I5
            for quads in i5s:
                # quads is actually a 5-tuple
                v1, v2, v3, v4, v5 = quads
                # Check if new_v is connected to ANY of them
                connected = False
                if (v1, new_v) in new_edges or (v2, new_v) in new_edges or (v3, new_v) in new_edges or (v4, new_v) in new_edges or (v5, new_v) in new_edges:
                    connected = True
                if not connected:
                    valid = False
                    break
            
            if valid:
                # Add the vertex and edges
                edges.update(new_edges)
                n = target_n
                added = True
                # Update K3s and I5s
                k3s = find_k3s(edges, n)
                i5s = find_i5s(edges, n)
                break
        
        if not added:
            # If we couldn't add this vertex, try next (lower) target? 
            # Or just break and return current best.
            # Since we want max n, and we failed on target_n, we might fail on target_n+1 too.
            # But maybe a different random seed would work?
            # For now, break and return what we have.
            break
            
    return {'n': n, 'edges': [[u, v] for u, v in edges]}
