import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 15
    # Initialize empty graph
    edges = set()
    
    # Helper to check validity
    def is_valid(edges):
        # Build adjacency list and degrees
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            
        # Check all paths of length 3
        # s - u - v - t
        # s != t
        # deg(s) != deg(t)
        
        # Optimization: Iterate over middle edges (u, v)
        # For each (u, v), find s in adj[u] \ {v} and t in adj[v] \ {u}
        # If s == t, skip (not simple path? s-u-v-t, s!=t required)
        # If deg[s] == deg[t], return False
        
        for u in range(n):
            for v in adj[u]:
                if u < v: # Undirected, process once? No, path direction matters for s,t
                    # Path s-u-v-t
                    # s is neighbor of u, not v
                    # t is neighbor of v, not u
                    pass
        
        # More efficient:
        # For each pair (u, v) in edges:
        #   For each s in adj[u] - {v}:
        #     For each t in adj[v] - {u}:
        #       If s != t and deg[s] == deg[t]: return False
        #       If s == t: continue (path s-u-v-s is a cycle, not a simple path with distinct endpoints? 
        #       "simple three-edge path" usually implies distinct vertices s,u,v,t.
        #       If s==t, it's a triangle. Endpoints are same vertex? No, s and t are endpoints.
        #       If s==t, deg[s]==deg[t] is trivially true.
        #       Does "simple path" allow s==t? No, simple path has distinct vertices.
        #       So s != t is required.
        
        for u in range(n):
            for v in adj[u]:
                # Only consider u < v to avoid duplicate processing? 
                # No, path s-u-v-t is different from s-v-u-t?
                # The set of paths is the same.
                # Let's just iterate all directed pairs.
                for s in adj[u]:
                    if s == v: continue
                    for t in adj[v]:
                        if t == u: continue
                        if s == t: continue
                        if deg[s] == deg[t]:
                            return False
        return True

    # Start with K7,8
    A = list(range(7))
    B = list(range(7, 15))
    for a in A:
        for b in B:
            edges.add((a, b))
    
    # Try to add intra edges
    # Candidate intra edges
    intra_candidates = []
    for i in range(7):
        for j in range(i+1, 7):
            intra_candidates.append((i, j))
    for i in range(7, 15):
        for j in range(i+1, 15):
            intra_candidates.append((i, j))
    
    random.shuffle(intra_candidates)
    
    current_edges = list(edges)
    
    # Greedy add
    for u, v in intra_candidates:
        if time.time() > deadline:
            break
        if (u, v) in edges or (v, u) in edges:
            continue
        
        new_edges = edges | {(u, v)}
        if is_valid(new_edges):
            edges = new_edges
        else:
            # Try to fix by removing some inter edges?
            # Too complex for now. Just skip.
            pass
            
    # Convert to list of lists
    result = []
    for u, v in edges:
        result.append([int(u), int(v)])
        
    return result
