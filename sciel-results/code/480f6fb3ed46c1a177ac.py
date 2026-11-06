import random
import time
import itertools

def is_k4_free(edges, n=18):
    # Build adjacency set
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check all 4-cliques
    # Optimized: iterate combinations
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        if b in adj[a] and c in adj[a] and d in adj[a] and \
           c in adj[b] and d in adj[b] and \
           d in adj[c]:
            return False
    return True

def max_triangle_free_induced(edges, n=18):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check sizes from n down to 1
    for size in range(n, 0, -1):
        for combo in itertools.combinations(range(n), size):
            is_tf = True
            # Check if any triangle in combo
            for i in range(len(combo)):
                for j in range(i+1, len(combo)):
                    for k in range(j+1, len(combo)):
                        u, v, w = combo[i], combo[j], combo[k]
                        if v in adj[u] and w in adj[u] and w in adj[v]:
                            is_tf = False
                            break
                    if not is_tf: break
                if not is_tf: break
            if is_tf:
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try to find a good H on 9 vertices
    best_h = None
    best_tf = 10 # Initialize high
    
    # Generate random maximal K4-free graphs on 9 vertices
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        # Start with empty graph
        h_edges = []
        adj = [set() for _ in range(9)]
        
        # Add random edges
        for u in range(9):
            for v in range(u+1, 9):
                if random.random() < 0.5:
                    # Check if adding (u,v) creates K4
                    # K4 must contain u,v and two common neighbors
                    common = adj[u] & adj[v]
                    creates_k4 = False
                    for w in common:
                        # If w is connected to x in common, and u-v added, 
                        # we need x to be connected to u and v? 
                        # Actually, if u-v is added, K4 is {u, v, w, x} 
                        # where w,x are common neighbors of u and v?
                        # No.
                        # If we add edge (u,v), a K4 is formed if there exist w, x 
                        # such that u-v, u-w, u-x, v-w, v-x, w-x are all edges.
                        # w, x must be in adj[u] and adj[v].
                        # And w-x must be an edge.
                        for x in common:
                            if x != w and x in adj[w]:
                                creates_k4 = True
                                break
                        if creates_k4:
                            break
                    if not creates_k4:
                        h_edges.append([u, v])
                        adj[u].add(v)
                        adj[v].add(u)
        
        # Calculate tf for this H
        tf_val = max_triangle_free_induced(h_edges, 9)
        if tf_val < best_tf:
            best_tf = tf_val
            best_h = h_edges
            if best_tf <= 3: # Can't get better than 3 for 9 vertices easily? 
                # Actually 3 is very good. 2 is even better.
                # But 4 is sufficient for score 8.
                break

    if best_h is None:
        # Fallback: use K6,6,6
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]

    # Construct G = H1 U H2
    # H1 vertices 0..8, H2 vertices 9..17
    g_edges = []
    for u, v in best_h:
        g_edges.append([u, v])
        g_edges.append([u+9, v+9])
        
    return g_edges
