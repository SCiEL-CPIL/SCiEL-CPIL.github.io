import sys
import time

def construct():
    n = 12
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Precompute for fast C4 detection
    # For each pair of vertices (u, v), list of common neighbors
    # Or just check incrementally.
    
    colors = [-1] * num_edges
    
    # Map edge to index
    edge_idx = {}
    for k, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = k
    
    def get_edge_idx(i, j):
        if i > j:
            i, j = j, i
        return edge_idx[(i, j)]
        
    # For each color c, maintain list of edges
    color_edges = {c: [] for c in range(5)}
    
    # For fast C4 check:
    # When adding edge (u, v) with color c, check if there is a w such that (u, w) and (v, w) are in color c.
    # We can maintain for each color c, a set of edges.
    # And for each pair (u, v), the number of common neighbors in color c?
    # That's expensive to update.
    
    # Simpler: 
    # For each color c, maintain adjacency list.
    # When adding (u, v) to color c:
    #   common = neighbors_c[u] & neighbors_c[v]
    #   if len(common) >= 1: # Actually, if len(common) >= 1, we have a triangle? No, C4 needs 2 common neighbors.
    #   if len(common) >= 1, it forms a triangle. C4 needs 2.
    #   Wait, C4 is 4 vertices. u-v-w-x-u.
    #   Common neighbors of u and v are w. If there are 2 common neighbors w1, w2, then u-w1-v-w2-u is a C4.
    #   So we need to ensure no pair (u, v) has 2 common neighbors in any color.
    
    adj = {c: [set() for _ in range(n)] for c in range(5)}
    
    def is_valid_add(u, v, c):
        # Check if adding (u, v) to color c creates a C4.
        # This happens if there are already 2 common neighbors in color c.
        # Or if adding it creates a situation where some other pair has 2 common neighbors?
        # No, C4s involving (u, v) must use (u, v) as an edge.
        # The C4 is u-v-w-x-u.
        # Edges: (u,v), (v,w), (w,x), (x,u).
        # So w is a common neighbor of u and v? No.
        # w is connected to v. x is connected to u. w and x are connected.
        # This is not "common neighbors of u and v".
        # Common neighbors of u and v are z such that (u,z) and (v,z) exist.
        # If there are 2 such z, say z1, z2, then u-z1-v-z2-u is a C4.
        # So the condition "no two vertices with two common neighbors" is correct.
        
        # So, before adding (u, v) to color c, we must ensure that
        # the number of common neighbors of u and v in color c is < 2.
        # Actually, if it is 2, adding the edge doesn't create a C4 *using that edge*?
        # The C4 is formed by the common neighbors themselves?
        # No, the C4 is u-z1-v-z2-u. The edges are (u,z1), (z1,v), (v,z2), (z2,u).
        # These edges must all exist.
        # So if we already have (u,z1), (v,z1), (u,z2), (v,z2), then a C4 exists.
        # We must prevent this state.
        # So when adding (u, v), we check if there are already 2 common neighbors.
        # If there are 0 or 1, it's safe.
        # If there are 2, it's invalid.
        
        common = adj[c][u] & adj[c][v]
        return len(common) < 2

    def backtrack(k):
        if k == num_edges:
            return True
            
        u, v = edges[k]
        
        for c in range(5):
            if is_valid_add(u, v, c):
                # Add edge
                colors[k] = c
                color_edges[c].append(k)
                adj[c][u].add(v)
                adj[c][v].add(u)
                
                if backtrack(k + 1):
                    return True
                
                # Undo
                colors[k] = -1
                color_edges[c].pop()
                adj[c][u].remove(v)
                adj[c][v].remove(u)
                
        return False

    start_time = time.time()
    if backtrack(0) and time.time() - start_time < 110:
        return {'n': n, 'colors': colors}
    
    # Fallback to n=11 algebraic construction
    n = 11
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            # Map diff to color 0..4
            # Diffs: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            # Pairs: (1,10), (2,9), (3,8), (4,7), (5,6)
            # Color 0: 1, 10
            # Color 1: 2, 9
            # Color 2: 3, 8
            # Color 3: 4, 7
            # Color 4: 5, 6
            if diff in [1, 10]:
                colors.append(0)
            elif diff in [2, 9]:
                colors.append(1)
            elif diff in [3, 8]:
                colors.append(2)
            elif diff in [4, 7]:
                colors.append(3)
            else:
                colors.append(4)
                
    return {'n': n, 'colors': colors}

