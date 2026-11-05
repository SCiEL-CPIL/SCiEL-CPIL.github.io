import time
import random
import numpy as np
import sys
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin for 150s limit
    
    n = 20
    num_colours = 20
    
    # Precompute all 4-cycles
    cycles = []
    vertices = list(range(n))
    for combo in combinations(vertices, 4):
        a, b, c, d = combo
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
    
    # Precompute for each edge (u,v) the list of 4-cycles containing it
    # For efficiency, we'll compute this on the fly but cache the structure
    # Actually, let's precompute: for each edge, which other edges form 4-cycles with it
    # A 4-cycle containing (u,v): u-v-w-x-u, needs edges (v,w), (w,x), (x,u)
    # So for edge (u,v), the relevant "paths" are: w adjacent to v, x adjacent to u, and (w,x) exists
    # We'll just iterate in can_colour but use neighbor lists
    
    def can_colour(adj_matrix, neighbors, new_edge_u, new_edge_v):
        """
        Checks if adding edge (u,v) allows for a valid colouring.
        neighbors: list of lists, neighbors[i] = list of (j, colour) for edges from i
        """
        u, v = new_edge_u, new_edge_v
        
        # 1. Proper edge colouring constraint:
        # Colour of (u,v) must be different from colours of all edges incident to u or v.
        forbidden_proper = set()
        for w, c in neighbors[u]:
            if w != v:
                forbidden_proper.add(c)
        for w, c in neighbors[v]:
            if w != u:
                forbidden_proper.add(c)
                
        # 2. 4-cycle rainbow constraint:
        # Find all 4-cycles containing edge (u,v): u-v-w-x-u
        # w must be neighbor of v, x must be neighbor of u, and (w,x) must exist
        forbidden_cycle = set()
        
        # Build a set of neighbors of u for quick lookup
        # Actually, let's just iterate
        for w, c_vw in neighbors[v]:
            if w == u:
                continue
            # For each w neighbor of v, find x that is neighbor of u and neighbor of w
            for x, c_xu in neighbors[u]:
                if x == v or x == w:
                    continue
                # Check if (w,x) exists
                if adj_matrix[w, x] == -1:
                    continue
                c_wx = adj_matrix[w, x]
                
                # Check if the 3 existing edges have distinct colours
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return None
                
                forbidden_cycle.add(c_vw)
                forbidden_cycle.add(c_wx)
                forbidden_cycle.add(c_xu)
                
        # Combine forbidden sets
        all_forbidden = forbidden_proper.union(forbidden_cycle)
        
        # Find a colour not in all_forbidden
        for c in range(num_colours):
            if c not in all_forbidden:
                return c
                
        return None

    def build_neighbors(adj):
        """Build neighbor lists from adjacency matrix."""
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] != -1:
                    c = adj[i, j]
                    neighbors[i].append((j, c))
                    neighbors[j].append((i, c))
        return neighbors

    # Strategy: multiple random restarts with greedy edge addition
    best_edges = 0
    best_adj = None
    
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
            
        # Initialize empty graph
        adj = np.full((n, n), -1, dtype=int)
        neighbors = [[] for _ in range(n)]
        
        # Create shuffled list of edges
        edges = all_edges[:]
        random.shuffle(edges)
        
        # Greedy Add - first pass
        for u, v in edges:
            c = can_colour(adj, neighbors, u, v)
            if c is not None:
                adj[u, v] = c
                adj[v, u] = c
                neighbors[u].append((v, c))
                neighbors[v].append((u, c))
                
        # Second pass: try to add skipped edges
        # Rebuild neighbors
        neighbors = build_neighbors(adj)
        for u, v in edges:
            if adj[u, v] != -1:
                continue
            c = can_colour(adj, neighbors, u, v)
            if c is not None:
                adj[u, v] = c
                adj[v, u] = c
                neighbors[u].append((v, c))
                neighbors[v].append((u, c))
                
        # Count edges
        edge_count = int(np.sum(adj != -1)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj.copy()

    # If best_adj is None, fallback
    if best_adj is None:
        adj = np.full((n, n), -1, dtype=int)
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
        best_adj = adj
        best_edges = 19

    # Final verification
    valid = True
    for cycle in cycles:
        e1, e2, e3, e4 = cycle
        c1 = best_adj[e1[0], e1[1]]
        c2 = best_adj[e2[0], e2[1]]
        c3 = best_adj[e3[0], e3[1]]
        c4 = best_adj[e4[0], e4[1]]
        
        if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
            continue
            
        if len({c1, c2, c3, c4}) < 4:
            valid = False
            break
            
    if not valid:
        adj = np.full((n, n), -1, dtype=int)
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
        best_adj = adj
        best_edges = 19

    # Format output
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] != -1:
                edges_list.append([u, v, int(best_adj[u, v])])
                
    return edges_list

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
    if res:
        print(f"Sample: {res[0]}")
