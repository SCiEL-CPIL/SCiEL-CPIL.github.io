import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    
    def is_c6_free(adj):
        # Check if graph has a C6
        # A C6 exists if there is a path of length 5 between u and v for some edge (u,v)?
        # No, we just need to know if the graph contains a C6.
        # This is hard to check directly without enumeration.
        # But we maintain the invariant that the graph is C6-free.
        # We only need to check if adding an edge creates a C6.
        pass

    def get_a5(adj):
        # Compute A^5
        # adj is boolean matrix
        a = adj.astype(np.int64)
        a2 = a @ a
        a4 = a2 @ a2
        a5 = a4 @ a
        return a5

    # Initial Construction
    # Try a few random starts
    best_edges = 0
    best_adj = None
    
    for trial in range(50):
        if time.time() > deadline - 5.0:
            break
            
        p = random.uniform(0.4, 0.7)
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i, j] = True
                    adj[j, i] = True
        
        # Greedily remove edges to make C6-free
        # This is expensive if done naively.
        # Better: Start with a known C6-free graph?
        # Or, just remove edges until no C6.
        # How to check if current graph has a C6?
        # We can check all 6-cycles. There are O(n^6) which is too slow.
        # Alternative: Use the property that if we build it up, we ensure C6-free.
        
        # Let's try a different init:
        # Start with empty graph, add edges greedily.
        adj = np.zeros((n, n), dtype=bool)
        edges = []
        
        # Shuffle vertices to break symmetry
        vertices = list(range(n))
        random.shuffle(vertices)
        
        for u in vertices:
            for v in vertices:
                if u >= v:
                    continue
                if time.time() > deadline - 5.0:
                    break
                # Check if adding (u, v) creates a C6
                # This requires checking if there is a path of length 5 between u and v
                # We can do this by BFS/DFS up to depth 5?
                # Or matrix power.
                # For init, maybe just use a heuristic:
                # If degree(u) + degree(v) is high, skip?
                # Let's use the matrix check. It's robust.
                
                # Temporarily add
                adj[u, v] = True
                adj[v, u] = True
                
                # Check if C6 created
                # Path of length 5 between u and v?
                # Actually, if we add (u,v), a C6 is formed if there was a path of length 5 in the graph BEFORE adding (u,v).
                # So we should check BEFORE adding.
                
                adj[u, v] = False
                adj[v, u] = False
                
                # Check path of length 5
                # Compute A^5
                a = adj.astype(np.int64)
                a2 = a @ a
                a4 = a2 @ a2
                a5 = a4 @ a
                
                if a5[u, v] == 0:
                    adj[u, v] = True
                    adj[v, u] = True
                    edges.append([u, v])
                    
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_adj = adj.copy()
            
    if best_adj is None:
        return []
        
    adj = best_adj
    edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i, j]]
    
    # Local Search
    # Current edge count
    current_edges = len(edges)
    
    # Precompute A^5
    a = adj.astype(np.int64)
    a2 = a @ a
    a4 = a2 @ a2
    a5 = a4 @ a
    
    # Helper to update A^5
    def update_a5():
        nonlocal a, a2, a4, a5
        a = adj.astype(np.int64)
        a2 = a @ a
        a4 = a2 @ a2
        a5 = a4 @ a

    stuck_count = 0
    
    while time.time() < deadline - 2.0:
        # Try to add an edge
        added = False
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i, j]:
                    if a5[i, j] == 0:
                        # Add edge
                        adj[i, j] = True
                        adj[j, i] = True
                        edges.append([i, j])
                        current_edges += 1
                        update_a5()
                        added = True
                        stuck_count = 0
                        break
            if added:
                break
        
        if not added:
            # Try a swap
            # Pick a random edge to remove
            if edges:
                idx = random.randint(0, len(edges)-1)
                u, v = edges[idx]
                
                # Remove
                adj[u, v] = False
                adj[v, u] = False
                edges.pop(idx)
                current_edges -= 1
                update_a5()
                
                # Try to add a different edge
                swap_success = False
                # Try random missing edges
                for _ in range(20):
                    ni = random.randint(0, n-1)
                    nj = random.randint(0, n-1)
                    if ni == nj:
                        continue
                    if ni > nj:
                        ni, nj = nj, ni
                    if not adj[ni, nj]:
                        if a5[ni, nj] == 0:
                            adj[ni, nj] = True
                            adj[nj, ni] = True
                            edges.append([ni, nj])
                            current_edges += 1
                            update_a5()
                            swap_success = True
                            stuck_count = 0
                            break
                
                if not swap_success:
                    # Revert
                    adj[u, v] = True
                    adj[v, u] = True
                    edges.append([u, v])
                    current_edges += 1
                    update_a5()
                    stuck_count += 1
            else:
                break
                
        if stuck_count > 100:
            # Restart local search with perturbation
            # Remove 5 random edges
            for _ in range(5):
                if edges:
                    idx = random.randint(0, len(edges)-1)
                    u, v = edges[idx]
                    adj[u, v] = False
                    adj[v, u] = False
                    edges.pop(idx)
                    current_edges -= 1
            update_a5()
            stuck_count = 0

    return edges

# EVOLVE-BLOCK-END
