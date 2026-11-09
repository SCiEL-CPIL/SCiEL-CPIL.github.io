import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Heuristic: Greedy construction
    # Vertices 0..39
    
    for restart in range(1000): # Limit restarts by time
        if time.time() > deadline:
            break
        
        # Randomize edge order for diversity
        edges_list = [(u, v) for u in range(40) for v in range(u+1, 40)]
        random.shuffle(edges_list)
        
        adj = np.zeros((40, 40), dtype=int)
        current_edges = []
        
        for u, v in edges_list:
            if time.time() > deadline:
                break
            
            # Check if adding (u,v) creates a C6
            # This is true if there is a path of length 5 between u and v
            # We use a fast check: BFS/DFS depth 5
            if not creates_c6(adj, u, v):
                adj[u, v] = 1
                adj[v, u] = 1
                current_edges.append((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
            # Optional: Local search to improve
            # Try to add more edges by shuffling again?
            # Or try to swap?
            # For now, just keep the best greedy result
            
    return [list(e) for e in best_edges]

def creates_c6(adj, u, v):
    # Check if there is a simple path of length 5 from u to v
    # Path: u - a - b - c - d - v
    # We can do DFS from u, depth 0
    # If we reach v at depth 5, return True
    
    # Optimization: 
    # Neighbors of u: N1
    # Neighbors of N1 (not u): N2
    # ...
    
    # Since graph is small, simple recursion is fine
    # But Python recursion might be slow. Let's use iterative BFS/DFS.
    
    # State: (node, depth, visited_mask)
    # visited_mask is an int bitmask of 40 bits
    
    if adj[u, v] == 1:
        return False # u and v are already connected? No, we are checking if ADDING u-v creates C6.
                     # So u and v are NOT connected in adj currently.
                     # Wait, if u and v are not connected, we are checking if path of length 5 exists.
    
    # BFS
    # Level 0: {u}
    # Level 1: neighbors of u
    # ...
    # Level 5: check if v is in level 5
    
    # To ensure simple path, we must track visited nodes.
    # Number of states: 40 * 5 * 2^40 is too big.
    # But depth is only 5.
    # Number of paths of length 5 is small.
    
    # Let's just enumerate paths of length 5.
    # u -> n1 -> n2 -> n3 -> n4 -> v
    # n1 in adj[u]
    # n2 in adj[n1], n2 != u
    # n3 in adj[n2], n3 not in {u, n1}
    # n4 in adj[n3], n4 not in {u, n1, n2}
    # v in adj[n4], v not in {u, n1, n2, n3}
    
    # This is at most 40 * 40 * 40 * 40 * 40 = 100M operations worst case?
    # No, degrees are low.
    # If degree is 5, 5^5 = 3125. Very fast.
    
    for n1 in range(40):
        if adj[u, n1] == 0 or n1 == v: continue
        for n2 in range(40):
            if adj[n1, n2] == 0 or n2 == u or n2 == n1 or n2 == v: continue
            for n3 in range(40):
                if adj[n2, n3] == 0 or n3 in (u, n1, n2, v): continue
                for n4 in range(40):
                    if adj[n3, n4] == 0 or n4 in (u, n1, n2, n3, v): continue
                    if adj[n4, v] == 1:
                        return True
    return False
