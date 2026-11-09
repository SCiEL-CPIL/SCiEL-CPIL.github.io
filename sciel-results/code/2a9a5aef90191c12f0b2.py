import random
import time

def construct():
    # Grötzsch graph on vertices 0..10
    # Cycle 0-1-2-3-4-0
    # 5 connected to 1, 4 (neighbors of 0)
    # 6 connected to 2, 0 (neighbors of 1)
    # 7 connected to 3, 1 (neighbors of 2)
    # 8 connected to 4, 2 (neighbors of 3)
    # 9 connected to 0, 3 (neighbors of 4)
    # 10 connected to 5,6,7,8,9
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [0, 5], [1, 5],
        [1, 6], [2, 6],
        [2, 7], [3, 7],
        [3, 8], [4, 8],
        [4, 9], [0, 9],
        [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]
    ]
    
    n = 16
    # Bitset adjacency
    # adj[u] is a bitmask of neighbors of u
    
    def get_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj

    def is_triangle_free(adj, u, v):
        # Check if adding edge (u, v) creates a triangle
        # Triangle exists if there is w such that (u,w) and (v,w) are edges
        common = adj[u] & adj[v]
        return common == 0

    def count_edges(adj):
        # Sum of bits / 2
        return sum(bin(a).count('1') for a in adj) // 2

    best_edges = edges
    best_count = len(edges)
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Greedy addition of edges
    # We can also try to add edges between old vertices if not present
    all_possible = set()
    for i in range(n):
        for j in range(i+1, n):
            if [i, j] not in edges and [j, i] not in edges:
                all_possible.add((i, j))
                
    current_edges = edges.copy()
    adj = get_adj(current_edges)
    
    # To make it faster, we can just run a randomized greedy
    # Try to add all possible edges in random order
    # If it fails, restart
    
    while time.time() - start_time < time_limit:
        random.shuffle(list(all_possible)) # Need a list
        # Re-create list for shuffling
        poss_list = list(all_possible)
        random.shuffle(poss_list)
        
        trial_edges = current_edges.copy() # Start from best found? Or from base?
        # Starting from base Grötzsch might be safer to ensure validity
        # But we can start from best found and try to improve
        # Let's start from best found
        trial_edges = best_edges.copy()
        adj = get_adj(trial_edges)
        
        for u, v in poss_list:
            if (u, v) in trial_edges or (v, u) in trial_edges:
                continue
            if is_triangle_free(adj, u, v):
                trial_edges.append([u, v])
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                
        if len(trial_edges) > best_count:
            best_count = len(trial_edges)
            best_edges = trial_edges
            
    return best_edges
