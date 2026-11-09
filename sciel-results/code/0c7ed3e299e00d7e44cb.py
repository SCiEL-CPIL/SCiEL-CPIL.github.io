import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n = 40
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    random.seed(42)
    
    # Bitset representation of adjacency
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_exists(u, v):
        return (adj[u] >> v) & 1
        
    def creates_c6(u, v):
        """Check if adding edge (u,v) creates a C6.
        Equivalent to checking if there is a simple path of length 5 from u to v in current graph.
        """
        # We want path u -> a -> b -> c -> d -> v
        # u and v are not adjacent in the path context (they are the endpoints)
        # The path must be simple.
        
        # Use DFS with bitmasks for visited.
        # Depth 5 is small.
        
        def dfs(current, depth, visited):
            if depth == 0:
                return current == v
            if current == v:
                return False # Reached target too early, path not long enough
            
            # Neighbors of current, excluding visited
            neighbors = adj[current] & ~visited
            
            # If depth == 1, we must go directly to v
            if depth == 1:
                return (adj[current] >> v) & 1
            
            # Exclude v from neighbors if depth > 1, because we can't hit v early
            if (neighbors >> v) & 1:
                neighbors &= ~(1 << v)
                
            while neighbors:
                lsb = neighbors & -neighbors
                next_v = lsb.bit_length() - 1
                neighbors ^= lsb
                if dfs(next_v, depth - 1, visited | lsb):
                    return True
            return False
        
        return dfs(u, 5, 1 << u)

    # 1. Greedy Construction
    edges_to_try = all_edges[:]
    random.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if not creates_c6(u, v):
            add_edge(u, v)
            
    current_edges = sum(bin(a).count("1") for a in adj) // 2
    best_edges = current_edges
    best_adj = adj[:]
    
    # 2. Iterated Local Search (ILS)
    start_time = time.time()
    time_limit = 115.0
    
    def try_add_all():
        nonlocal current_edges
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not edge_exists(i, j):
                        if not creates_c6(i, j):
                            add_edge(i, j)
                            current_edges += 1
                            changed = True
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = adj[:]
                            
    try_add_all()
    
    while (time.time() - start_time) < time_limit:
        # Perturbation: Remove k random edges
        k = random.randint(1, 3) # Smaller perturbation
        removed = []
        attempts = 0
        while len(removed) < k and attempts < 50:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                attempts += 1
                continue
            if u > v:
                u, v = v, u
            if edge_exists(u, v):
                remove_edge(u, v)
                current_edges -= 1
                removed.append((u, v))
                attempts = 0
            else:
                attempts += 1
                
        if not removed:
            continue
            
        try_add_all()
        
    # Restore best graph
    adj = best_adj
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

# EVOLVE-BLOCK-END
