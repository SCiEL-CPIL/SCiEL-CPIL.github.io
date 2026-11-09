"""Program for hexagon-free graphs on 40 vertices with many edges."""
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
        # Iterative DFS
        # Stack: (current_node, depth_remaining, visited_mask)
        # We want to reach v at depth 0.
        # So start at u, depth 5.
        
        stack = [(u, 5, 1 << u)]
        while stack:
            curr, depth, visited = stack.pop()
            if depth == 0:
                if curr == v:
                    return True
                continue
            if curr == v:
                continue
            
            # Get neighbors
            neighbors = adj[curr] & ~visited
            
            # Optimization: If depth == 1, we must go to v directly
            if depth == 1:
                if (neighbors >> v) & 1:
                    return True
                continue
            
            # If depth > 1, we cannot go to v yet
            neighbors &= ~(1 << v)
            
            while neighbors:
                lsb = neighbors & -neighbors
                next_v = lsb.bit_length() - 1
                neighbors ^= lsb
                stack.append((next_v, depth - 1, visited | lsb))
        return False

    # 1. Greedy Construction
    edges_to_try = all_edges[:]
    random.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if not creates_c6(u, v):
            add_edge(u, v)
            
    current_edges = sum(bin(a).count("1") for a in adj) // 2
    best_edges = current_edges
    best_adj = adj[:]
    
    # 2. Simulated Annealing
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    T = 10.0
    cooling = 0.9995
    min_T = 0.1
    
    # Track time of last improvement
    last_improvement_time = start_time
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Check for stagnation
        if time.time() - last_improvement_time > 5.0:
            # Restart from best
            adj = best_adj[:]
            current_edges = best_edges
            # Optional: reheat
            T = max(T, 5.0)
            last_improvement_time = time.time()

        # Perform a move
        move_type = random.random()
        
        if move_type < 0.6:
            # Try Add
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            if not edge_exists(u, v):
                if not creates_c6(u, v):
                    add_edge(u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        last_improvement_time = time.time()
        else:
            # Try Swap (Remove 1, Add 1)
            # Pick random edge to remove
            # To pick a random edge efficiently, we can iterate through adj?
            # Or just pick random u,v and check if edge exists.
            # With ~100 edges out of 780, probability of hitting an edge is ~13%.
            # This is low. Better to maintain a list of edges?
            # But list updates are costly.
            # Let's try picking random u, then random neighbor v.
            
            u = random.randint(0, n-1)
            neighbors_mask = adj[u]
            if neighbors_mask == 0:
                continue
            # Pick random set bit
            # random.getrandbits(40) & neighbors_mask ...
            # Simpler: convert to list? No, slow.
            # Use:
            rand_mask = random.getrandbits(40)
            v_mask = neighbors_mask & rand_mask
            if v_mask == 0:
                # Fallback: pick lowest bit
                v_mask = neighbors_mask & -neighbors_mask
            v = v_mask.bit_length() - 1
            
            if u == v: continue
            if u > v: u, v = v, u
            
            # Now (u,v) is an edge. Remove it.
            remove_edge(u, v)
            
            # Pick random non-edge to add
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y: continue
            if x > y: x, y = y, x
            
            if not edge_exists(x, y):
                if not creates_c6(x, y):
                    add_edge(x, y)
                    # current_edges unchanged
                    # If this allows future adds, it's good.
                    # We don't update best_edges, but we keep the state.
                    # If we get stuck, we might want to revert to best.
                else:
                    # Revert
                    add_edge(u, v)
            else:
                # Revert
                add_edge(u, v)
        
        T *= cooling
        
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
