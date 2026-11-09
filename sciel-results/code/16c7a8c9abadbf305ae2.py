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
    # adj[u] is an integer where bit v is 1 if (u,v) is an edge
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
        # DFS with bitmasks
        # visited: bitmask of visited vertices
        # current: current vertex
        # depth: remaining steps to reach v
        
        # We want path u -> ... -> v of length 5.
        # So we start at u, need 5 steps to reach v.
        
        def dfs(current, depth, visited):
            if depth == 0:
                return current == v
            if current == v:
                return False # Reached target too early
            
            # Get neighbors of current, excluding visited
            neighbors = adj[current] & ~visited
            
            # If depth == 1, we must go directly to v
            if depth == 1:
                return (adj[current] >> v) & 1
            
            # Otherwise, iterate neighbors
            # Optimization: if v is in neighbors, skip it (can't go through v now)
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
    # Shuffle edges
    edges_to_try = all_edges[:]
    random.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if not creates_c6(u, v):
            add_edge(u, v)
            
    current_edges = sum(bin(a).count("1") for a in adj) // 2
    best_edges = current_edges
    best_adj = adj[:]
    
    # 2. Simulated Annealing / Local Search
    # We want to maximize edges.
    # Moves:
    # - Add edge: +1 edge. Accept if valid.
    # - Swap edge: 0 edge change. Accept if valid. (Helps escape local minima)
    # - Remove edge: -1 edge. Accept with prob exp(-1/T). (Rarely useful, but helps if stuck)
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute non-edges for faster sampling? 
    # Or just sample random pairs.
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Perform a batch of moves
        for _ in range(100):
            move_type = random.random()
            
            if move_type < 0.5:
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
            elif move_type < 0.9:
                # Try Swap
                # Pick random existing edge to remove
                # Pick random non-existing edge to add
                # This is tricky to sample uniformly.
                # Let's pick random u,v for remove, random x,y for add.
                
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if not edge_exists(u, v): continue
                
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y: continue
                if x > y: x, y = y, x
                if edge_exists(x, y): continue
                
                # Check if swap is valid
                # Remove (u,v), check if adding (x,y) creates C6
                remove_edge(u, v)
                if not creates_c6(x, y):
                    add_edge(x, y)
                    # current_edges unchanged
                    # We don't update best_edges here because it's same count
                    # But we keep this state if it allows future adds
                else:
                    # Revert
                    remove_edge(x, y)
                    add_edge(u, v)
            else:
                # Try Remove (to escape)
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if edge_exists(u, v):
                    # Accept with probability exp(-1/T)
                    if random.random() < np.exp(-1.0 / T):
                        remove_edge(u, v)
                        current_edges -= 1
                        # Don't update best
                    else:
                        pass # Reject, do nothing
        
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
