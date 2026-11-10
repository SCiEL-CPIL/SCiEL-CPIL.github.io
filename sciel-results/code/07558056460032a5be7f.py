import time
import random
import math

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    time_limit = 140  # 10s margin for 150s limit
    
    # Helper to calculate number of C4s
    def count_c4(adj_mask):
        # adj_mask is a list of integers, where bit j is 1 if edge (i,j) exists
        # c_uw = popcount(adj_mask[u] & adj_mask[w])
        total = 0
        for u in range(n):
            for w in range(u + 1, n):
                c = (adj_mask[u] & adj_mask[w]).bit_count()
                if c >= 2:
                    total += c * (c - 1) // 2
        return total

    # Initialize with a random graph
    # To ensure connectivity and spread, maybe start with a random regular-ish graph
    # Or just random edges
    edges = set()
    # Generate random edges
    while len(edges) < target_edges:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    current_c4 = count_c4(adj_mask)
    best_c4 = current_c4
    best_edges = list(edges)
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    
    # Precompute list of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to remove
        edge_to_remove = random.choice(list(edges))
        u, v = edge_to_remove
        
        # Pick a random non-edge to add
        # Ensure it's not the same edge
        # And ensure we don't add an existing edge
        # We can just pick random u2, v2 and check
        added_edge = None
        attempts = 0
        while attempts < 100:
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            e2 = (min(u2, v2), max(u2, v2))
            if e2 not in edges:
                added_edge = e2
                break
            attempts += 1
        
        if added_edge is None:
            continue
            
        x, y = added_edge
        
        # Apply move
        edges.remove(edge_to_remove)
        adj_mask[u] ^= (1 << v)
        adj_mask[v] ^= (1 << u)
        
        edges.add(added_edge)
        adj_mask[x] |= (1 << y)
        adj_mask[y] |= (1 << x)
        
        new_c4 = count_c4(adj_mask)
        
        # Accept or reject
        if new_c4 < current_c4:
            current_c4 = new_c4
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_edges = list(edges)
        else:
            # Reject if worse, unless simulated annealing accepts
            delta = new_c4 - current_c4
            if random.random() < math.exp(-delta / T):
                current_c4 = new_c4
            else:
                # Revert
                edges.remove(added_edge)
                adj_mask[x] ^= (1 << y)
                adj_mask[y] ^= (1 << x)
                
                edges.add(edge_to_remove)
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    # Return best edges found
    return [[u, v] for u, v in best_edges]
