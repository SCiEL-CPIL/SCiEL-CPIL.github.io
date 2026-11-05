import time
import random
import itertools
from functools import lru_cache

DEADLINE = 115.0

def has_c5_fast(adj_mask, n):
    """Check if graph contains a 5-cycle using bitmasks."""
    for v in range(n):
        neighbors_mask = adj_mask[v]
        # Iterate over pairs of neighbors
        neighbors = [u for u in range(n) if (neighbors_mask >> u) & 1]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a = neighbors[i]
                d = neighbors[j]
                # Need b adjacent to a, c adjacent to d, b adjacent to c
                # b != v, a, d; c != v, a, d, b
                b_candidates = [b for b in range(n) if b != v and b != a and b != d and (adj_mask[a] >> b) & 1]
                for b in b_candidates:
                    # c must be adjacent to d and b, and c != v, a, d, b
                    c_mask = adj_mask[d] & adj_mask[b]
                    c_mask &= ~((1 << v) | (1 << a) | (1 << d) | (1 << b))
                    if c_mask:
                        return True
    return False

def has_independent_set_fast(adj_mask, n, k=9):
    """Check for independent set of size k using backtracking with bitmasks."""
    # Use a recursive approach with pruning
    def backtrack(remaining_mask, chosen_count):
        if chosen_count >= k:
            return True
        if remaining_mask == 0:
            return False
        # Count bits in remaining_mask
        if bin(remaining_mask).count('1') < (k - chosen_count):
            return False
        
        # Pick vertex with minimum degree in remaining
        best_v = -1
        best_deg = n + 1
        for v in range(n):
            if (remaining_mask >> v) & 1:
                # Count neighbors in remaining
                neigh_in_rem = adj_mask[v] & remaining_mask
                deg = bin(neigh_in_rem).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
        
        if best_v == -1:
            return False
        
        # Branch 1: include best_v
        new_remaining = remaining_mask & ~(adj_mask[best_v] | (1 << best_v))
        if backtrack(new_remaining, chosen_count + 1):
            return True
        
        # Branch 2: exclude best_v
        remaining_mask &= ~(1 << best_v)
        if backtrack(remaining_mask, chosen_count):
            return True
        
        return False
    
    return backtrack((1 << n) - 1, 0)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Base construction: 8 disjoint K4s (n=32)
    n_base = 32
    adj_base_mask = [0] * n_base
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(4):
                if i != j:
                    adj_base_mask[base + i] |= (1 << (base + j))
    
    best_n = n_base
    best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    # Try to find n=33 by adding a 33rd vertex to the 8 K4 structure
    # The new vertex must connect to vertices in a way that:
    # 1. No C5 is created
    # 2. No independent set of size 9 is created
    
    random.seed(42)
    
    found_33 = False
    adj_33_mask = None
    
    search_deadline = start_time + 110.0
    
    # Strategy: Try different connection patterns for vertex 32
    # Vertex 32 connects to some subset of the 32 vertices in the K4s
    # To avoid C5, the connections must be carefully chosen
    
    # Key insight: If vertex 32 connects to vertices in at most 2 K4s, 
    # we can potentially avoid creating C5s while keeping alpha low
    
    # Try all subsets of size 4-8 from one or two K4s
    # This is still large, so use random sampling with bias
    
    while time.time() < search_deadline and not found_33:
        # Choose 1 or 2 K4s to connect to
        num_k4s = random.choice([1, 1, 2, 2, 2])
        chosen_k4s = random.sample(range(8), num_k4s)
        
        # For each chosen K4, choose a subset of vertices to connect to
        connections = set()
        for k4_idx in chosen_k4s:
            base = 4 * k4_idx
            # Choose 2-4 vertices from this K4
            num_conn = random.randint(2, 4)
            chosen_vertices = random.sample(range(base, base + 4), num_conn)
            connections.update(chosen_vertices)
        
        # Build adjacency mask for 33 vertices
        adj_try_mask = [0] * 33
        # Copy base K4 edges
        for i in range(32):
            adj_try_mask[i] = adj_base_mask[i]
        
        # Add edges from vertex 32 to chosen connections
        for v in connections:
            adj_try_mask[32] |= (1 << v)
            adj_try_mask[v] |= (1 << 32)
        
        # Check for C5
        if has_c5_fast(adj_try_mask, 33):
            continue
        
        # Check for independent set of size 9
        if has_independent_set_fast(adj_try_mask, 33, 9):
            continue
        
        # Found a valid graph!
        found_33 = True
        adj_33_mask = adj_try_mask
        break
    
    if found_33:
        best_n = 33
        best_edges = []
        for i in range(33):
            for j in range(i + 1, 33):
                if (adj_33_mask[i] >> j) & 1:
                    best_edges.append([i, j])
    else:
        # Fallback: try a more aggressive random search with faster C5 check
        found_33 = False
        while time.time() < search_deadline and not found_33:
            adj_try_mask = [0] * 33
            for i in range(33):
                for j in range(i + 1, 33):
                    if random.random() < 0.55:
                        adj_try_mask[i] |= (1 << j)
                        adj_try_mask[j] |= (1 << i)
            
            # Remove C5s by finding and removing edges
            for _ in range(500):
                if time.time() > search_deadline:
                    break
                if not has_c5_fast(adj_try_mask, 33):
                    break
                # Find a C5 and remove a random edge from it
                # This is expensive, so do it carefully
                # For now, just skip this graph if it has C5
                break
            
            if has_c5_fast(adj_try_mask, 33):
                continue
            
            if not has_independent_set_fast(adj_try_mask, 33, 9):
                found_33 = True
                adj_33_mask = adj_try_mask
                break
    
    if found_33:
        best_n = 33
        best_edges = []
        for i in range(33):
            for j in range(i + 1, 33):
                if (adj_33_mask[i] >> j) & 1:
                    best_edges.append([i, j])
    else:
        best_n = 32
        best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    return {'n': best_n, 'edges': best_edges}
