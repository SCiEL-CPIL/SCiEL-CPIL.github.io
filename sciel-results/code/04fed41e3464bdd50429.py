import time
import random

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Bitmask utilities
    def bit(i):
        return 1 << i

    # Build Clebsch graph as bitmasks
    # Clebsch: vertices 0..15. Edge if xor has popcount 1 or 4.
    adj = [0] * n
    edge_count = 0
    for u in range(n):
        for v in range(u+1, n):
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist == 1 or dist == 4:
                adj[u] |= bit(v)
                adj[v] |= bit(u)
                edge_count += 1
    
    # Verify Clebsch is triangle-free
    # (It is known to be, but good to assert)
    def is_triangle_free(adj):
        for u in range(n):
            v_mask = adj[u]
            while v_mask:
                v = (v_mask & -v_mask).bit_length() - 1
                v_mask &= v_mask - 1
                if adj[u] & adj[v]:
                    return False
        return True
        
    assert is_triangle_free(adj), "Clebsch should be triangle-free"
    
    best_adj = adj[:]
    best_count = edge_count
    
    def try_add_edges(adj, deadline):
        """
        Greedy edge addition to maximize edges while keeping triangle-free.
        Since we start with a 4-chromatic graph and only add triangle-free edges,
        the 4-chromatic property is preserved.
        """
        count = sum(bin(a).count('1') for a in adj) // 2
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle edges for randomness
            edges_to_try = all_edges[:]
            random.shuffle(edges_to_try)
            
            for u, v in edges_to_try:
                if time.time() > deadline:
                    break
                # Check if edge exists
                if adj[u] & bit(v):
                    continue
                # Check triangle-free: no common neighbor
                if adj[u] & adj[v]:
                    continue
                # Add edge
                adj[u] |= bit(v)
                adj[v] |= bit(u)
                count += 1
                changed = True
        return adj, count

    # Greedy improvement from Clebsch
    current_adj = adj[:]
    current_adj, current_count = try_add_edges(current_adj, deadline - 10)
    
    if current_count > best_count:
        best_adj = current_adj
        best_count = current_count
        
    # Random restarts / Local Search
    random.seed(42)
    
    for restart in range(1000): # Can do many more now
        if time.time() > deadline - 5:
            break
        
        # Start from best or perturbed best
        if restart == 0:
            current_adj = best_adj[:]
        else:
            current_adj = best_adj[:]
            # Remove random edges to allow different structures
            num_remove = random.randint(2, 10)
            removals = 0
            attempts = 0
            while removals < num_remove and attempts < 200:
                u = random.randint(0, n-1)
                v = random.randint(u+1, n-1)
                if current_adj[u] & bit(v):
                    current_adj[u] &= ~bit(v)
                    current_adj[v] &= ~bit(u)
                    removals += 1
                attempts += 1
        
        # Greedy add
        current_adj, current_count = try_add_edges(current_adj, deadline - 5)
        
        if current_count > best_count:
            best_adj = current_adj
            best_count = current_count
            
    # Final polish: try single edge additions
    if time.time() < deadline - 1:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for u, v in all_edges:
                if time.time() > deadline - 1:
                    break
                if best_adj[u] & bit(v):
                    continue
                if best_adj[u] & best_adj[v]:
                    continue
                # Try adding
                best_adj[u] |= bit(v)
                best_adj[v] |= bit(u)
                best_count += 1
                improved = True

    # Convert back to list of edges
    edges = []
    for u in range(n):
        v_mask = best_adj[u]
        while v_mask:
            v = (v_mask & -v_mask).bit_length() - 1
            v_mask &= v_mask - 1
            if u < v:
                edges.append([u, v])
                
    return edges
