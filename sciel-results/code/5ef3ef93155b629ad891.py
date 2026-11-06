"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _verify_triangle_free(adj, n):
    """Check that the graph is triangle-free using adjacency bitmasks."""
    for u in range(n):
        mask_u = adj[u]
        while mask_u:
            v_bit = mask_u & -mask_u
            v = v_bit.bit_length() - 1
            rest = mask_u & ~(v_bit)
            rest2 = rest
            while rest2:
                w_bit = rest2 & -rest2
                w = w_bit.bit_length() - 1
                if adj[v] & (1 << w):
                    return False
                rest2 = rest2 & (rest2 - 1)
            mask_u = mask_u & (mask_u - 1)
    return True

def _max_independent_set_size(adj, n, deadline):
    """Find max independent set size using branch and bound with bitmasks."""
    best = 0
    import sys
    sys.setrecursionlimit(10000)
    
    def expand(chosen_mask, candidates, depth, time_left):
        nonlocal best
        if time.time() > deadline:
            return
        # Upper bound: depth + popcount(candidates)
        if depth + bin(candidates).count('1') <= best:
            return
        if candidates == 0:
            if depth > best:
                best = depth
            return
        
        # Choose a vertex to branch on - pick one with high degree in candidate set
        # for better pruning
        best_v = -1
        best_deg = -1
        temp = candidates
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            deg = bin(candidates & adj[v]).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp = temp & (temp - 1)
        
        v = best_v
        v_bit = 1 << v
        
        # Option 1: include v (try first for better pruning)
        new_candidates = candidates & ~adj[v] & ~(1 << v)
        expand(chosen_mask | v_bit, new_candidates, depth + 1, time_left)
        
        # Option 2: exclude v
        expand(chosen_mask, candidates ^ v_bit, depth, time_left)
    
    # Initial candidates: all vertices
    all_vertices = (1 << n) - 1
    expand(0, all_vertices, 0, time.time() + (deadline - time.time()) * 0.5)
    
    return best

def _is_valid_graph(adj, n, deadline):
    """Check if graph is triangle-free and has alpha < 10."""
    if not _verify_triangle_free(adj, n):
        return False
    alpha = _max_independent_set_size(adj, n, deadline)
    return alpha < 10

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with Andrasfai graph And(9), n=26
    n = 26
    adj = [0] * n
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
    
    best_n = n
    best_edges = edges[:]
    
    # Try to extend by adding vertices one at a time
    current_n = n
    current_adj = adj[:]
    
    while current_n < 41 and time.time() < deadline:
        # Try adding a new vertex
        new_v = current_n
        new_adj = current_adj + [0]
        
        # The new vertex must not form a triangle with any existing edge
        # and must not create an independent set of size 10
        
        # Strategy: try all possible connections for the new vertex
        # that maintain triangle-freeness
        
        # For triangle-freeness: if we connect new_v to u and v, then u and v must not be adjacent
        # So the neighbors of new_v must form an independent set in the existing graph
        
        # Let's try random subsets of vertices to connect to
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Randomly select neighbors for new vertex
            # Must form an independent set in current graph
            candidates = list(range(current_n))
            random.shuffle(candidates)
            
            # Try to build an independent set for neighbors
            neighbors = []
            for v in candidates:
                # Check if v is non-adjacent to all current neighbors
                compatible = True
                for nb in neighbors:
                    if current_adj[v] & (1 << nb):
                        compatible = False
                        break
                if compatible:
                    neighbors.append(v)
            
            # Try this connection pattern
            new_adj = current_adj + [0]
            for nb in neighbors:
                new_adj[new_v] |= (1 << nb)
                new_adj[nb] |= (1 << new_v)
            
            # Check if valid
            if _is_valid_graph(new_adj, current_n + 1, deadline):
                current_n += 1
                current_adj = new_adj
                best_n = current_n
                # Update edges
                best_edges = []
                for i in range(current_n):
                    for j in range(i + 1, current_n):
                        if current_adj[i] & (1 << j):
                            best_edges.append([i, j])
                found = True
                break
        
        if not found:
            break
    
    # Verify the best graph found
    if best_n > 0:
        # Reconstruct adj from best_edges for final verification
        final_adj = [0] * best_n
        for u, v in best_edges:
            final_adj[u] |= (1 << v)
            final_adj[v] |= (1 << u)
        
        if _is_valid_graph(final_adj, best_n, deadline):
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback to original Andrasfai graph
    return {'n': 26, 'edges': edges}
# EVOLVE-BLOCK-END
