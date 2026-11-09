import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    # Target n
    # We know n=46 requires delta >= 7.
    # n=47 requires delta >= 8.
    # Let's aim for n=46 first as it's easier.
    # If we find a valid graph for n=46, score is 46/39 > 1.
    # If we find n=47, score is 47/39.
    
    best_n = 39
    best_edges = []
    
    for target_n in [46, 47]:
        if time.time() > deadline:
            break
            
        # Initialize with Erdos-Renyi graph G(7) subgraph
        # G(7) vertices: 0..48
        # We take vertices 0..target_n-1
        
        q = 7
        n = target_n
        # Adjacency matrix
        adj = np.zeros((n, n), dtype=bool)
        
        # Build G(7) edges
        # (x1, y1) ~ (x2, y2) if y1 - y2 = (x1 - x2)^2
        # Map vertex i to (i % 7, i // 7)
        
        for u in range(n):
            x1, y1 = u % 7, u // 7
            for v in range(u + 1, n):
                x2, y2 = v % 7, v // 7
                if x1 == x2:
                    continue
                # Check if y1 - y2 == (x1 - x2)^2
                # Note: y1, y2 are in 0..6
                # (x1 - x2)^2 mod 7
                if (y1 - y2) % 7 == ((x1 - x2) * (x1 - x2)) % 7:
                    adj[u, v] = True
                    adj[v, u] = True
        
        # Now we have a C4-free graph on n vertices.
        # Current min degree is likely low (around 4-6).
        # We need min degree >= n - 39.
        # For n=46, need delta >= 7.
        # For n=47, need delta >= 8.
        
        min_deg_needed = n - 39
        
        # Local Search: Add edges
        # We maintain a set of forbidden pairs (would create C4)
        # A pair (u, v) is forbidden if |N(u) ∩ N(v)| > 0
        
        def get_degree(u):
            return np.sum(adj[u])
            
        def can_add_edge(u, v):
            # Check if N(u) and N(v) are disjoint
            # Using bitsets for speed
            # n is small (46), so boolean array intersection is fast
            return not np.any(adj[u] & adj[v])
            
        # Precompute neighbor bitsets?
        # n=46 fits in a Python int or numpy uint64.
        # Let's use Python ints for bitsets for speed.
        
        def build_bitsets():
            bits = [0] * n
            for i in range(n):
                mask = 0
                for j in range(n):
                    if adj[i, j]:
                        mask |= (1 << j)
                bits[i] = mask
            return bits
            
        bits = build_bitsets()
        
        def update_bit(i, j):
            bits[i] |= (1 << j)
            bits[j] |= (1 << i)
            
        def remove_bit(i, j):
            bits[i] &= ~(1 << j)
            bits[j] &= ~(1 << i)
            
        def can_add(u, v):
            return (bits[u] & bits[v]) == 0
            
        # Greedy Add
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            # Find vertex with min degree
            degs = [np.sum(adj[i]) for i in range(n)]
            min_deg = min(degs)
            if min_deg >= min_deg_needed:
                break
            
            # Find all pairs (u, v) that can be added
            # Prioritize pairs where min(deg[u], deg[v]) is high?
            # Or where adding increases the min degree of the graph?
            
            # Simple strategy: iterate through vertices with low degree
            # and try to connect them to non-neighbors that don't form C4
            
            # Sort vertices by degree
            verts = sorted(range(n), key=lambda x: degs[x])
            
            for u in verts:
                if degs[u] >= min_deg_needed:
                    continue
                # Try to find a v to connect to u
                # v should not be neighbor of u
                # v should not share neighbors with u
                # v should ideally have low degree to help balance?
                # Or high degree?
                # If we connect u to v, deg[u] and deg[v] increase.
                
                # Candidate v: not in N(u), and (bits[u] & bits[v]) == 0
                # We want to pick v such that we don't create new conflicts for others?
                # Hard to predict.
                
                # Heuristic: pick v with lowest degree among valid candidates
                # to keep the distribution flat.
                
                best_v = -1
                best_v_deg = float('inf')
                
                for v in range(n):
                    if v == u or adj[u, v]:
                        continue
                    if can_add(u, v):
                        if degs[v] < best_v_deg:
                            best_v = v
                            best_v_deg = degs[v]
                            
                if best_v != -1:
                    # Add edge (u, best_v)
                    adj[u, best_v] = True
                    adj[best_v, u] = True
                    update_bit(u, best_v)
                    degs[u] += 1
                    degs[best_v] += 1
                    added_any = True
                    # Break inner loop? Or continue to add more?
                    # Continuing might add many edges at once.
                    # Let's break to re-evaluate degrees.
                    break
        
        # Check if we met the condition
        final_degs = [np.sum(adj[i]) for i in range(n)]
        if min(final_degs) >= min_deg_needed:
            if n > best_n:
                best_n = n
                # Extract edges
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j]:
                            edges.append([i, j])
                best_edges = edges
                # If we found n=47, we can stop?
                # If n=46, we might try n=47?
                # The loop iterates [46, 47].
                # If 46 works, we save it. Then try 47.
                # If 47 works, we overwrite.
        
    # Return the best found
    return {'n': best_n, 'edges': best_edges}
