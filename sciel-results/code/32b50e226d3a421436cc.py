import random
import time
import numpy as np

def creates_c6_detailed(u, v, adj_masks):
    """
    Check if adding edge (u, v) creates a C6.
    Uses bitmask for speed.
    """
    n = len(adj_masks)
    # We need a path u-a-b-c-d-v
    # a in N(u), d in N(v)
    # b in N(a), c in N(b)
    # All distinct.
    
    # Optimization: Iterate over smaller neighborhood first?
    # Or just standard DFS.
    
    # To speed up, we can pre-compute intersections?
    # For now, simple DFS with backtracking.
    
    # Path: u -> a -> b -> c -> d -> v
    # a must be in N(u)
    # d must be in N(v)
    # b must be in N(a) \ {u}
    # c must be in N(b) \ {u, a}
    # d must be in N(c) \ {u, a, b} AND d in N(v)
    
    # Let's iterate a, then b, then c, then check d.
    
    mask_u = adj_masks[u]
    mask_v = adj_masks[v]
    
    # Get list of neighbors of u
    neighbors_u = [i for i in range(n) if (mask_u >> i) & 1]
    neighbors_v = [i for i in range(n) if (mask_v >> i) & 1]
    
    # If no neighbors, return False
    if not neighbors_u or not neighbors_v:
        return False
        
    for a in neighbors_u:
        if a == v: continue
        mask_a = adj_masks[a]
        # b in N(a) \ {u}
        # We can iterate b
        for b in range(n):
            if b == u or b == a: continue
            if not ((mask_a >> b) & 1): continue
            
            mask_b = adj_masks[b]
            # c in N(b) \ {u, a}
            for c in range(n):
                if c == u or c == a or c == b: continue
                if not ((mask_b >> c) & 1): continue
                
                mask_c = adj_masks[c]
                # d in N(c) \ {u, a, b}
                # AND d in N(v)
                # So d must be in N(c) & N(v)
                # And d != u, a, b
                # Let's find candidates for d
                # d_mask = mask_c & mask_v
                # Check if any d in d_mask is not u, a, b
                
                d_mask = mask_c & mask_v
                # Clear bits for u, a, b
                d_mask &= ~(1 << u)
                d_mask &= ~(1 << a)
                d_mask &= ~(1 << b)
                # Also d != c (guaranteed by simple graph, but c is not in N(c))
                
                if d_mask != 0:
                    return True
    return False

def construct():
    n = 40
    best_edges = []
    best_m = 0
    start_time = time.time()
    timeout = 115
    
    # Pre-generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_set = set(all_edges)
    
    for trial in range(100):
        if time.time() - start_time > timeout:
            break
        
        # Initialize with random regular graph
        # Degree 5 or 6
        d = random.choice([5, 6])
        
        # Generate random regular graph
        # Simple method: shuffle permutations
        adj = np.zeros((n, n), dtype=int)
        adj_masks = [0] * n
        
        # Create degree sequence
        # We just add random edges until we reach degree d?
        # Or use a proper generator.
        # For N=40, we can just add random edges and check degrees.
        # Or start with empty and add edges to balance degrees.
        
        # Let's try: Start with empty.
        # For each vertex, assign 'd' stubs.
        # Randomly pair stubs.
        stubs = []
        for i in range(n):
            stubs.extend([i] * d)
        
        random.shuffle(stubs)
        
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if adj[u, v] == 1:
                valid = False
                break
            adj[u, v] = 1
            adj[v, u] = 1
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
        if not valid:
            # Retry generation or just use a random graph of density ~d/2
            # For simplicity, if pairing fails, use random edges
            adj = np.zeros((n, n), dtype=int)
            adj_masks = [0] * n
            for u, v in random.sample(all_edges, int(n * d / 2)):
                adj[u, v] = 1
                adj[v, u] = 1
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)

        # Clean C6s
        # We need to find and remove edges from C6s.
        # Iterate until no C6s.
        # To find a C6, we can check all edges (u,v) and see if there is a path of length 5 in G\{(u,v)}.
        # This is expensive.
        # Alternative: Just try to add edges? No, we need to remove.
        
        # Let's use a heuristic:
        # While there is a C6:
        #   Pick a random edge (u,v).
        #   Check if removing it reduces the number of C6s?
        #   Or just check if it is part of a C6.
        
        # How to check if (u,v) is part of a C6?
        # There exists a path of length 5 between u and v in G \ {(u,v)}.
        # Let's compute A_temp = adj.copy()
        # A_temp[u,v] = 0; A_temp[v,u] = 0
        # Check if (A_temp^5)[u,v] > 0
        
        # This is O(N^3) per edge check. 40^3 * 105 edges ~ 2M ops. Fast.
        # But we do it in a loop.
        
        changed = True
        while changed:
            changed = False
            # Shuffle edges to randomize removal
            current_edges = [(u, v) for u in range(n) for v in range(u+1, n) if adj[u, v] == 1]
            random.shuffle(current_edges)
            
            for u, v in current_edges:
                if time.time() - start_time > timeout:
                    break
                
                # Check if (u,v) is in a C6
                # Create temporary adjacency matrix without (u,v)
                # Instead of full matrix mult, we can just check the condition.
                # But numpy matrix mult is fast.
                
                # Optimization: Only check if u and v have enough neighbors?
                # deg(u) >= 2 and deg(v) >= 2 is necessary.
                
                A_temp = adj.copy()
                A_temp[u, v] = 0
                A_temp[v, u] = 0
                
                # Compute A_temp^5
                # We can do this step by step or just power.
                # np.linalg.matrix_power is fine.
                A5 = np.linalg.matrix_power(A_temp, 5)
                
                if A5[u, v] > 0:
                    # There is a walk of length 5.
                    # Does it imply a simple path?
                    # In a simple graph, if there is a walk, there is a simple path?
                    # No, walk can repeat.
                    # But if we remove the edge, and there is a walk of length 5,
                    # it's highly likely there is a C6.
                    # To be safe, we should verify.
                    # But for heuristic, let's assume it's a C6 and remove.
                    # If we remove an edge that wasn't in a C6, we lose an edge unnecessarily.
                    # But we will try to add it back later.
                    
                    # Remove edge
                    adj[u, v] = 0
                    adj[v, u] = 0
                    adj_masks[u] &= ~(1 << v)
                    adj_masks[v] &= ~(1 << u)
                    changed = True
                    break # Restart the scan as graph changed
            
            if time.time() - start_time > timeout:
                break

        # Add edges
        # Repeatedly try to add edges
        improved = True
        while improved:
            improved = False
            # Compute A^5 for current graph
            A5 = np.linalg.matrix_power(adj, 5)
            
            # Candidates: non-edges
            candidates = [(u, v) for u in range(n) for v in range(u+1, n) if adj[u, v] == 0]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if time.time() - start_time > timeout:
                    break
                
                # Filter with A^5
                # If A5[u,v] == 0, no walk of length 5, so no C6.
                if A5[u, v] == 0:
                    # Safe to add
                    adj[u, v] = 1
                    adj[v, u] = 1
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    improved = True
                    # Restart or continue?
                    # If we continue, A5 is stale.
                    # We can break and restart the loop to recompute A5.
                    # Or we can just mark it and recompute at the end of the pass.
                    # Restarting is safer.
                    break
                
                else:
                    # A5[u,v] > 0, need detailed check
                    if not creates_c6_detailed(u, v, adj_masks):
                        adj[u, v] = 1
                        adj[v, u] = 1
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        improved = True
                        break
            
            if time.time() - start_time > timeout:
                break

        # Count edges
        m = int(np.sum(adj)) // 2
        if m > best_m:
            best_m = m
            best_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v] == 1:
                        best_edges.append([u, v])

    return best_edges
