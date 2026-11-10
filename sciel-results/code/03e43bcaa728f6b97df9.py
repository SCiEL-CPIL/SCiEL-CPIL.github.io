import numpy as np
import time
import random

def construct():
    n = 40
    start_time = time.time()
    timeout = 110
    
    # Initialize
    adj = np.zeros((n, n), dtype=np.int8)
    neighbors = [[] for _ in range(n)]
    adj_masks = [0] * n  # Bitmask of neighbors
    
    edges = []
    
    # Start with 8 disjoint copies of K5
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u][v] = 1
                adj[v][u] = 1
                neighbors[u].append(v)
                neighbors[v].append(u)
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.append([u, v])
                
    all_possible = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 0:
                all_possible.append((i, j))
                
    random.seed(42)
    random.shuffle(all_possible)
    
    degrees = [len(neighbors[i]) for i in range(n)]
    
    def has_c6(u, v):
        """
        Check if adding edge (u,v) creates a C6.
        C6: u - x1 - x2 - x3 - x4 - v
        """
        # x1 in N(u), x4 in N(v)
        # x2 in N(x1), x3 in N(x4)
        # (x2, x3) in E
        # x1, x2, x3, x4 distinct and not u, v
        
        # Optimize: Precompute masks for N(x4) inside the loop?
        # Or just use the bitmask check for the final intersection.
        
        # To avoid re-computing masks for x4, we can just iterate.
        # But bitmask intersection is O(1).
        
        for x1 in neighbors[u]:
            if x1 == v: continue
            for x4 in neighbors[v]:
                if x4 == u: continue
                if x1 == x4: continue
                
                # We need x2 in N(x1) and x3 in N(x4) such that x2-x3 is an edge.
                # x2, x3 not in {u, v, x1, x4}
                
                # Get mask of valid candidates for x3 from N(x4)
                # mask_x4 = adj_masks[x4]
                # We want to intersect with N(x2) for some x2 in N(x1).
                
                # Iterate x2
                for x2 in neighbors[x1]:
                    if x2 == u or x2 == v or x2 == x1 or x2 == x4:
                        continue
                    
                    # Check if any neighbor of x2 is in N(x4) and valid
                    # valid x3: in N(x4), not in {u, v, x1, x4, x2}
                    
                    # mask_x4 = adj_masks[x4]
                    # mask_x2 = adj_masks[x2]
                    # intersection = mask_x4 & mask_x2
                    # We need to clear bits for u, v, x1, x4, x2 from intersection
                    # if intersection != 0, then C6 exists.
                    
                    intersection = adj_masks[x4] & adj_masks[x2]
                    # Clear forbidden bits
                    # Forbidden: u, v, x1, x4, x2
                    # Note: x2 is in N(x1), so x2 is a neighbor of x1.
                    # x4 is a neighbor of v.
                    
                    # We can just check if there is any bit set in intersection
                    # that is not u, v, x1, x4, x2.
                    
                    # To be safe and fast, let's just loop if intersection is non-zero?
                    # Or clear bits.
                    # Since n=40, we can clear bits easily.
                    
                    # Create a mask of forbidden nodes
                    # forbidden = (1 << u) | (1 << v) | (1 << x1) | (1 << x4) | (1 << x2)
                    # if (intersection & ~forbidden): return True
                    
                    # This is efficient.
                    forbidden = (1 << u) | (1 << v) | (1 << x1) | (1 << x4) | (1 << x2)
                    if intersection & ~forbidden:
                        return True
                        
        return False

    while all_possible:
        if time.time() - start_time > timeout:
            break
        
        sample_size = min(len(all_possible), 500) # Check more candidates
        candidates = random.sample(all_possible, sample_size)
        
        best_edge = None
        best_score = -1e9
        
        for (u, v) in candidates:
            if has_c6(u, v):
                continue
            
            score = -(degrees[u] + degrees[v])
            # Optional: penalize common neighbors to reduce triangles/cliques
            # common = np.sum(adj[u] & adj[v]) # This is O(n)
            # score -= common * 0.5
            
            if score > best_score:
                best_score = score
                best_edge = (u, v)
                
        if best_edge is None:
            # If no edge in sample is valid, try to find one in the rest?
            # Or just break if we can't find any in a large sample.
            # Let's try a smaller sample or break.
            # For now, break to avoid infinite loop if graph is saturated.
            # But we might have valid edges in the unsampled part.
            # Let's do a linear scan of the remaining all_possible if sample fails?
            # That could be slow.
            # Let's just break.
            break
            
        u, v = best_edge
        adj[u][v] = 1
        adj[v][u] = 1
        neighbors[u].append(v)
        neighbors[v].append(u)
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        degrees[u] += 1
        degrees[v] += 1
        edges.append([u, v])
        
        # Remove from all_possible
        # all_possible.remove(best_edge) is O(N).
        # Better to maintain a set or just filter.
        # Since N is ~800, O(N) is fine.
        try:
            all_possible.remove(best_edge)
        except ValueError:
            pass

    return edges
