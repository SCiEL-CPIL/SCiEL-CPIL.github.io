import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    # Bitmask representation
    # adj[i] is an integer where bit j is 1 if (i, j) is an edge
    
    def can_add_edge(adj, u, v):
        # Check if adding (u, v) creates a K3,3
        # u, v must be in opposite partitions
        # A = {u, x, y}, B = {v, z, w}
        # x, y in N(v) \ N(u)
        # z, w in N(u) \ N(v)
        
        nu = adj[u]
        nv = adj[v]
        
        # Candidates for x, y (part of A, connected to v, not u)
        # Must not be u or v
        cand_x_mask = nv & ~nu & ~(1 << u) & ~(1 << v)
        
        # Candidates for z, w (part of B, connected to u, not v)
        cand_z_mask = nu & ~nv & ~(1 << u) & ~(1 << v)
        
        # Extract lists from masks for iteration
        # n=20, so masks fit in int
        cand_x = [i for i in range(n) if (cand_x_mask >> i) & 1]
        cand_z = [i for i in range(n) if (cand_z_mask >> i) & 1]
        
        if len(cand_x) < 2 or len(cand_z) < 2:
            return True
            
        # Iterate pairs (x, y) from cand_x
        for i in range(len(cand_x)):
            x = cand_x[i]
            for j in range(i + 1, len(cand_x)):
                y = cand_x[j]
                
                # x, y must be non-adjacent
                if (adj[x] >> y) & 1:
                    continue
                    
                # We need z, w in cand_z such that:
                # 1. z, w non-adjacent
                # 2. x connected to z, w
                # 3. y connected to z, w
                
                # Intersection of neighbors of x and y within cand_z
                common_mask = (adj[x] & adj[y]) & cand_z_mask
                
                # Get list of candidates for z, w
                common_z = [k for k in cand_z if (common_mask >> k) & 1]
                
                if len(common_z) < 2:
                    continue
                    
                # Check if there exist two non-adjacent vertices in common_z
                for a in range(len(common_z)):
                    z = common_z[a]
                    for b in range(a + 1, len(common_z)):
                        w = common_z[b]
                        # z, w must be non-adjacent
                        if not ((adj[z] >> w) & 1):
                            # Found a K3,3
                            return False
                            
        return True

    adj = [0] * n
    edges = set()
    
    # Greedy construction with randomization
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if can_add_edge(adj, u, v):
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add((u, v))
            
    # Local Search: 1-2 exchange or simple addition
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add an edge
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in edges:
                    continue
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.add((u, v))
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try 1-2 exchange: remove one edge, add two
            # This is complex. Let's try a simple random restart or perturbation if stuck
            # For now, just break if no simple addition works, as 1-2 exchange is risky to implement correctly in short time
            # However, to maximize, we should try.
            # Given the constraints, let's just ensure we return the best found.
            pass

    result = [[u, v] for u, v in edges]
    return result
