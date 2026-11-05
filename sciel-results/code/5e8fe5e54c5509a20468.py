import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 16
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Fast 3-colorability check using backtracking with bitmasks
    def is_3_colorable(adj):
        # Compute degrees and order vertices by decreasing degree
        degrees = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
        
        # colors[v] = -1 (uncolored), 0, 1, 2
        colors = [-1] * n
        color_mask = [0] * n  # bit c set if neighbor has color c
        
        # Precompute neighbors
        neighbors = []
        for v in range(n):
            mask = adj[v]
            nbrs = []
            while mask:
                lsb = mask & -mask
                u = lsb.bit_length() - 1
                nbrs.append(u)
                mask -= lsb
            neighbors.append(nbrs)
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            available = 0b111 & ~color_mask[v]
            
            # Try each available color
            while available:
                lsb = available & -available
                c = lsb.bit_length() - 1
                available -= lsb
                
                colors[v] = c
                updates = []
                for u in neighbors[v]:
                    if colors[u] == -1:
                        old = color_mask[u]
                        new = old | (1 << c)
                        if new != old:
                            color_mask[u] = new
                            updates.append((u, old))
                
                if backtrack(idx + 1):
                    return True
                
                for u, old in updates:
                    color_mask[u] = old
                colors[v] = -1
            
            return False
        
        return backtrack(0)
    
    # Local search: try to add edges to improve density
    def local_improve(adj, edge_count, time_limit=0.5):
        start = time.time()
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            # Try adding each non-edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        # Check if adding (u,v) creates a triangle
                        if (adj[u] & adj[v]) == 0:
                            # Try adding the edge
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            
                            if not is_3_colorable(adj):
                                improved = True
                                break
                            else:
                                # Revert
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                        if improved:
                            break
                if improved:
                    break
        return adj
    
    # Randomized Greedy Construction with local search
    while time.time() < deadline:
        # Reset adjacency
        adj = [0] * n
        edge_count = 0
        
        # Shuffle edges for this trial
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            # Check if edge (u, v) can be added
            if (adj[u] & adj[v]) == 0:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        # Check if this maximal triangle-free graph is 3-colorable
        if not is_3_colorable(adj):
            # Try local search to improve
            adj_improved = local_improve(adj, edge_count, time_limit=0.2)
            # Count edges in improved graph
            improved_count = sum(bin(adj_improved[v]).count('1') for v in range(n)) // 2
            
            if improved_count > best_count:
                best_count = improved_count
                # Reconstruct edge list from adj_improved
                best_edges = []
                for u in range(n):
                    mask = adj_improved[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            best_edges.append([u, v])
                        mask -= lsb
        elif time.time() > deadline - 10:
            # If we're running out of time and haven't found a good graph,
            # break to return what we have
            break

    # Fallback: If no 4-chromatic graph was found, return the Clebsch graph (40 edges)
    if best_count == 0:
        clebsch_edges = []
        for u in range(n):
            for bit_idx in range(4):
                v = u ^ (1 << bit_idx)
                if u < v:
                    clebsch_edges.append([u, v])
            v = u ^ 15
            if u < v:
                clebsch_edges.append([u, v])
        best_edges = clebsch_edges

    return best_edges
