import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 16
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Fast 3-colorability check using backtracking with bitmasks and degree ordering
    def is_3_colorable(adj):
        # Compute degrees and order vertices by decreasing degree
        degrees = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
        pos = {v: i for i, v in enumerate(order)}
        
        # colors[v] = -1 (uncolored), 0, 1, 2
        colors = [-1] * n
        
        # Precompute for each vertex, the list of neighbors
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
        
        # Track which colors are used by colored neighbors for each vertex
        # color_mask[v] has bit c set if some neighbor of v has color c
        color_mask = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Available colors are those not in color_mask[v]
            available = 0b111 & ~color_mask[v]
            
            # Try each available color
            while available:
                lsb = available & -available
                c = lsb.bit_length() - 1
                available -= lsb
                
                # Assign color c to v
                colors[v] = c
                # Update color_mask for all neighbors of v
                updates = []
                for u in neighbors[v]:
                    if colors[u] == -1:  # Only uncolored neighbors need update
                        old = color_mask[u]
                        new = old | (1 << c)
                        if new != old:
                            color_mask[u] = new
                            updates.append((u, old))
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                for u, old in updates:
                    color_mask[u] = old
                colors[v] = -1
            
            return False
        
        return backtrack(0)
    
    # Local search: try to improve a graph by swapping edges
    # Given a triangle-free graph, try removing an edge and adding another to increase density
    # while maintaining triangle-freeness and non-3-colorability
    def local_improve(adj, edge_count, time_limit=0.5):
        start = time.time()
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            # Try each edge: remove it, see if we can add a better one
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        # Try removing (u,v)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        
                        # Try adding an edge that was previously blocked
                        # Find edges that were blocked by (u,v)
                        for w in range(n):
                            if w != u and w != v and (adj[w] & adj[u] & adj[v]) == 0:
                                # Check if (u,w) or (v,w) can be added
                                for a, b in [(u, w), (v, w)]:
                                    if a < b and (adj[a] & adj[b]) == 0:
                                        # Check if this creates a better graph
                                        adj[a] |= (1 << b)
                                        adj[b] |= (1 << a)
                                        if not is_3_colorable(adj):
                                            improved = True
                                            break
                                        adj[a] &= ~(1 << b)
                                        adj[b] &= ~(1 << a)
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        
                        if not improved:
                            # Restore (u,v)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        
                        if improved:
                            break
                if improved:
                    break
        return adj, improved
    
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
            adj_improved, _ = local_improve(adj, edge_count, time_limit=0.3)
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
        elif time.time() > deadline - 5:
            # If we're running out of time and haven't found a good graph,
            # try a few more quick trials
            pass

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
