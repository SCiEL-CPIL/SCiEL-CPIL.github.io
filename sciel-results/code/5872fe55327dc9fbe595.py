import random
import time

def construct():
    deadline = time.time() + 115  # safety margin
    n = 16
    target_edges = 65
    
    def score_graph(adj_mask, deg):
        """Compute the max triangle degree-sum using bitmask adjacency.
        adj_mask[i] is an integer where bit j is set if edge (i,j) exists."""
        max_sum = 0
        # Iterate over all triangles u < v < w
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] >> v) & 1:
                    continue
                # Common neighbors of u and v
                common = adj_mask[u] & adj_mask[v]
                # Only consider w > v to avoid double counting
                common &= ~((1 << (v + 1)) - 1)
                while common:
                    w = common.bit_length() - 1
                    common ^= (1 << w)
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
        return max_sum

    def edges_to_mask(edges):
        """Convert edge list to bitmask representation and compute degrees."""
        adj_mask = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj_mask, deg

    def mask_to_edges(adj_mask):
        """Convert bitmask back to edge list."""
        edges = []
        for u in range(n):
            mask = adj_mask[u]
            for v in range(u + 1, n):
                if (mask >> v) & 1:
                    edges.append([u, v])
        return edges

    def local_search(adj_mask, deg, edges):
        """Perform greedy 1-swap local search using bitmask representation.
        Checks all possible swaps and takes the first one that improves the score."""
        current_score = score_graph(adj_mask, deg)
        
        while time.time() < deadline:
            improved = False
            
            # Generate list of non-edges and edges to iterate
            # This is faster than nested loops with condition checks inside
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj_mask[u] >> v) & 1:
                        non_edges.append((u, v))
            
            edges_list = list(edges) # Snapshot for iteration
            
            # Shuffle to add randomness to the order of checks, avoiding bias
            random.shuffle(non_edges)
            random.shuffle(edges_list)
            
            for u, v in non_edges:
                for x, y in edges_list:
                    # Skip if adding (u,v) is the same as removing (x,y)
                    if (u == x and v == y) or (u == y and v == x):
                        continue
                    
                    # Perform swap: Add (u,v), Remove (x,y)
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    adj_mask[x] &= ~(1 << y)
                    adj_mask[y] &= ~(1 << x)
                    deg[u] += 1
                    deg[v] += 1
                    deg[x] -= 1
                    deg[y] -= 1
                    
                    new_score = score_graph(adj_mask, deg)
                    
                    if new_score < current_score:
                        # Accept
                        current_score = new_score
                        improved = True
                        # Update edges list
                        # Remove (x,y)
                        for i, e in enumerate(edges):
                            if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                                edges.pop(i)
                                break
                        edges.append([u, v])
                        # Break all loops to restart scan from beginning
                        # This ensures we always start from a "fresh" state
                        break
                    else:
                        # Undo
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        adj_mask[x] |= (1 << y)
                        adj_mask[y] |= (1 << x)
                        deg[u] -= 1
                        deg[v] -= 1
                        deg[x] += 1
                        deg[y] += 1
                        
                if improved:
                    break
            
            if not improved:
                break
                
        return adj_mask, deg, edges, current_score

    def perturb(adj_mask, deg, edges, strength=5):
        """Apply random swaps to escape local optimum."""
        for _ in range(strength):
            # Pick random non-edge to add
            while True:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u < v and not (adj_mask[u] >> v) & 1:
                    break
            # Pick random edge to remove
            while True:
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x < y and (adj_mask[x] >> y) & 1:
                    break
            
            if (u == x and v == y) or (u == y and v == x):
                continue

            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[x] &= ~(1 << y)
            adj_mask[y] &= ~(1 << x)
            deg[u] += 1
            deg[v] += 1
            deg[x] -= 1
            deg[y] -= 1
            
            for i, e in enumerate(edges):
                if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                    edges.pop(i)
                    break
            edges.append([u, v])

    best_adj_mask = None
    best_deg = None
    best_edges = None
    best_score = float('inf')
    
    # Initial state: Start with K_{8,8} plus one intra-part edge
    init_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    adj_mask, deg = edges_to_mask(init_edges)
    edges = [e[:] for e in init_edges]
    
    # First local search
    adj_mask, deg, edges, score = local_search(adj_mask, deg, edges)
    
    if score < best_score:
        best_score = score
        best_adj_mask = adj_mask[:]
        best_deg = deg[:]
        best_edges = [e[:] for e in edges]
        
    # Iterated Local Search
    while time.time() < deadline:
        # Start from best known solution
        adj_curr = best_adj_mask[:]
        deg_curr = best_deg[:]
        edges_curr = [e[:] for e in best_edges]
        
        # Perturb
        perturb(adj_curr, deg_curr, edges_curr, strength=3)
        
        # Local search
        adj_out, deg_out, edges_out, score_out = local_search(adj_curr, deg_curr, edges_curr)
        
        if score_out < best_score:
            best_score = score_out
            best_adj_mask = adj_out[:]
            best_deg = deg_out[:]
            best_edges = [e[:] for e in edges_out]
            
        # If we are stuck at a high score for a long time, we might try a completely random restart
        # But K_{8,8} is a very good starting point.
        
    return {'n': 16, 'edges': best_edges}
