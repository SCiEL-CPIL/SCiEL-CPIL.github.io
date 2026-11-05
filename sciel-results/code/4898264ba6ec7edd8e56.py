# EVOLVE-BLOCK-START
def construct():
    import random
    import numpy as np

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, n):
        """
        Returns a list of integers, where conflict_mask[i] has bit j set
        if edge i and edge j conflict.
        """
        m = len(edges)
        if m == 0:
            return []
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        conflict_masks = [0] * m
        for i in range(m):
            u, v = edges[i]
            # Blockers: u, v, N(u), N(v)
            blockers = {u, v}
            blockers.update(adj[u])
            blockers.update(adj[v])
            
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                x, y = edges[j]
                if x in blockers or y in blockers:
                    mask |= (1 << j)
            conflict_masks[i] = mask
        return conflict_masks

    def greedy_color(conflict_masks, order):
        """
        Greedy First-Fit using integer bitmasks.
        order: list of edge indices.
        """
        m = len(order)
        if m == 0:
            return []
            
        color_of_edge = [-1] * m
        color_masks = [] # list of ints
        
        for idx in order:
            assigned = -1
            for c in range(len(color_masks)):
                if (conflict_masks[idx] & color_masks[c]) == 0:
                    assigned = c
                    break
            if assigned == -1:
                assigned = len(color_masks)
                color_masks.append(0)
            color_of_edge[idx] = assigned
            color_masks[assigned] |= (1 << idx)
        return color_of_edge

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        conflict_masks = build_conflict_bitmasks(edges, n)
        
        # Heuristic 1: Degree in conflict graph (descending)
        degrees = [bin(mask).count('1') for mask in conflict_masks]
        order1 = sorted(range(m), key=lambda i: -degrees[i])
        
        # Heuristic 2: Random
        order2 = list(range(m))
        random.Random(seed + 1).shuffle(order2)
        
        # Heuristic 3: Conflict Density (same as degree here for this graph type? 
        # Actually, for strong coloring, the "degree" in the conflict graph is the number of conflicts.
        # Let's try a few random orders to be safe.
        
        best_colors = None
        best_count = float('inf')
        
        orders = [order1, order2]
        # Add a few more random orders
        for k in range(5):
            o = list(range(m))
            random.Random(seed + 100 + k).shuffle(o)
            orders.append(o)
            
        for order in orders:
            colors = greedy_color(conflict_masks, order)
            count = max(colors) + 1 if colors else 0
            if count < best_count:
                best_count = count
                best_colors = colors
                
        return best_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        colors = solve_instance(n, d, seed)
        out.append(colors)
    
    return out
# EVOLVE-BLOCK-END
