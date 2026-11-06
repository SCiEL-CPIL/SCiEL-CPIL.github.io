"""Rewritten program: greedy + backtracking C4-free 5-coloring of K_n, maximizing n."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin
    
    best_n = 0
    best_colors = None
    
    # Try from large n downward
    for n in [28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline - 10:
            break
        if n <= best_n:
            break
        
        result = _try_build(n, deadline)
        if result is not None:
            best_n = n
            best_colors = result
            break
    
    if best_n == 0:
        n = 11
        best_n = n
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    return {'n': best_n, 'colors': best_colors}


def _try_build(n, deadline):
    """Try to build a valid 5-coloring of K_n."""
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edge_list)
    
    for attempt in range(30):
        if time.time() > deadline:
            return None
        
        random.seed(attempt * 7919 + 13)
        
        # adj[c][v] = bitmask of neighbors of v in color c
        adj = [[0] * n for _ in range(5)]
        colors = [0] * num_edges
        edge_idx = [0]  # use list for mutability in nested function
        
        def can_add(u, v, c):
            new_u = adj[c][u] | (1 << v)
            new_v = adj[c][v] | (1 << u)
            
            # Check pair (u,v): at most 1 common neighbor
            common_uv = (new_u & new_v) & ~(1 << u) & ~(1 << v)
            if bin(common_uv).count('1') > 1:
                return False
            
            # Check pairs involving u or v with other vertices
            for w in range(n):
                if w == u or w == v:
                    continue
                # Pair (u,w)
                common_uw = (new_u & adj[c][w]) & ~(1 << u) & ~(1 << w)
                if bin(common_uw).count('1') > 1:
                    return False
                # Pair (v,w)
                common_vw = (new_v & adj[c][w]) & ~(1 << v) & ~(1 << w)
                if bin(common_vw).count('1') > 1:
                    return False
            
            return True
        
        def add_edge(u, v, c):
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            colors[edge_idx[0]] = c
            edge_idx[0] += 1
        
        def remove_edge(u, v, c):
            edge_idx[0] -= 1
            colors[edge_idx[0]] = 0
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
        
        node_limit = 50000
        nodes = [0]
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos >= num_edges:
                return True
            
            nodes[0] += 1
            if nodes[0] > node_limit:
                return False
            
            i, j = edge_list[pos]
            
            color_order = [0, 1, 2, 3, 4]
            random.shuffle(color_order)
            
            for c in color_order:
                if can_add(i, j, c):
                    add_edge(i, j, c)
                    if backtrack(pos + 1):
                        return True
                    remove_edge(i, j, c)
            
            return False
        
        if backtrack(0):
            return colors
    
    return None
