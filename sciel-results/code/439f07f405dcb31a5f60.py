"""Greedy + backtracking C4-free 5-coloring of complete graphs."""
import time
import random

def construct():
    deadline = time.time() + 140
    
    # Start with known n=11 circular construction
    n0 = 11
    colors0 = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    best_n = n0
    best_colors = colors0
    
    for attempt in range(1000):
        if time.time() > deadline - 5:
            break
        
        target = 12 + (attempt % 18)
        result = greedy_build(target, attempt)
        if result is not None:
            n, colors = result
            if n > best_n:
                best_n = n
                best_colors = colors
    
    return {'n': best_n, 'colors': best_colors}


def greedy_build(target_n, seed):
    """Build C4-free 5-coloring of K_target_n using greedy + backtracking."""
    rng = random.Random(seed)
    
    # colors[(i,j)] = color for i<j
    colors = {}
    # adj[v][c] = set of neighbors of vertex v in color c
    adj = [None] * target_n
    
    def can_add(v, u, c):
        """Can we add edge (v,u) in color c without creating a C4?"""
        # C4 exists if v and u have >=2 common neighbors in color c
        common = adj[u][c] & adj[v][c]
        return len(common) < 2
    
    def add_edge(v, u, c):
        key = (v, u) if v < u else (u, v)
        colors[key] = c
        adj[v][c].add(u)
        adj[u][c].add(v)
    
    def remove_edge(v, u, c):
        key = (v, u) if v < u else (u, v)
        del colors[key]
        adj[v][c].discard(u)
        adj[u][c].discard(v)
    
    # Initialize vertex 0
    adj[0] = [set() for _ in range(5)]
    
    for v in range(1, target_n):
        adj[v] = [set() for _ in range(5)]
        
        # Assign colors to edges (v, 0), (v, 1), ..., (v, v-1)
        order = list(range(v))
        rng.shuffle(order)
        
        # Backtracking search
        def backtrack(idx):
            if idx == v:
                return True
            
            u = order[idx]
            # Try colors in random order
            color_order = list(range(5))
            rng.shuffle(color_order)
            
            for c in color_order:
                if can_add(v, u, c):
                    add_edge(v, u, c)
                    if backtrack(idx + 1):
                        return True
                    remove_edge(v, u, c)
            
            return False
        
        if not backtrack(0):
            return None  # Failed
    
    # Build output list
    color_list = []
    for i in range(target_n):
        for j in range(i + 1, target_n):
            color_list.append(colors[(i, j)])
    
    return (target_n, color_list)
