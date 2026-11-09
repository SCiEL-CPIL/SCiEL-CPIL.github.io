import time
import random
from itertools import combinations

def is_2_colorable(edges, n):
    # Same as before
    colors = [0] * n
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    def propagate(v, c):
        if colors[v] != 0 and colors[v] != c:
            return False
        if colors[v] == c:
            return True
        colors[v] = c
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            unassigned_in_e = [u for u in e if u != v and colors[u] == 0]
            assigned_in_e = [colors[u] for u in e if u != v and colors[u] != 0]
            
            if len(unassigned_in_e) == 1:
                u = unassigned_in_e[0]
                if all(x == c for x in assigned_in_e):
                    if not propagate(u, 3-c):
                        return False
                elif all(x == 3-c for x in assigned_in_e):
                    if not propagate(u, c):
                        return False
            elif len(unassigned_in_e) == 0:
                if all(x == c for x in assigned_in_e) or all(x == 3-c for x in assigned_in_e):
                    return False
        return True

    def backtrack(idx):
        if idx == n:
            return True
        old_colors = colors[:]
        if propagate(idx, 1):
            if backtrack(idx + 1):
                return True
        colors = old_colors[:]
        old_colors = colors[:]
        if propagate(idx, 2):
            if backtrack(idx + 1):
                return True
        return False

    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n = 16
    edges = []
    edge_set = set()
    
    # Greedy construction
    while time.time() < deadline:
        # Generate random coloring
        coloring = [random.choice([1, 2]) for _ in range(n)]
        
        # Find monochromatic 5-sets in this coloring
        mono_edges = []
        for combo in combinations(range(n), 5):
            c = coloring[combo[0]]
            if all(coloring[v] == c for v in combo):
                mono_edges.append(combo)
        
        if not mono_edges:
            continue
            
        # Pick a random mono edge
        e = random.choice(mono_edges)
        if e not in edge_set:
            edge_set.add(e)
            edges.append(list(e))
            
        # Check if current hypergraph is non-2-colorable
        if not is_2_colorable(edges, n):
            return {'n': n, 'edges': edges}
            
        # If we exceed 126 edges, break
        if len(edges) > 126:
            break

    # Fallback
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
