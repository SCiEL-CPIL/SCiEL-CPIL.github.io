"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _get_diameter_scipy(adj, n):
    rows = []
    cols = []
    for u in range(n):
        for v in adj[u]:
            rows.append(u)
            cols.append(v)
    if not rows:
        return 0
    data = np.ones(len(rows))
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    dist = shortest_path(graph, method='D')
    if np.any(np.isinf(dist)):
        return float('inf')
    return np.max(dist)

def _is_triangle_free(adj, u, v, n):
    nu = set(adj[u])
    for w in adj[v]:
        if w in nu:
            return False
    return True

def construct():
    n = P['n']
    base_edges = [tuple(e) for e in P['edges']]
    base_edge_set = set()
    for u, v in base_edges:
        base_edge_set.add((min(u,v), max(u,v)))
    
    adj_base = _build_adj(base_edges, n)
    
    all_possible_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edge_set:
                continue
            if abs(u - v) == 2:
                continue
            all_possible_edges.append((u, v))
            
    beam_width = 20
    max_depth = 8
    
    initial_adj = [list(x) for x in adj_base]
    initial_edge_set = set(base_edge_set)
    initial_diam = _get_diameter_scipy(adj_base, n)
    
    beam = [(initial_diam, 0, [], initial_adj, initial_edge_set)]
    
    best_solution = None
    
    for depth in range(max_depth):
        next_beam = []
        for diam, num_edges, added_edges, adj, edge_set in beam:
            if diam <= 4:
                if best_solution is None or num_edges < len(best_solution):
                    best_solution = list(added_edges)
                continue
            
            candidates = []
            for u, v in all_possible_edges:
                if (min(u,v), max(u,v)) in edge_set:
                    continue
                if _is_triangle_free(adj, u, v, n):
                    candidates.append((u, v))
            
            candidates.sort(key=lambda x: abs(x[0]-x[1]), reverse=True)
            candidates = candidates[:20]
            
            for u, v in candidates:
                new_adj = [list(x) for x in adj]
                new_adj[u].append(v)
                new_adj[v].append(u)
                
                new_edge_set = set(edge_set)
                new_edge_set.add((min(u,v), max(u,v)))
                
                new_diam = _get_diameter_scipy(new_adj, n)
                
                new_added = list(added_edges)
                new_added.append((u, v))
                
                next_beam.append((new_diam, num_edges + 1, new_added, new_adj, new_edge_set))
        
        next_beam.sort(key=lambda x: (x[0], x[1]))
        beam = next_beam[:beam_width]
        
        if best_solution is not None and depth >= len(best_solution):
            break

    if best_solution is None:
        return []
        
    return [[int(u), int(v)] for u, v in best_solution]
# EVOLVE-BLOCK-END
