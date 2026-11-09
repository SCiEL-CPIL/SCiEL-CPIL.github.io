"""Rewritten program: maximum clique via Tomita-style coloring + Bron-Kerbosch."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _color_sort(P_set, adj):
    """
    Assigns colors to vertices in P_set such that no two adjacent vertices
    have the same color. Returns a list of (vertex, color) sorted by color
    in descending order.
    """
    if not P_set:
        return []
    
    P_avail = set(P_set)
    colors = {}
    color = 0
    
    while P_avail:
        color += 1
        # Sort candidates by degree descending to potentially find larger independent sets
        candidates = sorted(list(P_avail), key=lambda x: len(adj[x]), reverse=True)
        
        current_color_set = []
        for v in candidates:
            if v in P_avail:
                current_color_set.append(v)
                P_avail.remove(v)
                # Remove neighbors of v from P_avail
                neighbors = adj[v] & P_avail
                P_avail -= neighbors
                
        for v in current_color_set:
            colors[v] = color
            
    # Sort by color descending
    sorted_vertices = sorted(colors.keys(), key=lambda x: colors[x], reverse=True)
    return [(v, colors[v]) for v in sorted_vertices]

def _bron_kerbosch_tomita(adj, R, P_set, X, best_clique, deadline):
    """
    Bron-Kerbosch with Tomita coloring.
    """
    if time.time() > deadline:
        return best_clique
    
    if not P_set and not X:
        if len(R) > len(best_clique):
            best_clique = list(R)
        return best_clique
        
    # Color sort
    colored_P = _color_sort(P_set, adj)
    
    # Process in descending color order
    for v, c in colored_P:
        if time.time() > deadline:
            return best_clique
            
        # Pruning: If |R| + c <= |best_clique|, we can stop.
        if len(R) + c <= len(best_clique):
            break
            
        new_R = R | {v}
        new_P = P_set & adj[v]
        new_X = X & adj[v]
        
        best_clique = _bron_kerbosch_tomita(adj, new_R, new_P, new_X, best_clique, deadline)
        
        # Move v from P to X
        P_set.discard(v)
        X.add(v)
        
    return best_clique

def _max_clique(adj):
    n = P['n']
    deadline = time.time() + 50
    
    # Calculate degeneracy order
    remaining = list(range(n))
    order = []
    while remaining:
        min_deg = min(len(adj[v] & set(remaining)) for v in remaining)
        v = next(v for v in remaining if len(adj[v] & set(remaining)) == min_deg)
        order.append(v)
        remaining.remove(v)
        
    best = []
    # Process in degeneracy order
    for i, v in enumerate(order):
        if time.time() > deadline:
            break
        P_set = set()
        for w in adj[v]:
            if w in order[i+1:]:
                P_set.add(w)
        if not P_set:
            continue
        X = set()
        for w in adj[v]:
            if w in order[:i]:
                X.add(w)
                
        best = _bron_kerbosch_tomita(adj, {v}, P_set, X, best, deadline)
        
    if not best:
        best = _bron_kerbosch_tomita(adj, set(), set(range(n)), set(), best, deadline)
        
    return best

def construct():
    adj = _build_adj()
    clique = _max_clique(adj)
    
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
        
    clique = sorted(clique)
    branches = list(clique)
    
    paths = []
    for u, v in combinations(clique, 2):
        if v in adj[u]:
            paths.append([u, v])
        else:
            from collections import deque
            visited = {u}
            queue = deque([(u, [u])])
            found = None
            while queue:
                node, path = queue.popleft()
                if node == v:
                    found = path
                    break
                for nb in adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, path + [nb]))
            if found:
                paths.append(found)
            else:
                paths.append([u, v])
                
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
