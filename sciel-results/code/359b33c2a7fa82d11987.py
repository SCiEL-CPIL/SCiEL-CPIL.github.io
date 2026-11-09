"""Combined program: maximum clique via Tomita-style coloring + Bron-Kerbosch with degeneracy ordering."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degeneracy_order(adj):
    n = P['n']
    remaining = list(range(n))
    remaining_set = set(remaining)
    order = []
    while remaining:
        min_deg = min(len(adj[v] & remaining_set) for v in remaining)
        v = next(v for v in remaining if len(adj[v] & remaining_set) == min_deg)
        order.append(v)
        remaining.remove(v)
        remaining_set.discard(v)
    return order

def _color_sort(vertices, adj, deadline):
    """Greedy coloring to produce an upper bound and ordering for Tomita's algorithm."""
    if not vertices:
        return []
    
    uncolored = list(vertices)
    color_classes = []
    colors = {}
    
    # Process vertices in a specific order for better coloring
    # Sort by degree in descending order for better coloring quality
    uncolored.sort(key=lambda v: -len(adj[v]))
    
    while uncolored:
        if time.time() > deadline:
            break
        current_class = []
        i = 0
        while i < len(uncolored):
            v = uncolored[i]
            # Check if v can be added to current class (not adjacent to any in current class)
            can_add = True
            for u in current_class:
                if u in adj[v]:
                    can_add = False
                    break
            if can_add:
                current_class.append(v)
                uncolored.pop(i)
            else:
                i += 1
        color_classes.append(current_class)
    
    # Return list of (color_index, vertex) sorted by color descending
    result = []
    for ci, cls in enumerate(color_classes):
        for v in cls:
            result.append((ci, v))
    # Sort by color descending, then by vertex
    result.sort(key=lambda x: (-x[0], x[1]))
    return result

def _tomita_clique(adj, R, P_list, X, best_clique, deadline):
    """Tomita's maximum clique algorithm with coloring."""
    if time.time() > deadline:
        return best_clique
    
    if not P_list:
        if len(R) > len(best_clique):
            best_clique = list(R)
        return best_clique
    
    if len(R) + len(P_list) <= len(best_clique):
        return best_clique
    
    # Color sort P_list
    colored = _color_sort(P_list, adj, deadline)
    
    # Process vertices in reverse order of coloring
    for color, v in reversed(colored):
        if time.time() > deadline:
            return best_clique
        
        # Upper bound: |R| + color + 1
        if len(R) + color + 1 <= len(best_clique):
            break
        
        new_R = R | {v}
        new_P = [w for w in P_list if w in adj[v]]
        new_X = [w for w in X if w in adj[v]]
        
        best_clique = _tomita_clique(adj, new_R, new_P, new_X, best_clique, deadline)
        
        # Remove v from P and add to X
        P_list.remove(v)
        X.append(v)
    
    return best_clique

def _max_clique(adj):
    n = P['n']
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    order = _degeneracy_order(adj)
    best = []
    
    for i, v in enumerate(order):
        if time.time() > deadline:
            break
        
        # P = neighbors of v that come after v in degeneracy order
        P_set = set()
        for w in adj[v]:
            if w in order[i+1:]:
                P_set.add(w)
        
        if not P_set:
            continue
        
        # X = neighbors of v that come before v in degeneracy order
        X_set = set()
        for w in adj[v]:
            if w in order[:i]:
                X_set.add(w)
        
        best = _tomita_clique(adj, {v}, list(P_set), list(X_set), best, deadline)
    
    if not best:
        # Fallback
        best = _tomita_clique(adj, set(), list(range(n)), [], best, deadline)
    
    return best

def _find_simple_path(adj, u, v, forbidden):
    """Find a simple path from u to v avoiding forbidden vertices (except endpoints)."""
    if u == v:
        return [u]
    
    visited = {u}
    queue = deque([(u, [u])])
    
    while queue:
        node, path = queue.popleft()
        if node == v:
            return path
        for nb in adj[node]:
            if nb not in visited and nb not in forbidden:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    
    return None

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    adj = _build_adj()
    clique = _max_clique(adj)
    
    # Ensure we have at least 2 branches
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
    
    clique = sorted(clique)
    branches = list(clique)
    
    # Build paths: for each pair of branches, find a simple path
    # Since they form a clique, each pair is directly adjacent
    paths = []
    branch_set = set(branches)
    
    for u, v in combinations(clique, 2):
        if time.time() > deadline:
            # Fallback: just use direct edges
            for remaining_u, remaining_v in list(combinations(clique, 2)):
                if [remaining_u, remaining_v] not in paths:
                    paths.append([remaining_u, remaining_v])
            break
        
        # Since it's a clique, direct edge should exist
        if v in adj[u]:
            paths.append([u, v])
        else:
            # Find path avoiding other branches
            forbidden = branch_set - {u, v}
            path = _find_simple_path(adj, u, v, forbidden)
            if path:
                paths.append(path)
            else:
                # Last resort: direct edge even if not present (shouldn't happen for clique)
                paths.append([u, v])
    
    # Verify and fix if needed
    if len(paths) != len(list(combinations(clique, 2))):
        # Rebuild paths with fallback
        paths = []
        for u, v in combinations(clique, 2):
            if v in adj[u]:
                paths.append([u, v])
            else:
                forbidden = branch_set - {u, v}
                path = _find_simple_path(adj, u, v, forbidden)
                if path:
                    paths.append(path)
                else:
                    paths.append([u, v])
    
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
