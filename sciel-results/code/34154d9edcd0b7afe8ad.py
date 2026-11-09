"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

import time
from itertools import combinations

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degeneracy_order(adj):
    n = P['n']
    remaining = set(range(n))
    order = []
    while remaining:
        # find vertex with minimum degree in remaining subgraph
        min_deg = min(len(adj[v] & remaining) for v in remaining)
        # pick one with that degree (break ties by index for determinism)
        candidates = [v for v in remaining if len(adj[v] & remaining) == min_deg]
        v = min(candidates)
        order.append(v)
        remaining.remove(v)
    return order

def _tomita_clique(adj, order):
    """Find maximum clique using Bron-Kerbosch with pivot and degeneracy ordering."""
    n = P['n']
    best_clique = []
    
    # Precompute adjacency as sets for fast intersection
    adj_sets = [set(adj[i]) for i in range(n)]
    
    def bronk(R, P_set, X_set):
        nonlocal best_clique
        if not P_set and not X_set:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Pivot selection: choose vertex in P ∪ X with max |N(v) ∩ P|
        U = P_set | X_set
        if U:
            pivot = max(U, key=lambda v: len(adj_sets[v] & P_set))
            candidates = [v for v in P_set if v not in adj_sets[pivot]]
        else:
            candidates = list(P_set)
        
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & adj_sets[v]
            new_X = X_set & adj_sets[v]
            bronk(new_R, new_P, new_X)
            P_set.discard(v)
            X_set.add(v)
    
    # Use degeneracy ordering for better performance
    for v in order:
        bronk({v}, adj_sets[v] & set(order[order.index(v)+1:]), set())
    
    return best_clique

def _find_max_clique(adj):
    """Find maximum clique, trying multiple approaches."""
    n = P['n']
    
    # First, try Bron-Kerbosch with degeneracy ordering
    order = _degeneracy_order(adj)
    clique = _tomita_clique(adj, order)
    
    if len(clique) >= 2:
        return clique
    
    # Fallback: find any edge
    if P['edges']:
        u, v = P['edges'][0]
        return [u, v]
    
    return []

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s limit
    
    adj = _build_adj()
    
    # Find maximum clique
    clique = _find_max_clique(adj)
    
    if len(clique) < 2:
        # Fallback: return any edge
        u, v = P['edges'][0]
        branches = [u, v]
        paths = [[u, v]]
    else:
        branches = list(clique)
        
        # For each pair of branches, find a simple path avoiding other branches
        # Since clique vertices are all mutually adjacent, direct edges work
        # But we need to ensure path interiors are disjoint and avoid all branches
        # Direct edges have no interior, so they're valid as long as they don't pass through other branches
        # A direct edge [u, v] has no interior vertices, so it's always valid
        
        paths = []
        for u, v in combinations(branches, 2):
            # Check if u and v are adjacent
            if v in adj[u]:
                paths.append([u, v])
            else:
                # Need to find a path avoiding all branch vertices
                # BFS from u to v avoiding all branches except endpoints
                forbidden = set(branches) - {u, v}
                visited = {u}
                parent = {u: None}
                queue = [u]
                found = False
                while queue:
                    curr = queue.pop(0)
                    if curr == v:
                        found = True
                        break
                    for neighbor in adj[curr]:
                        if neighbor not in visited and neighbor not in forbidden:
                            visited.add(neighbor)
                            parent[neighbor] = curr
                            queue.append(neighbor)
                if found:
                    path = []
                    node = v
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    paths.append(path)
                else:
                    # Should not happen in a connected graph, but fallback to direct edge if possible
                    paths.append([u, v])
    
    # Ensure types are correct: branches is list of ints, paths is list of lists of ints
    branches = [int(b) for b in branches]
    paths = [[int(p) for p in path] for path in paths]
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Close to deadline, return immediately
        pass
    
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
