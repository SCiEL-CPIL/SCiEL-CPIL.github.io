import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _enumerate_maximal_bicliques(adj, n, deadline, max_bicliques=500):
    """
    Enumerate maximal bicliques using a Bron-Kerbosch-like approach.
    For each pair of vertices (u,v) with an edge, find the maximal biclique
    containing that edge.
    """
    bicliques = []
    seen = set()
    
    # Sort vertices by degree for efficiency
    deg = [len(adj[v]) for v in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: -deg[x])
    
    for i, u in enumerate(sorted_vertices):
        if time.time() > deadline:
            break
        for j in range(i, len(sorted_vertices)):
            v = sorted_vertices[j]
            if time.time() > deadline:
                break
            
            key = _edge_key(u, v)
            if key not in adj[u]:
                continue
            
            # Find maximal biclique containing edge (u,v)
            # Left candidates: vertices connected to all of right
            # Right candidates: vertices connected to all of left
            
            # Start with left={u}, right={v}
            left = {u}
            right = {v}
            
            # Iteratively expand
            for _ in range(100):
                if time.time() > deadline:
                    break
                changed = False
                
                # Try to add vertices to left
                new_left = set(left)
                for w in range(n):
                    if w not in left and w not in right:
                        # w can be added to left if connected to all in right
                        if all(_edge_key(w, x) in adj[w] for x in right):
                            new_left.add(w)
                            changed = True
                if new_left != left:
                    left = new_left
                
                # Try to add vertices to right
                new_right = set(right)
                for w in range(n):
                    if w not in left and w not in right:
                        # w can be added to right if connected to all in left
                        if all(_edge_key(w, x) in adj[w] for x in left):
                            new_right.add(w)
                            changed = True
                if new_right != right:
                    right = new_right
                
                if not changed:
                    break
            
            # Create a canonical representation
            biclique_key = (frozenset(left), frozenset(right))
            if biclique_key not in seen:
                seen.add(biclique_key)
                bicliques.append((list(left), list(right)))
                
                if len(bicliques) >= max_bicliques:
                    return bicliques
    
    return bicliques

def _greedy_set_cover(bicliques, edge_set, deadline):
    """
    Greedy set cover: repeatedly pick the biclique that covers the most uncovered edges.
    """
    remaining = set(edge_set)
    selected = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        best_biclique = None
        best_count = 0
        
        for i, (left, right) in enumerate(bicliques):
            # Count how many remaining edges this biclique covers
            count = 0
            for u in left:
                for v in right:
                    key = _edge_key(u, v)
                    if key in remaining:
                        count += 1
            
            if count > best_count:
                best_count = count
                best_biclique = i
        
        if best_biclique is None or best_count == 0:
            # Fallback: pick any remaining edge as a single-edge biclique
            u, v = next(iter(remaining))
            selected.append([[u], [v]])
            remaining.discard(_edge_key(u, v))
        else:
            left, right = bicliques[best_biclique]
            selected.append([left, right])
            for u in left:
                for v in right:
                    key = _edge_key(u, v)
                    if key in remaining:
                        remaining.discard(key)
    
    return selected

def _try_merge(bicliques, edge_set, n, deadline):
    """Try to merge pairs of bicliques if their union forms a valid biclique."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try merging: (L1∪L2, R1∪R2)
                L_new = set(L1) | set(L2)
                R_new = set(R1) | set(R2)
                
                # Check all cross pairs are edges
                valid = True
                for u in L_new:
                    for v in R_new:
                        if u == v or _edge_key(u, v) not in edge_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check coverage matches
                covered_i = set(_edge_key(u, v) for u in L1 for v in R1)
                covered_j = set(_edge_key(u, v) for u in L2 for v in R2)
                covered_new = set(_edge_key(u, v) for u in L_new for v in R_new)
                
                if covered_new == covered_i | covered_j:
                    bicliques[i] = [list(L_new), list(R_new)]
                    bicliques.pop(j)
                    changed = True
                    break
            if changed:
                break
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    edges = P['edges']
    n = P['n']
    edge_set = set(_edge_key(u, v) for u, v in edges)
    adj = _build_adjacency(edge_set, n)
    
    best_bicliques = None
    best_count = float('inf')
    
    # Phase 1: Enumerate maximal bicliques
    maximal_bicliques = _enumerate_maximal_bicliques(adj, n, deadline, max_bicliques=1000)
    
    if maximal_bicliques:
        # Phase 2: Greedy set cover using maximal bicliques
        bicliques = _greedy_set_cover(maximal_bicliques, edge_set, deadline)
        
        # Phase 3: Try to merge to reduce count further
        bicliques = _try_merge(bicliques, edge_set, n, deadline)
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
    
    # Fallback if enumeration didn't work well
    if best_bicliques is None or best_count > n:
        # Star partition as fallback
        bicliques = []
        for v in range(n):
            if adj[v]:
                bicliques.append([[v], list(adj[v])])
        
        bicliques = _try_merge(bicliques, edge_set, n, deadline)
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
    
    if best_bicliques is None:
        # Absolute fallback
        best_bicliques = [[[e[0]], [e[1]]] for e in edge_set]
    
    # Ensure types are correct
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result
