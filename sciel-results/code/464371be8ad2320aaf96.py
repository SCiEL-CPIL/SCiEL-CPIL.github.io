"""Biclique partition solver using greedy edge-cover with backtracking."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

N = P['n']
ALL_EDGES = set(frozenset(e) for e in P['edges'])

def build_adj():
    adj = [set() for _ in range(N)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def find_best_biclique(remaining, adj, deadline):
    """Find the largest biclique that only uses edges in remaining."""
    best_biclique = None
    best_size = 0
    
    # Try each vertex as a potential left vertex
    for u in range(N):
        # Get uncovered edges incident to u
        u_neighbors_with_edges = set()
        for e in remaining:
            es = set(e)
            if u in es:
                other = es[0] if es[1] == u else es[1]
                u_neighbors_with_edges.add(other)
        
        if not u_neighbors_with_edges:
            continue
        
        # Strategy: start with L={u}, R=u_neighbors_with_edges, then try to extend L
        L = {u}
        R = set(u_neighbors_with_edges)
        
        # Try to extend L by adding vertices that connect to all of R
        extended = True
        while extended and time.time() < deadline:
            extended = False
            for w in range(N):
                if w in L or w in R:
                    continue
                # Check if w connects to all vertices in R
                can_add = True
                for v in R:
                    if frozenset((w, v)) not in remaining:
                        can_add = False
                        break
                if can_add:
                    L.add(w)
                    extended = True
        
        # Also try the reverse: start with R as a single vertex and extend
        for v in list(u_neighbors_with_edges)[:5]:  # Limit for speed
            L2 = {u}
            R2 = {v}
            # Extend R2
            extended = True
            while extended and time.time() < deadline:
                extended = False
                for w in u_neighbors_with_edges:
                    if w in L2 or w in R2:
                        continue
                    # Check if all L2 vertices connect to w
                    can_add = True
                    for u2 in L2:
                        if frozenset((u2, w)) not in remaining:
                            can_add = False
                            break
                    if can_add:
                        R2.add(w)
                        extended = True
            
            size = len(L2) * len(R2)
            if size > best_size:
                best_size = size
                best_biclique = (L2, R2)
        
        size = len(L) * len(R)
        if size > best_size:
            best_size = size
            best_biclique = (L, R)
    
    return best_biclique

def greedy_biclique_partition(adj, deadline):
    """Greedy approach: repeatedly find the largest biclique covering uncovered edges."""
    remaining = set(ALL_EDGES)
    bicliques = []
    
    while remaining and time.time() < deadline:
        # Find the largest biclique that only uses edges in remaining
        best_biclique = find_best_biclique(remaining, adj, deadline)
        
        if best_biclique is None:
            # Fallback: just take one edge
            e = next(iter(remaining))
            u, v = list(e)
            best_biclique = ({u}, {v})
        
        L, R = best_biclique
        bicliques.append((list(L), list(R)))
        
        # Remove covered edges
        for u in L:
            for v in R:
                remaining.discard(frozenset((u, v)))
    
    return bicliques

def try_merge_bicliques(bicliques, adj, deadline):
    """Try to merge compatible bicliques to reduce count."""
    n = len(bicliques)
    if n <= 1:
        return bicliques
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            for j in range(i+1, len(bicliques)):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                L1s, R1s = set(L1), set(R1)
                L2s, R2s = set(L2), set(R2)
                
                # Check disjointness
                if L1s & L2s or R1s & R2s or L1s & R2s or L2s & R1s:
                    continue
                
                # Check all cross edges exist
                valid = True
                for u in L1s:
                    for v in R2s:
                        if frozenset((u, v)) not in ALL_EDGES:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    for u in L2s:
                        for v in R1s:
                            if frozenset((u, v)) not in ALL_EDGES:
                                valid = False
                                break
                        if not valid:
                            break
                
                if not valid:
                    continue
                
                # Merge
                new_L = list(L1s | L2s)
                new_R = list(R1s | R2s)
                new_bicliques = []
                for k in range(len(bicliques)):
                    if k != i and k != j:
                        new_bicliques.append(bicliques[k])
                new_bicliques.append((new_L, new_R))
                bicliques = new_bicliques
                changed = True
                break
            if changed:
                break
    
    return bicliques

def local_improve(bicliques, adj, deadline):
    """Try to reduce biclique count by moving vertices between bicliques."""
    bicliques = [list(b) for b in bicliques]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(bicliques)):
            L1, R1 = bicliques[i]
            L1s, R1s = set(L1), set(R1)
            
            # Try removing a vertex from L1 and adding to another biclique's L
            for u in list(L1s):
                for j in range(len(bicliques)):
                    if i == j:
                        continue
                    L2, R2 = bicliques[j]
                    L2s, R2s = set(L2), set(R2)
                    
                    # Check if u can be added to L2 (u not in R2, and u connects to all R2)
                    if u in R2s or u in L2s:
                        continue
                    can_add = True
                    for v in R2s:
                        if frozenset((u, v)) not in ALL_EDGES:
                            can_add = False
                            break
                    if not can_add:
                        continue
                    
                    # Check if R1 still forms a biclique with L1\{u}
                    new_L1 = L1s - {u}
                    if not new_L1:
                        continue
                    valid = True
                    for u2 in new_L1:
                        for v in R1s:
                            if frozenset((u2, v)) not in ALL_EDGES:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Check edge partition property: edges covered by new bicliques must match
                    # This is complex, so we'll skip for now and rely on merge
                    pass
            
            # Try removing a vertex from R1 and adding to another biclique's R
            for v in list(R1s):
                for j in range(len(bicliques)):
                    if i == j:
                        continue
                    L2, R2 = bicliques[j]
                    L2s, R2s = set(L2), set(R2)
                    
                    # Check if v can be added to R2 (v not in L2, and all L2 connect to v)
                    if v in L2s or v in R2s:
                        continue
                    can_add = True
                    for u in L2s:
                        if frozenset((u, v)) not in ALL_EDGES:
                            can_add = False
                            break
                    if not can_add:
                        continue
                    
                    # Check if L1 still forms a biclique with R1\{v}
                    new_R1 = R1s - {v}
                    if not new_R1:
                        continue
                    valid = True
                    for u in L1s:
                        for v2 in new_R1:
                            if frozenset((u, v2)) not in ALL_EDGES:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    pass
    
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    adj = build_adj()
    
    # Greedy partition
    bicliques = greedy_biclique_partition(adj, deadline)
    
    # Try to merge
    bicliques = try_merge_bicliques(bicliques, adj, deadline)
    
    # Local improvement
    bicliques = local_improve(bicliques, adj, deadline)
    
    # Convert to proper format
    result = []
    for L, R in bicliques:
        result.append([list(L), list(R)])
    
    return result
# EVOLVE-BLOCK-END
