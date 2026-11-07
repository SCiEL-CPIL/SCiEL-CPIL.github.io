"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_edge_set():
    es = set()
    for u, v in P['edges']:
        es.add((min(u,v), max(u,v)))
    return es

def _is_biclique(L, R, edge_set):
    """Check if all cross pairs between L and R are edges."""
    for u in L:
        for v in R:
            if (min(u,v), max(u,v)) not in edge_set:
                return False
    return True

def _greedy_biclique_partition(edge_set, deadline):
    """Greedy: repeatedly find largest biclique, remove its edges, recurse."""
    remaining = set(edge_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        # Find a good biclique from remaining edges
        best_biclique = None
        best_size = 0
        
        # Get all vertices involved in remaining edges
        verts = set()
        for u, v in remaining:
            verts.add(u)
            verts.add(v)
        verts = list(verts)
        
        # Try each vertex as a starting point for left side
        for start in verts:
            if time.time() > deadline:
                break
            # Collect all neighbors of start in remaining edges
            neighbors = set()
            for u, v in remaining:
                if u == start:
                    neighbors.add(v)
                elif v == start:
                    neighbors.add(u)
            
            if not neighbors:
                continue
            
            # Try to grow a biclique: L = {start} + some neighbors, R = common neighbors
            # Strategy: L starts as {start}, R = neighbors(start)
            # Then try to add more vertices to L that connect to all of R
            L = [start]
            R = list(neighbors)
            
            # Try to grow R by finding vertices connected to all of L
            # Actually, let's try: fix L, find max R, or fix R, find max L
            
            # Approach: start with L={start}, R=neighbors(start)
            # Then iteratively add to L vertices that connect to all of R
            changed = True
            while changed:
                changed = False
                candidates = set(verts) - set(L)
                for cand in list(candidates):
                    if cand in R:
                        continue
                    # Check if cand connects to all of R
                    all_connect = True
                    for r in R:
                        if (min(cand, r), max(cand, r)) not in remaining:
                            all_connect = False
                            break
                    if all_connect:
                        L.append(cand)
                        changed = True
            
            # Now try to grow R: vertices that connect to all of L
            R_set = set(R)
            changed = True
            while changed:
                changed = False
                candidates = set(verts) - set(L)
                for cand in list(candidates):
                    if cand in R_set:
                        continue
                    all_connect = True
                    for l in L:
                        if (min(cand, l), max(cand, l)) not in remaining:
                            all_connect = False
                            break
                    if all_connect:
                        R_set.add(cand)
                        changed = True
            R = list(R_set)
            
            if len(L) * len(R) > best_size:
                best_size = len(L) * len(R)
                best_biclique = (L, R)
        
        if best_biclique is None or best_size == 0:
            # Fallback: single edge
            u, v = next(iter(remaining))
            bicliques.append(([u], [v]))
            remaining.discard((u, v))
        else:
            L, R = best_biclique
            bicliques.append((L, R))
            # Remove all edges in this biclique from remaining
            for u in L:
                for v in R:
                    remaining.discard((min(u,v), max(u,v)))
    
    return bicliques

def _improve_partition(bicliques, edge_set, deadline):
    """Try to improve by merging, splitting, and local search."""
    bicliques = [list(b) for b in bicliques]
    
    # Try to merge pairs of bicliques if their union is still a valid biclique
    # Actually, merging isn't straightforward. Instead, try local search.
    
    # Local search: try moving vertices between bicliques
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            L_i, R_i = bicliques[i]
            # Try moving a vertex from L_i to R_i or vice versa
            for j, u in enumerate(L_i):
                # Try moving u from L_i to R_i
                L_new = L_i[:j] + L_i[j+1:]
                R_new = R_i + [u]
                if L_new and R_new and _is_biclique(L_new, R_new, edge_set):
                    # Check that all edges still covered
                    # This is complex; skip for now
                    pass
                # Try moving u from L_i to another biclique's L or R
                for k in range(len(bicliques)):
                    if k == i:
                        continue
                    L_k, R_k = bicliques[k]
                    # Move u to L_k
                    L_k_new = L_k + [u]
                    if L_k_new and R_k and _is_biclique(L_k_new, R_k, edge_set):
                        bicliques[i] = [L_new, R_i]
                        bicliques[k] = [L_k_new, R_k]
                        improved = True
                        break
                    # Move u to R_k
                    R_k_new = R_k + [u]
                    if L_k and R_k_new and _is_biclique(L_k, R_k_new, edge_set):
                        bicliques[i] = [L_new, R_i]
                        bicliques[k] = [L_k, R_k_new]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s
    
    edge_set = _build_edge_set()
    
    # Greedy biclique partition
    bicliques = _greedy_biclique_partition(edge_set, deadline)
    
    # Convert to list format and ensure types are correct
    result = []
    for L, R in bicliques:
        result.append([list(L), list(R)])
    
    # Ensure all vertices in bicliques are Python ints
    for bic in result:
        bic[0] = [int(v) for v in bic[0]]
        bic[1] = [int(v) for v in bic[1]]
    
    return result
# EVOLVE-BLOCK-END
