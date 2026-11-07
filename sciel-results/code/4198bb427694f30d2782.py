"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_edge_set():
    es = set()
    for a, b in P['edges']:
        es.add((min(a,b), max(a,b)))
    return es

def _is_biclique(L, R, edge_set):
    """Check if all cross pairs are edges."""
    for u in L:
        for v in R:
            pair = (min(u,v), max(u,v))
            if pair not in edge_set:
                return False
    return True

def _cross_edges(L, R):
    """Get set of cross edges between L and R."""
    es = set()
    for u in L:
        for v in R:
            es.add((min(u,v), max(u,v)))
    return es

def _validate_partition(bicliques, edge_set):
    """Validate that bicliques form a valid partition."""
    seen = set()
    for L, R in bicliques:
        if not L or not R:
            return False
        # Check all cross pairs are edges
        for u in L:
            for v in R:
                pair = (min(u,v), max(u,v))
                if pair not in edge_set:
                    return False
                if pair in seen:
                    return False
                seen.add(pair)
    return seen == edge_set

def _greedy_biclique_partition(edge_set, deadline):
    """Greedy: repeatedly find largest valid biclique covering uncovered edges."""
    n = P['n']
    uncovered = set(edge_set)
    bicliques = []
    
    while uncovered and time.time() < deadline:
        best_biclique = None
        best_size = 0
        
        # Try each pair of vertex sets
        for i in range(n):
            for j in range(i+1, n):
                # Try {i} as left, find max right
                L = [i]
                R = [j]
                # Expand R
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    for k in range(n):
                        if k not in R and k != i:
                            # Check if k can be added to R
                            can_add = True
                            for u in L:
                                if (min(u,k), max(u,k)) not in edge_set:
                                    can_add = False
                                    break
                            if can_add:
                                R.append(k)
                                changed = True
                
                # Check if this biclique covers any uncovered edges
                ce = _cross_edges(L, R)
                covered = ce & uncovered
                if len(covered) > best_size:
                    best_size = len(covered)
                    best_biclique = (L, R)
        
        if best_biclique is None:
            # Fallback: single edge
            e = next(iter(uncovered))
            best_biclique = ([e[0]], [e[1]])
            best_size = 1
        
        L, R = best_biclique
        bicliques.append([L, R])
        ce = _cross_edges(L, R)
        uncovered -= ce
    
    return bicliques

def _local_search(bicliques, edge_set, deadline):
    """Try to reduce number of bicliques via merging."""
    bicliques = [list(b) for b in bicliques]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_bic = len(bicliques)
        for i in range(n_bic):
            if time.time() > deadline:
                break
            for j in range(i+1, n_bic):
                # Try merging biclique i and j
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try all ways to merge: (L1∪L2, R1), (L1, R1∪R2), (L1∪L2, R1∪R2)
                for Lm, Rm in [(list(set(L1+L2)), list(R1)),
                                (list(L1), list(set(R1+R2))),
                                (list(set(L1+L2)), list(set(R1+R2)))]:
                    if not Lm or not Rm:
                        continue
                    if not _is_biclique(Lm, Rm, edge_set):
                        continue
                    # Check that merging doesn't create duplicate edge coverage
                    # The new biclique's cross edges must be subset of (old i cross + old j cross)
                    ce_new = _cross_edges(Lm, Rm)
                    ce_old_i = _cross_edges(L1, R1)
                    ce_old_j = _cross_edges(L2, R2)
                    if ce_new <= (ce_old_i | ce_old_j):
                        # Valid merge
                        new_bicliques = [list(b) for b in bicliques if b is not bicliques[i] and b is not bicliques[j]]
                        new_bicliques.append([Lm, Rm])
                        if _validate_partition(new_bicliques, edge_set):
                            bicliques = new_bicliques
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s budget, 10s safety margin
    
    edge_set = _build_edge_set()
    
    # Start with greedy partition
    bicliques = _greedy_biclique_partition(edge_set, deadline)
    
    # Local search to improve
    if time.time() < deadline:
        bicliques = _local_search(bicliques, edge_set, deadline)
    
    # Final validation and conversion
    if not _validate_partition(bicliques, edge_set):
        # Fallback to seed solution
        bicliques = [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
    
    # Convert to proper types
    result = []
    for L, R in bicliques:
        L = [int(x) for x in L]
        R = [int(x) for x in R]
        result.append([L, R])
    
    return result
# EVOLVE-BLOCK-END
