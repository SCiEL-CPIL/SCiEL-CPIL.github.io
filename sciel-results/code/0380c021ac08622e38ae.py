"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    n = P['n']
    
    # Build static adjacency sets for host graph
    host_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        host_adj[u].add(v)
        host_adj[v].add(u)
        
    # Initialize remaining edges as a set of frozensets for O(1) lookup and removal
    remaining_edges = set()
    for u, v in P['edges']:
        if u < v:
            remaining_edges.add(frozenset((u, v)))
        else:
            remaining_edges.add(frozenset((v, u)))
            
    best_solution = None
    best_count = float('inf')
    
    def _expand_biclique(start_L, start_R, remaining, n):
        """Expand biclique by adding vertices connected to all of the other side."""
        L = set(start_L)
        R = set(start_R)
        changed = True
        while changed:
            changed = False
            # Try to expand L: find w connected to all in R
            best_w_L = None
            best_score_L = -1
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for r in R:
                    if frozenset((w, r)) not in remaining:
                        ok = False
                        break
                if ok:
                    # Score: prefer vertices that allow more R expansion
                    # Count how many vertices in R they're connected to (all of them by definition)
                    # Use degree as tiebreaker
                    score = len(R) * 1000 + sum(1 for x in range(n) if x not in L and x not in R and frozenset((w, x)) in remaining)
                    if score > best_score_L:
                        best_score_L = score
                        best_w_L = w
            if best_w_L is not None:
                L.add(best_w_L)
                changed = True
            
            # Try to expand R: find w connected to all in L
            best_w_R = None
            best_score_R = -1
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for l in L:
                    if frozenset((w, l)) not in remaining:
                        ok = False
                        break
                if ok:
                    score = len(L) * 1000 + sum(1 for x in range(n) if x not in L and x not in R and frozenset((w, x)) in remaining)
                    if score > best_score_R:
                        best_score_R = score
                        best_w_R = w
            if best_w_R is not None:
                R.add(best_w_R)
                changed = True
        return L, R
    
    def _try_merge(bicliques, remaining_edges):
        """Try to merge bicliques if their union forms a valid biclique."""
        merged = True
        while merged:
            merged = False
            new_bicliques = []
            used = [False] * len(bicliques)
            for i in range(len(bicliques)):
                if used[i]:
                    continue
                Li, Ri = set(bicliques[i][0]), set(bicliques[i][1])
                for j in range(i + 1, len(bicliques)):
                    if used[j]:
                        continue
                    Lj, Rj = set(bicliques[j][0]), set(bicliques[j][1])
                    # Try merging: new_L = Li ∪ Lj, new_R = Ri ∪ Rj
                    # Check if all cross edges exist in remaining_edges
                    new_L = Li | Lj
                    new_R = Ri | Rj
                    valid = True
                    for l in new_L:
                        for r in new_R:
                            if frozenset((l, r)) not in remaining_edges:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        used[j] = True
                        Li = new_L
                        Ri = new_R
                        merged = True
                        break
                new_bicliques.append([sorted(Li), sorted(Ri)])
            if merged:
                bicliques = new_bicliques
        return bicliques
    
    # Run multiple attempts with different heuristics
    for attempt in range(30):
        if time.time() > deadline:
            break
            
        current_remaining = set(remaining_edges)
        bicliques = []
        valid_solution = True
        
        while current_remaining:
            if time.time() > deadline:
                valid_solution = False
                break
                
            # Compute degrees in current_remaining
            deg = [0] * n
            for e in current_remaining:
                u, v = list(e)
                deg[u] += 1
                deg[v] += 1
            
            if not any(deg):
                break
            
            # Try multiple candidate starting vertices and pick the best biclique
            best_L = None
            best_R = None
            best_removed = 0
            
            # Candidate vertices: top by degree, plus some random ones for diversity
            candidates = sorted(range(n), key=lambda x: -deg[x])[:10]
            if attempt % 5 == 0:
                # Add some random candidates for diversity
                for _ in range(5):
                    candidates.append(random.randint(0, n - 1))
            
            for v_start in candidates:
                if deg[v_start] == 0:
                    continue
                
                # Initialize with L={v_start}, R=neighbors of v_start in current_remaining
                R_init = set()
                for e in current_remaining:
                    if v_start in e:
                        other = list(e)[0] if list(e)[0] != v_start else list(e)[1]
                        R_init.add(other)
                
                if not R_init:
                    continue
                
                L, R = _expand_biclique({v_start}, R_init, current_remaining, n)
                
                # Count edges removed
                removed = 0
                for l in L:
                    for r in R:
                        if frozenset((l, r)) in current_remaining:
                            removed += 1
                
                # Prefer larger bicliques (more edges removed), tiebreak by product of side sizes
                if removed > best_removed or (removed == best_removed and len(L) * len(R) > len(best_L or []) * len(best_R or [])):
                    best_removed = removed
                    best_L = L
                    best_R = R
            
            if best_L is None or best_removed == 0:
                valid_solution = False
                break
            
            # Remove edges
            for l in best_L:
                for r in best_R:
                    e = frozenset((l, r))
                    if e in current_remaining:
                        current_remaining.remove(e)
            
            bicliques.append([sorted(best_L), sorted(best_R)])
        
        if not valid_solution:
            continue
        
        # Try to merge bicliques
        bicliques = _try_merge(bicliques, remaining_edges)
        
        # Verify solution
        covered_edges = set()
        for left, right in bicliques:
            for l in left:
                for r in right:
                    e = frozenset((l, r))
                    if e in covered_edges:
                        valid_solution = False
                        break
                    covered_edges.add(e)
                    if e not in remaining_edges:
                        valid_solution = False
                        break
                if not valid_solution:
                    break
            if not valid_solution:
                break
        
        if not valid_solution:
            continue
        
        if covered_edges != remaining_edges:
            continue
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_solution = bicliques
        
        if best_count == 1:
            break

    if best_solution is None:
        # Fallback: Star decomposition
        bicliques = []
        for u in range(n):
            R_u = [v for v in host_adj[u] if v > u]
            if R_u:
                bicliques.append([[u], sorted(R_u)])
        best_solution = bicliques
        best_count = len(bicliques)
        
    # Convert to required format
    final_result = []
    for bic in best_solution:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        final_result.append([left, right])
        
    return final_result

# EVOLVE-BLOCK-END
