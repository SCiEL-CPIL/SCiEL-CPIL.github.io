"""Biclique partition: greedy maximal extraction + local search merging."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_biclique(L, R, adj):
    for u in L:
        for v in R:
            if v not in adj[u]:
                return False
    return True

def _biclique_edges(L, R):
    return frozenset(frozenset((u, v)) for u in L for v in R)

def _greedy_partition(adj, deadline):
    n = P['n']
    all_edges = set(frozenset((u, v)) for u, v in P['edges'])
    remaining = set(all_edges)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        # Find a maximal biclique in remaining edges
        # Start from the edge with most remaining neighbors
        best_biclique = None
        best_size = 0
        
        # Try growing from each remaining edge
        for e in list(remaining):
            u, v = tuple(e)
            # Maximal common neighbors in remaining
            # L candidates: vertices connected to v via remaining edges
            # R candidates: vertices connected to u via remaining edges
            L_cand = set()
            R_cand = set()
            for e2 in remaining:
                a, b = tuple(e2)
                if a == u:
                    R_cand.add(b)
                elif b == u:
                    R_cand.add(a)
                if a == v:
                    L_cand.add(b)
                elif b == v:
                    L_cand.add(a)
            
            # Try to grow: L must connect to all of R, R must connect to all of L
            # Start with L={u}, R={v} and try to expand
            L = {u}
            R = {v}
            
            # Iteratively add vertices
            changed = True
            while changed:
                changed = False
                # Try adding to L
                for w in L_cand - L:
                    if all(w in adj[x] for x in R):
                        L.add(w)
                        changed = True
                # Try adding to R
                for w in R_cand - R:
                    if all(w in adj[x] for x in L):
                        R.add(w)
                        changed = True
            
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_biclique = (L, R)
        
        if best_biclique is None:
            # Fallback: single edge
            e = next(iter(remaining))
            u, v = tuple(e)
            best_biclique = ({u}, {v})
        
        L, R = best_biclique
        biclique_edges = _biclique_edges(L, R)
        # Verify all are in remaining
        biclique_edges = biclique_edges & remaining
        if not biclique_edges:
            e = next(iter(remaining))
            u, v = tuple(e)
            L, R = {u}, {v}
            biclique_edges = frozenset([frozenset((u, v))])
        
        remaining -= biclique_edges
        bicliques.append((list(L), list(R)))
    
    return bicliques

def _try_merge(bicliques, adj, deadline):
    """Try to merge pairs of bicliques if the union forms a valid biclique."""
    n = P['n']
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                # Try all 4 combinations of merging
                # (L1+L2, R1), (L1+L2, R2), (L1, R1+R2), (L2, R1+R2)
                # Also (L1+L2, R1+R2) if valid
                candidates = []
                L1s, L2s = set(L1), set(L2)
                R1s, R2s = set(R1), set(R2)
                
                # Check if merging L sides works
                Lm = L1s | L2s
                if _is_biclique(Lm, R1, adj):
                    candidates.append((list(Lm), list(R1)))
                if _is_biclique(Lm, R2, adj):
                    candidates.append((list(Lm), list(R2)))
                if _is_biclique(Lm, R1s | R2s, adj):
                    candidates.append((list(Lm), list(R1s | R2s)))
                
                # Check if merging R sides works
                Rm = R1s | R2s
                if _is_biclique(L1, Rm, adj):
                    candidates.append((list(L1), list(Rm)))
                if _is_biclique(L2, Rm, adj):
                    candidates.append((list(L2), list(Rm)))
                
                if candidates:
                    # Pick the candidate that covers the most edges
                    best = None
                    best_edges = -1
                    for Lc, Rc in candidates:
                        edges = len(Lc) * len(Rc)
                        if edges > best_edges:
                            best_edges = edges
                            best = (Lc, Rc)
                    bicliques[i] = best
                    bicliques.pop(j)
                    changed = True
                    break
            if changed:
                break
    return bicliques

def _try_split_and_reabsorb(bicliques, adj, deadline):
    """Try to split a biclique and reassign parts to others or create better ones."""
    n = P['n']
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            L, R = bicliques[i]
            # Try removing one vertex from L and seeing if it can be absorbed elsewhere
            for u in list(L):
                L_rem = [x for x in L if x != u]
                if not L_rem:
                    continue
                # Check if {u} can be absorbed into another biclique
                absorbed = False
                for j in range(len(bicliques)):
                    if j == i:
                        continue
                    Lj, Rj = bicliques[j]
                    if _is_biclique([u] + Lj, Rj, adj):
                        bicliques[j] = ([u] + Lj, Rj)
                        absorbed = True
                        break
                    if _is_biclique(Lj, [u] + Rj, adj):
                        bicliques[j] = (Lj, [u] + Rj)
                        absorbed = True
                        break
                if absorbed:
                    if L_rem and R:
                        bicliques[i] = (L_rem, R)
                    else:
                        bicliques.pop(i)
                        break
                    changed = True
                    break
            if changed:
                break
    return bicliques

def construct():
    start = time.time()
    deadline = start + 80.0  # 80s safety margin
    
    adj = _build_adj()
    
    # Initial greedy partition
    bicliques = _greedy_partition(adj, deadline)
    
    # Local search: merging
    bicliques = _try_merge(bicliques, adj, deadline)
    
    # Try split and reabsorb
    bicliques = _try_split_and_reabsorb(bicliques, adj, deadline)
    
    # Another round of merging after splits
    bicliques = _try_merge(bicliques, adj, deadline)
    
    # Convert to proper format: list of [left, right] with int vertices
    result = []
    for L, R in bicliques:
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        if L_int and R_int:
            result.append([L_int, R_int])
    
    return result
# EVOLVE-BLOCK-END
