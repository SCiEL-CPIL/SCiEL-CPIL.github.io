"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

import sys
import time

sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
    
    m = len(edge_set)
    
    if m == 0:
        return []
    
    # Enumerate maximal bicliques by growing from each edge
    maximal_bicliques = []
    seen_bicliques = set()
    
    def grow_to_maximal(L, R):
        """Grow biclique (L, R) to a maximal one."""
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if x not in L and x not in R:
                    if all(x in adj[y] for y in R):
                        L.add(x)
                        changed = True
            for x in range(n):
                if x not in L and x not in R:
                    if all(x in adj[y] for y in L):
                        R.add(x)
                        changed = True
        return L, R
    
    for u, v in edges:
        L, R = grow_to_maximal({u}, {v})
        key = (frozenset(L), frozenset(R))
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            maximal_bicliques.append((L, R))
        
        L2, R2 = grow_to_maximal({v}, {u})
        key2 = (frozenset(L2), frozenset(R2))
        if key2 not in seen_bicliques:
            seen_bicliques.add(key2)
            maximal_bicliques.append((L2, R2))
    
    # Compute edges covered by each biclique
    biclique_edges = []
    for L, R in maximal_bicliques:
        edges_covered = set()
        for u in L:
            for v in R:
                e = frozenset((u, v))
                if e in edge_set:
                    edges_covered.add(e)
        if edges_covered:
            biclique_edges.append(edges_covered)
    
    # Build edge-to-biclique index
    edge_to_bicliques = {e: [] for e in edge_set}
    for i, be in enumerate(biclique_edges):
        for e in be:
            if e in edge_to_bicliques:
                edge_to_bicliques[e].append(i)
    
    # Greedy initialization
    remaining = set(edge_set)
    greedy_cover = []
    while remaining:
        best_bi = None
        best_ratio = 0
        for i, be in enumerate(biclique_edges):
            covered = len(be & remaining)
            if covered > 0:
                ratio = covered / (len(be) + 1)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_bi = i
        if best_bi is None:
            break
        greedy_cover.append(best_bi)
        remaining -= biclique_edges[best_bi]
    
    best_cover = greedy_cover[:]
    best_count = len(greedy_cover)
    
    # Backtracking with branch-and-bound
    start_time = time.time()
    time_limit = 50  # Leave 10s buffer
    
    def backtrack(remaining_edges, selected, count):
        nonlocal best_cover, best_count
        
        if time.time() - start_time > time_limit:
            return
        
        if not remaining_edges:
            if count < best_count:
                best_count = count
                best_cover = selected[:]
            return
        
        if count >= best_count:
            return
        
        # Pick edge with fewest covering bicliques
        min_count = float('inf')
        min_edge = None
        for e in remaining_edges:
            cnt = len(edge_to_bicliques[e])
            if cnt < min_count:
                min_count = cnt
                min_edge = e
        
        if min_count == 0:
            return
        
        # Sort candidates by coverage
        candidates = sorted(edge_to_bicliques[min_edge], key=lambda i: -len(biclique_edges[i] & remaining_edges))
        
        for bi in candidates:
            be = biclique_edges[bi]
            new_remaining = remaining_edges - be
            if len(new_remaining) < len(remaining_edges):
                selected.append(bi)
                backtrack(new_remaining, selected, count + 1)
                selected.pop()
                if time.time() - start_time > time_limit:
                    return
    
    backtrack(frozenset(edge_set), [], 0)
    
    # Convert to required format
    result = []
    for bi in best_cover:
        L, R = maximal_bicliques[bi]
        left = sorted(list(L))
        right = sorted(list(R))
        if left and right and not set(left) & set(right):
            result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
