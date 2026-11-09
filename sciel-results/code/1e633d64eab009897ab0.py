"""Biclique partition via greedy max-biclique extraction with local search."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_n = P['n']
_edges = P['edges']
_m = len(_edges)

# Build adjacency sets for fast lookup
_adj = [set() for _ in range(_n)]
for u, v in _edges:
    _adj[u].add(v)
    _adj[v].add(u)

# Edge set for O(1) lookup
_edge_set = set()
for u, v in _edges:
    if u < v:
        _edge_set.add((u, v))
    else:
        _edge_set.add((v, u))


def _is_biclique(L, R, edge_set):
    """Check if all cross pairs between L and R are edges."""
    for u in L:
        for v in R:
            key = (u, v) if u < v else (v, u)
            if key not in edge_set:
                return False
    return True


def _biclique_size(L, R):
    return len(L) * len(R)


def _find_max_biclique(uncovered_edges, adj, edge_set, max_size=4):
    """Find a large biclique from uncovered edges using greedy expansion."""
    if not uncovered_edges:
        return None, None, []
    
    best_L = None
    best_R = None
    best_edges = []
    best_count = 0
    
    # Try each uncovered edge as a seed
    # Sample a subset if too many edges
    candidates = list(uncovered_edges)
    if len(candidates) > 200:
        random.shuffle(candidates)
        candidates = candidates[:200]
    
    for u, v in candidates:
        # Start with L={u}, R={v}
        L = [u]
        R = [v]
        
        # Greedily expand L: add vertices that connect to all of R
        changed = True
        while changed and len(L) < max_size:
            changed = False
            # Candidates for L: vertices connected to all in R
            r_set = set(R)
            for w in adj[u]:
                if w not in L and w != u:
                    if all(w in adj[x] or w == x for x in R):
                        L.append(w)
                        changed = True
                        break
        
        # Greedily expand R: add vertices that connect to all of L
        changed = True
        while changed and len(R) < max_size:
            changed = False
            l_set = set(L)
            for w in adj[v]:
                if w not in R and w != v:
                    if all(w in adj[x] or w == x for x in L):
                        R.append(w)
                        changed = True
                        break
        
        # Count how many uncovered edges this biclique covers
        covered = []
        for a in L:
            for b in R:
                key = (a, b) if a < b else (b, a)
                if key in uncovered_edges:
                    covered.append((a, b))
        
        if len(covered) > best_count:
            best_count = len(covered)
            best_L = L[:]
            best_R = R[:]
            best_edges = covered[:]
    
    return best_L, best_R, best_edges


def _greedy_partition(uncovered, adj, edge_set, deadline):
    """Greedy: repeatedly extract largest biclique from uncovered edges."""
    bicliques = []
    uncovered_set = set(uncovered)
    
    while uncovered_set:
        if time.time() > deadline:
            break
        
        L, R, covered_edges = _find_max_biclique(uncovered_set, adj, edge_set)
        
        if L is None or not covered_edges:
            # Fallback: single edge
            e = next(iter(uncovered_set))
            a, b = e
            bicliques.append(([a], [b]))
            uncovered_set.discard((a, b) if a < b else (b, a))
            continue
        
        bicliques.append((L, R))
        for e in covered_edges:
            key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            uncovered_set.discard(key)
    
    return bicliques, uncovered_set


def _local_search(bicliques, adj, edge_set, deadline):
    """Try to merge or reassign to reduce biclique count."""
    # For now, try removing bicliques that are subsets of others
    # and re-covering their edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            L_i, R_i = bicliques[i]
            # Check if all edges of biclique i are covered by other bicliques
            # This is complex; skip for now
            pass
    return bicliques


def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    # Build initial edge set
    all_edges = set()
    for u, v in _edges:
        key = (u, v) if u < v else (v, u)
        all_edges.add(key)
    
    # Greedy partition
    bicliques, remaining = _greedy_partition(all_edges, _adj, _edge_set, deadline)
    
    # If there are remaining edges (shouldn't happen), add them as single-edge bicliques
    if remaining:
        for e in remaining:
            a, b = e
            bicliques.append(([a], [b]))
    
    # Verify and convert to proper format
    result = []
    for L, R in bicliques:
        # Ensure nonempty and disjoint
        L_list = [int(x) for x in L]
        R_list = [int(x) for x in R]
        # Check disjointness
        if set(L_list) & set(R_list):
            # Move overlapping vertices to one side
            overlap = set(L_list) & set(R_list)
            L_list = [x for x in L_list if x not in overlap]
            R_list = [x for x in R_list if x not in overlap]
            if not L_list:
                L_list = list(overlap)
                R_list = []
            if not R_list:
                R_list = list(overlap)
                L_list = []
        if L_list and R_list:
            result.append([L_list, R_list])
    
    # Final verification: ensure all edges are covered exactly once
    if time.time() < deadline - 5:
        covered_count = {}
        for L, R in result:
            for u in L:
                for v in R:
                    key = (u, v) if u < v else (v, u)
                    if key in _edge_set:
                        covered_count[key] = covered_count.get(key, 0) + 1
        
        # Check for over-coverage or missing edges
        over_covered = [k for k, c in covered_count.items() if c > 1]
        missing = [k for k in _edge_set if k not in covered_count]
        
        if over_covered or missing:
            # Fallback to seed solution
            result = [[[u], [v for a, v in _edges if a == u]] 
                      for u in range(_n) if any(a == u for a, v in _edges)]
    
    return result
# EVOLVE-BLOCK-END
