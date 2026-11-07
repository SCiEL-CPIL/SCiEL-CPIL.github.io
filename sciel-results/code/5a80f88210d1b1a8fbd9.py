import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _edge_key(u, v):
    return (min(u, v), max(u, v))


def _find_maximal_bicliques(adj, n, deadline):
    """
    Find candidate bicliques by trying all pairs of vertices as seeds
    and expanding greedily. Returns list of (frozenset(L), frozenset(R), edge_set).
    """
    bicliques = []
    seen = set()
    
    for u in range(n):
        if time.time() > deadline:
            break
        for v in range(n):
            if u == v or time.time() > deadline:
                continue
            key = (u, v)
            if key in seen:
                continue
            
            # Try both orientations
            for L_init, R_init in [({u}, {v}), ({v}, {u})]:
                L = set(L_init)
                R = set(R_init)
                
                # Expand: add vertices that connect to all on the other side
                changed = True
                while changed:
                    changed = False
                    # Try adding to L
                    for w in range(n):
                        if w not in L and w not in R:
                            if all(w in adj[r] for r in R):
                                L.add(w)
                                changed = True
                    # Try adding to R
                    for w in range(n):
                        if w not in R and w not in L:
                            if all(w in adj[l] for l in L):
                                R.add(w)
                                changed = True
                
                if len(L) > 0 and len(R) > 0:
                    # Compute edge set
                    edge_set = frozenset(
                        _edge_key(l, r) for l in L for r in R
                    )
                    if edge_set not in seen:
                        seen.add(edge_set)
                        bicliques.append((frozenset(L), frozenset(R), edge_set))
                        # Also mark the reverse orientation as seen
                        seen.add(edge_set)
    
    return bicliques


def _greedy_set_cover(uncovered_edges, candidate_bicliques, adj, n, deadline):
    """
    Greedy set cover: repeatedly pick the biclique that covers the most
    uncovered edges.
    """
    uncovered = set(uncovered_edges)
    selected = []
    
    while uncovered and time.time() < deadline:
        best_biclique = None
        best_count = 0
        best_cover = None
        
        for L, R, edge_set in candidate_bicliques:
            # Count how many uncovered edges this biclique covers
            cover = edge_set & uncovered
            count = len(cover)
            if count > best_count:
                best_count = count
                best_biclique = (L, R, edge_set)
                best_cover = cover
        
        if best_count == 0:
            # No candidate covers any remaining edge; add single-edge bicliques
            break
        
        selected.append(best_biclique)
        uncovered -= best_cover
    
    # Handle any remaining uncovered edges
    if uncovered:
        for e in uncovered:
            u, v = e
            selected.append((frozenset({u}), frozenset({v}), frozenset({e})))
    
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for the original graph
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # All edges as frozensets
    all_edges = set()
    for u, v in edges:
        all_edges.add(frozenset(_edge_key(u, v)))
    
    # Phase 1: Find candidate maximal bicliques
    # Use a time budget for this phase
    biclique_deadline = start_time + 30.0
    candidate_bicliques = _find_maximal_bicliques(adj, n, biclique_deadline)
    
    # Phase 2: Greedy set cover
    cover_deadline = start_time + 50.0
    selected = _greedy_set_cover(all_edges, candidate_bicliques, adj, n, cover_deadline)
    
    # Build result
    bicliques = []
    for L, R, edge_set in selected:
        bicliques.append([list(L), list(R)])
    
    return bicliques
