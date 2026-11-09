"""Biclique partition minimization for a random graph."""
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    adj = [set() for _ in range(P['n'])]
    edge_set = set()
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
        edge_set.add((min(a,b), max(a,b)))
    return adj, edge_set

def _is_valid_biclique_partition(bicliques, edge_set, adj):
    """Check that cross-edge sets partition all edges exactly once, all cross pairs are edges, sides disjoint nonempty."""
    covered = {}
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left) & set(right):
            return False
        for u in left:
            for v in right:
                key = (min(u,v), max(u,v))
                if key not in adj[u]:
                    return False
                covered[key] = covered.get(key, 0) + 1
    if len(covered) != len(edge_set):
        return False
    for k in edge_set:
        if covered.get(k, 0) != 1:
            return False
    return True

def _greedy_biclique_partition(edge_set, adj, deadline):
    """Greedy: repeatedly find a large biclique from remaining edges, remove those edges."""
    remaining = list(edge_set)
    remaining_set = set(edge_set)
    bicliques = []
    
    def find_best_biclique(edges_list):
        """Find a biclique covering many of the given edges."""
        if not edges_list:
            return None, 0
        best = None
        best_count = 0
        # Try each edge as a seed, then grow
        for (u, v) in edges_list[:50]:  # limit seeds for speed
            # Start with L={u}, R={v}
            L = {u}
            R = {v}
            # Grow: add vertices that connect to all on the other side
            changed = True
            while changed:
                changed = False
                # Try adding to L: vertex x not in L such that x connects to all in R and (x,r) in remaining_set for all r in R
                for x in range(P['n']):
                    if x in L or x in R:
                        continue
                    if all((min(x,r), max(x,r)) in remaining_set for r in R):
                        L.add(x)
                        changed = True
                        break
                if changed:
                    continue
                # Try adding to R
                for x in range(P['n']):
                    if x in L or x in R:
                        continue
                    if all((min(l,x), max(l,x)) in remaining_set for l in L):
                        R.add(x)
                        changed = True
                        break
            count = 0
            for l in L:
                for r in R:
                    if (min(l,r), max(l,r)) in remaining_set:
                        count += 1
            if count > best_count:
                best_count = count
                best = (L, R)
        return best, best_count
    
    while remaining_set and time.time() < deadline:
        bl, count = find_best_biclique(list(remaining_set))
        if bl is None or count == 0:
            # Fallback: single edge
            u, v = next(iter(remaining_set))
            bl = ({u}, {v})
            count = 1
        L, R = bl
        bicliques.append((sorted(L), sorted(R)))
        # Remove covered edges
        for l in L:
            for r in R:
                key = (min(l,r), max(l,r))
                remaining_set.discard(key)
    
    # Cover any leftover edges as single-edge bicliques
    for (u, v) in remaining_set:
        bicliques.append(([u], [v]))
    
    return bicliques

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill
    
    adj, edge_set = _build_adjacency()
    
    # Strategy 1: Greedy biclique partition
    bicliques = _greedy_biclique_partition(edge_set, adj, deadline)
    
    # Verify
    if not _is_valid_biclique_partition(bicliques, edge_set, adj):
        # Fallback: one biclique per edge
        bicliques = [([u], [v]) for (u, v) in edge_set]
    
    # Convert to required format: list of [left, right] with ints
    result = []
    for left, right in bicliques:
        result.append([list(left), list(right)])
    
    return result
# EVOLVE-BLOCK-END
