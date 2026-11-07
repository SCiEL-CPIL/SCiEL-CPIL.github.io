"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75

    n = P['n']
    edges = P['edges']

    # Build adjacency and edge sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    all_edges = list(edge_set)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    m = len(all_edges)

    # For each vertex, precompute which edges it touches
    vertex_edges = [set() for _ in range(n)]
    for u, v in all_edges:
        vertex_edges[u].add((u, v))
        vertex_edges[v].add((u, v))

    # Build complement adjacency (non-edges) for biclique finding
    # In a biclique (L,R), no two vertices in L can be non-adjacent (they must both connect to all of R)
    # Actually: for a biclique, every u in L must be adjacent to every v in R.
    # So within L, vertices don't need to be adjacent to each other.
    # The constraint is: for all u in L, v in R: (u,v) is an edge.
    #
    # To find maximal bicliques, we can use a bipartite approach:
    # A biclique is defined by (L, R) where L and R are disjoint and all cross-edges exist.
    # Two vertices u, v in L are "compatible" if they share at least one common neighbor in R.
    # Actually, for maximal biclique enumeration, let's use a different approach.

    # Approach: Enumerate all maximal bicliques using a variant of Bron-Kerbosch.
    # A biclique (L,R) is maximal if no vertex can be added to L or R while maintaining the property.
    #
    # Key insight: For a fixed set R, the maximal L is N(R) = {u : u adjacent to all v in R}.
    # For a fixed set L, the maximal R is N(L) = {v : v adjacent to all u in L}.
    #
    # We'll enumerate by trying all subsets of a small side.
    # Since n=30, we can try all pairs/triples on one side.

    # Better approach: Use the fact that for any biclique, if we fix one vertex on each side,
    # the biclique is determined by common neighbors.
    #
    # Let's enumerate maximal bicliques by:
    # 1. For each edge (u,v), the maximal biclique containing it with u on left and v on right
    #    has L = {u} ∪ (common neighbors of all in R that are not in R)
    #    R = {v} ∪ (common neighbors of all in L that are not in L)
    #    Iterate until fixed point.
    # 2. Deduplicate.

    # Precompute: for each pair of vertices, their common neighbors
    common_neighbors = {}
    for u in range(n):
        for v in range(n):
            if u != v:
                common_neighbors[(u, v)] = adj[u] & adj[v]

    # For each subset of vertices of size 1, 2, 3 on the "left" side,
    # find the maximal right side, then extend left.
    # This is O(n^3 * n^2) which is manageable for n=30.

    def extend_biclique(L, R):
        """Extend (L, R) to a maximal biclique."""
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            # Extend L: add vertices adjacent to all in R
            if R:
                new_L = set()
                for v in R:
                    new_L |= adj[v]
                new_L = new_L - R
                new_L = {u for u in new_L if all(u in adj[v] for v in R)}
                for u in new_L:
                    if u not in L:
                        L.add(u)
                        changed = True
            # Extend R: add vertices adjacent to all in L
            if L:
                new_R = set()
                for u in L:
                    new_R |= adj[u]
                new_R = new_R - L
                new_R = {v for v in new_R if all(v in adj[u] for u in L)}
                for v in new_R:
                    if v not in R:
                        R.add(v)
                        changed = True
        return L, R

    # Enumerate candidate bicliques by fixing small left sets
    biclique_set = set()  # frozenset of (L_frozen, R_frozen)

    # Try all single vertices on left
    for u in range(n):
        if time.time() > deadline:
            break
        for v in adj[u]:
            L, R = extend_biclique({u}, {v})
            biclique_set.add((frozenset(L), frozenset(R)))

    # Try all pairs on left
    for u in range(n):
        for v in range(u + 1, n):
            if time.time() > deadline:
                break
            # Common neighbors of u and v
            cn = adj[u] & adj[v]
            if not cn:
                continue
            for w in cn:
                L, R = extend_biclique({u, v}, {w})
                biclique_set.add((frozenset(L), frozenset(R)))

    # Try all triples on left
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                if time.time() > deadline:
                    break
                cn = adj[u] & adj[v] & adj[w]
                if not cn:
                    continue
                for x in cn:
                    L, R = extend_biclique({u, v, w}, {x})
                    biclique_set.add((frozenset(L), frozenset(R)))

    # Also try fixing small right sets (symmetric)
    for v in range(n):
        if time.time() > deadline:
            break
        for u in adj[v]:
            L, R = extend_biclique({u}, {v})
            biclique_set.add((frozenset(L), frozenset(R)))

    # Convert to list and compute edge coverage for each biclique
    bicliques = []
    biclique_edges = []  # bitmask or set of edges covered

    for L_f, R_f in biclique_set:
        if not L_f or not R_f:
            continue
        # Compute edges covered
        covered = set()
        for u in L_f:
            for v in R_f:
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    covered.add(key)
        if covered:
            bicliques.append((list(L_f), list(R_f)))
            biclique_edges.append(frozenset(covered))

    # Also add trivial bicliques (single edges) as fallback
    for e in all_edges:
        u, v = e
        bicliques.append([u], [v])
        biclique_edges.append(frozenset([e]))

    num_bicliques = len(bicliques)

    # Build edge-to-biclique mapping
    edge_to_bics = [set() for _ in range(m)]
    for i in range(num_bicliques):
        for e in biclique_edges[i]:
            idx = edge_index[e]
            edge_to_bics[idx].add(i)

    # Now solve minimum set cover: select minimum bicliques to cover all edges
    # Use backtracking with memoization

    all_edges_mask = (1 << m) - 1  # Only valid if m <= 63, but m can be up to ~435

    # Since m can be large, use a different representation
    # Use a list of frozensets for remaining edges

    # Precompute biclique edge sets as frozensets for fast operations
    biclique_edge_sets = []
    for i in range(num_bicliques):
        biclique_edge_sets.append(frozenset(edge_index[e] for e in biclique_edges[i]))

    # Remove bicliques that are subsets of others (dominated)
    # A biclique i is dominated by j if edges(i) ⊆ edges(j)
    # This can significantly reduce the search space
    dominated = [False] * num_bicliques
    for i in range(num_bicliques):
        if dominated[i]:
            continue
        for j in range(i + 1, num_bicliques):
            if dominated[j]:
                continue
            if biclique_edge_sets[i] <= biclique_edge_sets[j]:
                dominated[i] = True
                break
            elif biclique_edge_sets[j] <= biclique_edge_sets[i]:
                dominated[j] = True

    # Filter out dominated bicliques
    kept = [i for i in range(num_bicliques) if not dominated[i]]
    kept_bicliques = [bicliques[i] for i in kept]
    kept_edge_sets = [biclique_edge_sets[i] for i in kept]
    num_kept = len(kept)

    # Rebuild edge-to-biclique mapping
    edge_to_bics2 = [set() for _ in range(m)]
    for i, i_orig in enumerate(kept):
        for e_idx in kept_edge_sets[i]:
            edge_to_bics2[e_idx].add(i)

    # Backtracking search for minimum set cover
    best_cover = None
    best_count = m + 1  # Upper bound: one biclique per edge

    # Sort edges by number of covering bicliques (most constrained first)
    # For the search, we'll pick the uncovered edge with fewest options

    def solve(remaining_edges, selected, depth):
        nonlocal best_cover, best_count

        if time.time() > deadline:
            return

        if not remaining_edges:
            if len(selected) < best_count:
                best_count = len(selected)
                best_cover = list(selected)
            return

        # Lower bound: at least 1 more needed
        if depth >= best_count:
            return

        # Pick the edge with fewest covering bicliques
        min_count = float('inf')
        min_edge = None
        for e in remaining_edges:
            cnt = len(edge_to_bics2[e])
            if cnt < min_count:
                min_count = cnt
                min_edge = e
            if cnt == 1:
                break

        if min_count == 0:
            return  # Can't cover this edge

        # Branch on bicliques that cover min_edge
        candidates = list(edge_to_bics2[min_edge])

        # Sort candidates by coverage size (larger first)
        candidates.sort(key=lambda i: -len(kept_edge_sets[i]))

        for bic_idx in candidates:
            new_selected = selected + [bic_idx]
            new_remaining = remaining_edges - kept_edge_sets[bic_idx]

            if len(new_selected) >= best_count:
                continue

            solve(frozenset(new_remaining), new_selected, depth + 1)

    # Initial remaining edges
    initial_remaining = frozenset(range(m))

    # First, try to find a good upper bound with greedy
    def greedy_cover():
        remaining = set(range(m))
        selected = []
        while remaining:
            best_bic = None
            best_cov = 0
            for i in range(num_kept):
                cov = len(remaining & kept_edge_sets[i])
                if cov > best_cov:
                    best_cov = cov
                    best_bic = i
            if best_bic is None or best_cov == 0:
                break
            selected.append(best_bic)
            remaining -= kept_edge_sets[best_bic]
        return selected

    greedy_sol = greedy_cover()
    if greedy_sol:
        best_count = len(greedy_sol)
        best_cover = greedy_sol

    # Now try to improve with backtracking
    # Limit the depth of search
    solve(initial_remaining, [], 0)

    # Build result from best_cover
    if best_cover is not None:
        result = []
        for bic_idx in best_cover:
            L, R = kept_bicliques[bic_idx]
            L = [int(x) for x in L]
            R = [int(x) for x in R]
            if L and R:
                result.append([L, R])
    else:
        # Fallback: one biclique per edge
        result = []
        for u, v in all_edges:
            result.append([[int(u)], [int(v)]])

    # Validate the result
    def validate(result):
        covered = set()
        for L, R in result:
            if not L or not R:
                return False
            for u in L:
                for v in R:
                    key = (min(u, v), max(u, v))
                    if key in covered:
                        return False
                    if key not in edge_set:
                        return False
                    covered.add(key)
        return covered == edge_set

    if not validate(result):
        # Fallback: one biclique per edge
        result = []
        for u, v in all_edges:
            result.append([[int(u)], [int(v)]])

    return result
# EVOLVE-BLOCK-END
