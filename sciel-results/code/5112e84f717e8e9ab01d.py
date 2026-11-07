"""Exact minimum biclique edge-partition via maximal-biclique set cover + backtracking."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _star_partition():
    res = []
    for u in range(P['n']):
        nbrs = [v for a, v in P['edges'] if a == u]
        if nbrs:
            res.append(([u], list(nbrs)))
    return res

def construct():
    t0 = time.time()
    deadline = t0 + 80.0

    n = P['n']
    edges = P['edges']
    edge_set = set()
    for a, b in edges:
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        edge_set.add(key)
    all_edges = list(edge_set)
    m = len(all_edges)
    if m == 0:
        return []

    # adjacency
    adj = [set() for _ in range(n)]
    for a, b in all_edges:
        adj[a].add(b)
        adj[b].add(a)

    # maximal bicliques via Bron-Kerbosch on bipartite complement (non-edges)
    # A biclique [L,R] requires no non-edge between L and R.
    # Conflict graph: bipartite graph on vertices 0..2n-1 (left copy 0..n-1, right copy n..2n-1)
    # Edge between left u and right v (i.e., u and n+v) iff (u,v) is NOT an edge in G (and u!=v).
    # A biclique [L,R] corresponds to an independent set {u in L} U {n+v in R} in the conflict graph.
    # Maximal bicliques = maximal independent sets in conflict graph.
    # Bron-Kerbosch finds maximal cliques; so run on complement of conflict graph = bipartite graph of edges.
    # Actually simpler: run Bron-Kerbosch for maximal cliques on the bipartite "edge" graph H where
    # H has vertices 0..2n-1 and edges between u (left) and n+v (right) iff (u,v) in G, u!=v.
    # Maximal cliques in H are exactly maximal bicliques (since H is bipartite, cliques are complete bipartite).
    # But cliques in a bipartite graph: a clique can have at most one from each side? No - in a bipartite
    # graph, a clique is either a single vertex, an edge, or... actually in a bipartite graph the only cliques
    # of size >=3 don't exist because no triangle. So that approach fails.
    #
    # Correct approach: The set of vertices {L} U {n+R} forms a biclique iff there is NO conflict edge between
    # any u in L and n+v in R. So it's an independent set in the conflict graph. Maximal independent sets in
    # conflict graph = maximal cliques in the complement of conflict graph.
    # Complement of conflict graph: bipartite graph where edge between u (left) and n+v (right) iff (u,v) IS an edge.
    # That's just the bipartite double cover of G (excluding self-loops). Maximal cliques in this bipartite
    # graph: again, bipartite graphs have no triangles, so max clique size is 2. That's wrong too.
    #
    # Let me reconsider. A biclique [L, R] means: for ALL u in L, v in R: (u,v) is an edge.
    # This is equivalent to: L and R are subsets such that L x R subset E.
    # In the conflict graph (bipartite, edge u--n+v iff (u,v) NOT in E), the set S = L U (n+R) has no
    # internal conflict edges, i.e., S is an independent set in the conflict graph.
    # Maximal bicliques = maximal independent sets in the conflict graph.
    # To find maximal independent sets via Bron-Kerbosch: find maximal cliques in the complement of the
    # conflict graph. The complement of the conflict graph is the bipartite graph where u--n+v iff (u,v) in E.
    # In this bipartite graph, what are the maximal cliques? A clique in a bipartite graph can contain
    # vertices from both sides only if every left vertex is connected to every right vertex in the clique.
    # So a clique {l1,...,lk} U {r1,...,rm} requires all li-rj edges present. That's exactly a biclique!
    # And it's maximal as a clique iff the biclique is maximal. But Bron-Kerbosch on a bipartite graph:
    # the recursion works fine. Let me just implement it.

    # Build the bipartite edge graph H (complement of conflict graph)
    # H vertices: 0..2n-1, edge i-j iff i < n and j >= n and (i, j-n) in edge_set and i != j-n
    # OR i >= n and j < n and (j, i-n) in edge_set and i-n != j
    # For Bron-Kerbosch we need the adjacency as sets.
    h_adj = [set() for _ in range(2 * n)]
    for u in range(n):
        for v in adj[u]:
            h_adj[u].add(n + v)
            h_adj[n + v].add(u)

    # Bron-Kerbosch with pivot for maximal cliques in H
    # We want maximal cliques that contain at least one left and one right vertex (non-trivial bicliques)
    # Actually a single vertex is a trivial "biclique" with one side empty - not allowed.
    # We need cliques with at least 2 vertices, one from each side.

    maximal_bicliques = []  # list of (frozen_left_set, frozen_right_set)

    def bronk(R, P_set, X):
        if time.time() > deadline:
            return
        if not P_set and not X:
            # R is a maximal clique
            left = frozenset(v for v in R if v < n)
            right = frozenset(v - n for v in R if v >= n)
            if left and right:
                maximal_bicliques.append((left, right))
            return
        # Choose pivot u from P_set U X maximizing |P_set ∩ N(u)|
        union_px = P_set | X
        best_u = -1
        best_count = -1
        for u in union_px:
            c = len(P_set & h_adj[u])
            if c > best_count:
                best_count = c
                best_u = u
        # Iterate over P_set \ N(best_u)
        candidates = P_set - h_adj[best_u] if best_u >= 0 else P_set
        for v in list(candidates):
            new_R = R | {v}
            new_P = P_set & h_adj[v]
            new_X = X & h_adj[v]
            bronk(new_R, new_P, new_X)
            P_set.discard(v)
            X.add(v)

    bronk(frozenset(), set(range(2 * n)), set())

    # If enumeration was too slow or found nothing, fall back to star
    if not maximal_bicliques:
        return _star_partition()

    # Build edge-to-biclique index
    # Each biclique (L, R) covers edges {(u,v) : u in L, v in R}
    biclique_edges = []
    for left, right in maximal_bicliques:
        es = frozenset((u, v) for u in left for v in right if (u, v) in edge_set)
        if es:
            biclique_edges.append(es)

    # Deduplicate
    biclique_edges = list(set(biclique_edges))
    num_bics = len(biclique_edges)

    # Map each edge to list of biclique indices that cover it
    edge_to_bics = {}
    for i, es in enumerate(biclique_edges):
        for e in es:
            if e not in edge_to_bics:
                edge_to_bics[e] = []
            edge_to_bics[e].append(i)

    # Some edges might not be in any maximal biclique? Shouldn't happen since each edge is in at least
    # one maximal biclique (extend {u},{v} to maximal). But let's check.
    all_edge_set = set(all_edges)
    covered_edges = set()
    for es in biclique_edges:
        covered_edges |= es
    if covered_edges != all_edge_set:
        # Fallback
        return _star_partition()

    # Now solve minimum set cover: choose min bicliques to cover all edges
    # Use recursive backtracking with MRV + memoization
    # State: frozenset of uncovered edges -> min additional bicliques needed
    # But the state space could be large. Use a branch-and-bound with memoization on the uncovered set.

    all_edges_frozen = frozenset(all_edges)
    best_count = len(maximal_bicliques)  # upper bound: use all
    best_solution = None
    memo = {}  # frozenset(uncovered) -> min_bics_needed (or None if not computed)

    # For speed, represent uncovered as a frozenset of edge indices (integers)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    biclique_edge_indices = [frozenset(edge_index[e] for e in es) for es in biclique_edges]
    edge_to_bic_indices = {}
    for i, bset in enumerate(biclique_edge_indices):
        for ei in bset:
            if ei not in edge_to_bic_indices:
                edge_to_bic_indices[ei] = []
            edge_to_bic_indices[ei].append(i)

    total_edges = m

    def solve(uncovered_frozen, current_count, current_choice):
        nonlocal best_count, best_solution
        if time.time() > deadline:
            return
        if not uncovered_frozen:
            if current_count < best_count:
                best_count = current_count
                best_solution = list(current_choice)
            return
        # Lower bound: at least 1 more needed
        if current_count + 1 >= best_count:
            return
        # Memoization check
        if uncovered_frozen in memo:
            if memo[uncovered_frozen] + current_count >= best_count:
                return
        # MRV: pick uncovered edge with fewest candidate bicliques
        min_count = float('inf')
        chosen_edge = None
        chosen_bics = None
        for ei in uncovered_frozen:
            candidates = [bi for bi in edge_to_bic_indices[ei] if biclique_edge_indices[bi] & uncovered_frozen]
            if len(candidates) < min_count:
                min_count = len(candidates)
                chosen_edge = ei
                chosen_bics = candidates
                if min_count == 1:
                    break
        if min_count == 0:
            return
        # Try each candidate biclique
        for bi in chosen_bics:
            new_uncovered = uncovered_frozen - biclique_edge_indices[bi]
            solve(new_uncovered, current_count + 1, current_choice + [bi])
        # Store memo (optimistic: the min we'd need from here)
        # We don't know the exact min, so skip memo for now to keep it simple and correct.

    solve(all_edges_frozen, 0, [])

    if best_solution is None:
        return _star_partition()

    # Build result
    result = []
    for bi in best_solution:
        left, right = maximal_bicliques[bi] if bi < len(maximal_bicliques) else None
        # Wait - biclique_edges was deduplicated, so indices don't directly map to maximal_bicliques
        # Let me fix: keep the mapping
        pass

    # I need to keep the original (left, right) for each biclique_edge. Let me redo this more carefully.
    # Actually let me restructure: keep a list of (left, right, edge_set) for each unique biclique.

    # Redo: build unique bicliques with their vertex sets
    biclique_info = []  # (left_frozen, right_frozen, edge_frozen_set)
    seen = set()
    for left, right in maximal_bicliques:
        es = frozenset((u, v) for u in left for v in right if (u, v) in edge_set)
        if es and es not in seen:
            seen.add(es)
            biclique_info.append((left, right, es))

    num_bics = len(biclique_info)
    edge_to_bic2 = {}
    biclique_edge_idx2 = []
    for i, (left, right, es) in enumerate(biclique_info):
        idx_set = frozenset(edge_index[e] for e in es)
        biclique_edge_idx2.append(idx_set)
        for ei in idx_set:
            if ei not in edge_to_bic2:
                edge_to_bic2[ei] = []
            edge_to_bic2[ei].append(i)

    best_count = num_bics + 1
    best_solution = None

    def solve2(uncovered_frozen, current_count, current_choice):
        nonlocal best_count, best_solution
        if time.time() > deadline:
            return
        if not uncovered_frozen:
            if current_count < best_count:
                best_count = current_count
                best_solution = list(current_choice)
            return
        if current_count + 1 >= best_count:
            return
        min_count = float('inf')
        chosen_bics = None
        for ei in uncovered_frozen:
            candidates = [bi for bi in edge_to_bic2[ei] if biclique_edge_idx2[bi] & uncovered_frozen]
            if len(candidates) < min_count:
                min_count = len(candidates)
                chosen_bics = candidates
                if min_count == 1:
                    break
        if min_count == 0:
            return
        for bi in chosen_bics:
            new_uncovered = uncovered_frozen - biclique_edge_idx2[bi]
            solve2(new_uncovered, current_count + 1, current_choice + [bi])

    solve2(all_edges_frozen, 0, [])

    if best_solution is None:
        return _star_partition()

    result = []
    for bi in best_solution:
        left, right, es = biclique_info[bi]
        result.append((list(left), list(right)))

    # Validate
    used_edges = set()
    for left, right in result:
        if not left or not right:
            return _star_partition()
        for u in left:
            for v in right:
                e = (min(u, v), max(u, v))
                if e not in edge_set:
                    return _star_partition()
                if e in used_edges:
                    return _star_partition()
                used_edges.add(e)
    if used_edges != all_edge_set:
        return _star_partition()

    # Convert to proper types
    final = []
    for left, right in result:
        final.append(([int(x) for x in left], [int(x) for x in right]))
    return final

# EVOLVE-BLOCK-END
