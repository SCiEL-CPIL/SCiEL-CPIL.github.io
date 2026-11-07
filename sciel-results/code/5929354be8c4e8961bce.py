"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edges_to_set(edges):
    s = set()
    for u, v in edges:
        s.add((min(u, v), max(u, v)))
    return s

def _enumerate_maximal_bicliques(adj, n, deadline):
    """Enumerate maximal bicliques using Bron-Kerbosch on complement bipartite graph.
    A biclique [L,R] in G corresponds to an independent set in the complement bipartite graph
    where left side = all vertices, right side = all vertices, and edges exist for non-edges in G.
    """
    # Build complement adjacency (for the bipartite representation)
    # For vertex i on left, comp_adj[i] = set of vertices j on right where (i,j) is NOT an edge in G
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)
    
    bicliques = []
    
    def bron_kerbosch(P_set, X_set, R_set):
        if time.time() > deadline:
            return
        if not P_set and not X_set:
            if R_set:
                # R_set is a maximal independent set in complement bipartite graph
                # This means all vertices in R_set form a biclique in G
                # But we need to split into L and R such that all cross-pairs are edges
                # Actually, the independent set in the bipartite complement means
                # no two vertices in R_set are connected by a complement edge
                # i.e., for any i,j in R_set with i != j, (i,j) is an edge in G
                # So R_set itself forms a clique in G, which is a trivial biclique
                # We need a different approach
                
                # Let me reconsider: we want bicliques [L, R] where every (u,v) with u in L, v in R is an edge.
                # This is equivalent to: L and R are disjoint, and there are no complement edges between L and R.
                # So we're looking for pairs (L, R) such that L ∩ R = ∅ and no complement edge between L and R.
                # This is a bipartite independent set problem where we assign each vertex to L, R, or neither.
                
                bicliques.append(list(R_set))
            return
        if time.time() > deadline:
            return
        
        # Choose pivot to minimize branching
        pivot = None
        max_count = -1
        for v in (P_set | X_set):
            count = len(P_set & comp_adj[v])
            if count > max_count:
                max_count = count
                pivot = v
        
        if pivot is not None:
            candidates = P_set - comp_adj[pivot]
        else:
            candidates = P_set
        
        for v in list(candidates):
            new_P = P_set & comp_adj[v]
            new_X = X_set & comp_adj[v]
            new_R = R_set | {v}
            bron_kerbosch(new_P, new_X, new_R)
            P_set.discard(v)
            X_set.add(v)
            if time.time() > deadline:
                return
    
    # Actually, let me use a simpler approach: enumerate all maximal bicliques directly
    # A biclique is defined by (L, R) where L ∩ R = ∅ and all cross pairs are edges.
    # We can find maximal bicliques by starting from each vertex and expanding.
    
    bicliques = []
    seen = set()
    
    def find_biclique_from_seeds(seed_L, seed_R, adj, n):
        """Expand seed sets to maximal biclique."""
        L = set(seed_L)
        R = set(seed_R)
        if L & R:
            return None
        # Verify all cross edges exist
        for u in L:
            for v in R:
                if v not in adj[u]:
                    return None
        
        # Expand L: add vertex v if v is connected to all of R and v not in R
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if v in L or v in R:
                    continue
                if all(v in adj[w] for w in R):
                    L.add(v)
                    changed = True
        
        # Expand R: add vertex w if w is connected to all of L and w not in L
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w in L or w in R:
                    continue
                if all(w in adj[u] for u in L):
                    R.add(w)
                    changed = True
        
        # Final check
        if not L or not R:
            return None
        for u in L:
            for v in R:
                if v not in adj[u]:
                    return None
        
        key = (frozenset(L), frozenset(R))
        return (L, R)
    
    # Generate candidate bicliques from pairs of vertices
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                continue
            bic = find_biclique_from_seeds({i}, {j}, adj, n)
            if bic is not None:
                L, R = bic
                # Canonical form: smaller set as L
                if len(L) <= len(R):
                    key = (frozenset(L), frozenset(R))
                else:
                    key = (frozenset(R), frozenset(L))
                if key not in seen:
                    seen.add(key)
                    bicliques.append([list(L), list(R)])
    
    # Also try single-vertex bicliques (star-like)
    for i in range(n):
        neighbors = list(adj[i])
        if neighbors:
            bic = find_biclique_from_seeds({i}, set(neighbors), adj, n)
            if bic is not None:
                L, R = bic
                if len(L) <= len(R):
                    key = (frozenset(L), frozenset(R))
                else:
                    key = (frozenset(R), frozenset(L))
                if key not in seen:
                    seen.add(key)
                    bicliques.append([list(L), list(R)])
    
    return bicliques

def _solve_exact_cover(bicliques, edges_set, deadline):
    """Solve minimum biclique partition using Algorithm X with Dancing Links.
    We want to select minimum number of bicliques such that every edge is covered exactly once.
    This is a set cover / exact cover problem.
    """
    n = P['n']
    m = len(edges_set)
    
    # Build the incidence matrix: rows = bicliques, columns = edges
    # row r covers edge c if edge c is a cross-edge of biclique r
    
    # First, build edge index
    edge_idx = {e: i for i, e in enumerate(sorted(edges_set))}
    sorted_edges = sorted(edges_set)
    
    # For each biclique, compute which edges it covers
    biclique_edges = []
    for bic in bicliques:
        L, R = bic
        edges_covered = set()
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e in edges_set:
                    edges_covered.add(e)
        if edges_covered:
            biclique_edges.append(edges_covered)
    
    num_bics = len(biclique_edges)
    if num_bics == 0:
        return None
    
    # Build column headers (one per edge) and rows
    # Dancing Links structure
    # We'll use a simplified approach: greedy with backtracking
    
    # Sort bicliques by number of edges covered (descending) for better heuristics
    biclique_order = sorted(range(num_bics), key=lambda i: -len(biclique_edges[i]))
    
    # Try to find minimum cover using iterative deepening
    # Start with lower bound: max(1, ceil(m / max_biclique_size))
    max_size = max(len(be) for be in biclique_edges) if biclique_edges else 1
    lower_bound = max(1, (m + max_size - 1) // max_size)
    
    # Use branch and bound
    best_solution = None
    best_count = float('inf')
    
    def _search(selected, remaining_edges, depth, max_depth):
        nonlocal best_solution, best_count
        if time.time() > deadline:
            return
        if len(selected) >= best_count:
            return
        if not remaining_edges:
            if len(selected) < best_count:
                best_count = len(selected)
                best_solution = list(selected)
            return
        if depth >= max_depth:
            return
        
        # Choose the most constrained edge (appears in fewest remaining bicliques)
        # For simplicity, pick the first remaining edge
        edge = next(iter(remaining_edges))
        
        # Find all bicliques that cover this edge
        candidates = []
        for bi_idx in biclique_order:
            if bi_idx in selected:
                continue
            if edge in biclique_edges[bi_idx]:
                # Check if adding this biclique doesn't conflict
                # (i.e., all its edges are in remaining_edges)
                if biclique_edges[bi_idx] <= remaining_edges:
                    candidates.append(bi_idx)
        
        if not candidates:
            return
        
        # Sort candidates by coverage (most edges first)
        candidates.sort(key=lambda i: -len(biclique_edges[i]))
        
        for bi_idx in candidates:
            new_selected = selected + [bi_idx]
            new_remaining = remaining_edges - biclique_edges[bi_idx]
            _search(new_selected, new_remaining, depth + 1, max_depth)
    
    # Iterative deepening
    for max_depth in range(lower_bound, lower_bound + 10):
        _search([], set(edges_set), 0, max_depth)
        if best_solution is not None:
            break
    
    if best_solution is None:
        return None
    
    # Reconstruct bicliques
    result = []
    for bi_idx in best_solution:
        result.append(bicliques[bi_idx])
    return result

def _greedy_partition(bicliques, edges_set, deadline):
    """Greedy set cover: repeatedly pick biclique covering most uncovered edges."""
    remaining = set(edges_set)
    selected = []
    
    while remaining and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for i, bic in enumerate(bicliques):
            L, R = bic
            count = 0
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        count += 1
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx == -1 or best_count == 0:
            break
        selected.append(best_idx)
        L, R = bicliques[best_idx]
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                remaining.discard(e)
    
    # Add individual edges for any remaining
    for e in remaining:
        u, v = e
        selected.append(-1)  # marker for single edge
    
    # Reconstruct
    result = []
    for idx in selected:
        if idx == -1:
            # Find the edge
            pass
        else:
            result.append(bicliques[idx])
    
    # Handle remaining edges as single-edge bicliques
    # Actually let's redo this more carefully
    remaining = set(edges_set)
    result = []
    while remaining and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for i, bic in enumerate(bicliques):
            L, R = bic
            count = 0
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        count += 1
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx == -1 or best_count == 0:
            break
        result.append(bicliques[best_idx])
        L, R = bicliques[best_idx]
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                remaining.discard(e)
    
    for e in remaining:
        u, v = e
        result.append([[u], [v]])
    
    return result

def _validate(bicliques, edges_set):
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            for v in right:
                if u == v:
                    return False
                e = (min(u, v), max(u, v))
                if e not in edges_set:
                    return False
                if e in covered:
                    return False
                covered.add(e)
    return covered == edges_set

def construct():
    start_time = time.time()
    deadline = start_time + 80
    adj = _build_adj()
    edges_set = _edges_to_set(P['edges'])
    n = P['n']
    m = len(edges_set)
    
    # Star partition as fallback
    def star_partition():
        result = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                result.append([list(neighbors), [u]])
        return result
    
    # Enumerate candidate bicliques
    bicliques = _enumerate_maximal_bicliques(adj, n, deadline)
    
    # Try exact cover
    best = None
    if time.time() < deadline - 10 and bicliques:
        exact = _solve_exact_cover(bicliques, edges_set, deadline - 10)
        if exact is not None and _validate(exact, edges_set):
            best = exact
    
    # Greedy as backup
    if best is None and bicliques:
        greedy = _greedy_partition(bicliques, edges_set, deadline)
        if _validate(greedy, edges_set):
            best = greedy
    
    if best is None:
        best = star_partition()
    
    # Ensure types are correct
    result = []
    for left, right in best:
        result.append([list(left), list(right)])
    return result

# EVOLVE-BLOCK-END
