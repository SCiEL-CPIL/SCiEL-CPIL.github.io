import time
from random import Random
from collections import deque

def construct():
    _T0 = time.time()
    _DEADLINE = 140.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _min_clique_cover_complement(adj, k):
        """Minimum clique cover of graph (k vertices, adj) = min coloring of complement.
        Uses greedy with DSATUR ordering. Returns list of cliques (each a list of vertices)."""
        if k <= 1:
            return []
        # Build complement adjacency
        comp = [set(range(k)) for _ in range(k)]
        for s in range(k):
            comp[s].discard(s)
            for w in adj[s]:
                comp[s].discard(w)
        
        # DSATUR greedy coloring
        colors = [-1] * k
        saturation = [0] * k  # number of different colors in neighbors
        sat_colors = [set() for _ in range(k)]
        degree = [len(comp[s]) for s in range(k)]
        
        # Order: pick uncolored vertex with max saturation, then max degree
        uncolored = set(range(k))
        for _ in range(k):
            # Pick vertex with max (saturation, degree)
            best_v = None
            best_key = None
            for v in uncolored:
                key = (len(sat_colors[v]), degree[v])
                if best_key is None or key > best_key:
                    best_key = key
                    best_v = v
            uncolored.discard(best_v)
            
            # Assign smallest available color
            c = 0
            while c in sat_colors[best_v]:
                c += 1
            colors[best_v] = c
            
            # Update saturation of neighbors
            for w in comp[best_v]:
                if w in uncolored and c not in sat_colors[w]:
                    sat_colors[w].add(c)
        
        # Build cliques from color classes
        max_c = max(colors) if colors else 0
        cliques = [[] for _ in range(max_c + 1)]
        for s in range(k):
            cliques[colors[s]].append(s)
        return [c for c in cliques if len(c) >= 2]

    def _solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # C vertices: 0..k-1, I vertices: k..n-1
        # C-C edges: all pairs in C (complete graph)
        # Cross edges: (u, v) where u < k <= v

        # Build N_C[v] for each I-vertex v
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)

        cross_edges = sorted((u, v) for (u, v) in edge_set if u >= k or v >= k)
        # Normalize: (c, i) where c < k, i >= k
        cross_edges_norm = []
        for (a, b) in cross_edges:
            if a < k and b >= k:
                cross_edges_norm.append((a, b))
            elif b < k and a >= k:
                cross_edges_norm.append((b, a))
        
        # Regime 1 baseline: 1 C-only clique + per-edge 2-cliques
        base_cliques = [list(range(k))]
        for (u, v) in cross_edges_norm:
            base_cliques.append(sorted([u, v]))
        best_cliques = base_cliques
        best_count = len(base_cliques)

        # Regime 2: All C-C edges covered by I-vertex cliques.
        # For each I-vertex v, its cliques are {v} ∪ S where S is a clique in C
        # (all vertices in S are pairwise adjacent in C, which is always true since C is complete).
        # Wait - C is a complete graph! So any subset of C is a clique.
        # So for I-vertex v, we can form cliques {v} ∪ S for ANY S ⊆ N_C[v].
        # The C-C edges covered by v's cliques are all edges within each S.
        # We need: partition of all C-C edges such that each part is a clique (any subset works)
        # and each part S is assigned to a v with S ⊆ N_C[v].
        #
        # Since C is complete, any subset is a clique. So we just need to partition the
        # C-C edge set into groups, where group G is assigned to v iff for all (u,w) in G,
        # u,w ∈ N_C[v]. Equivalently, the set of C-vertices involved in G must be ⊆ N_C[v].
        #
        # This is equivalent to: assign each C-C edge (u,w) to an I-vertex v such that
        # u,w ∈ N_C[v], and then for each v, the edges assigned to v define a set of
        # C-vertices V_v, and we need to partition V_v into cliques (any partition works
        # since C is complete). The number of cliques for v = number of parts in the
        # partition of V_v. To minimize total cliques, for each v we want to minimize
        # the number of parts = we want to make each part as large as possible.
        # Since any subset is a clique, we can put ALL of V_v in one clique!
        # So for each v, if V_v is non-empty, it contributes exactly 1 clique: {v} ∪ V_v.
        #
        # Therefore: minimize number of I-vertices v with V_v non-empty, subject to
        # every C-C edge (u,w) being assigned to some v with u,w ∈ N_C[v].
        #
        # This is a set cover problem! Each I-vertex v "covers" all C-C edges (u,w)
        # with u,w ∈ N_C[v]. We want minimum number of v's whose union covers all C-C edges.
        #
        # But wait - we also need cross edges covered. Cross edge (u,v) must be in a clique
        # containing v. If v has clique {v} ∪ V_v, then cross edge (u,v) is covered iff u ∈ V_v.
        # So we need: for each cross edge (u,v), u ∈ V_v.
        # And V_v is the set of C-vertices involved in C-C edges assigned to v.
        # But we can also include extra vertices in V_v beyond those needed for C-C edges,
        # as long as they're in N_C[v]. Including u in V_v covers cross edge (u,v).
        #
        # So the constraint is: for each cross edge (u,v), u ∈ V_v, and V_v ⊆ N_C[v].
        # And for each C-C edge (u,w), there exists v with u,w ∈ V_v.
        # Minimize: number of v with V_v non-empty.
        #
        # This is still complex but let's try a greedy approach:
        # 1. Start with empty assignment.
        # 2. For each I-vertex v, compute the set of C-C edges it can cover: E_v = {(u,w) : u,w ∈ N_C[v]}.
        # 3. Greedy set cover: repeatedly pick v that covers most uncovered C-C edges.
        # 4. After covering all C-C edges, check cross edge constraints. For each cross edge
        #    (u,v) where u ∉ V_v, we need to add u to V_v. This might require adding v's
        #    clique even if v wasn't selected. If u ∉ N_C[v], it's impossible (but cross edge
        #    exists so u ∈ N_C[v]). So we can always add u to V_v.
        # 5. The cost is: number of v's with V_v non-empty.

        # Precompute for each v: N_C[v] as a set, and the C-C edges it covers
        # C-C edge is identified by (u,w) with u<w<k
        # For efficiency, represent N_C[v] as a bitmask if k is small, else as a set
        
        # Greedy set cover for C-C edges
        all_cc_edges = [(u, w) for u in range(k) for w in range(u+1, k)]
        total_cc = len(all_cc_edges)
        
        # For each v, compute which C-C edges it can cover
        # Edge (u,w) is covered by v iff u ∈ N_C[v] and w ∈ N_C[v]
        v_cc_edges = {}
        for v in range(k, n):
            nc = N_C[v]
            if len(nc) >= 2:
                # All pairs within nc
                nc_list = sorted(nc)
                edges_v = [(a, b) for i, a in enumerate(nc_list) for b in nc_list[i+1:]]
                v_cc_edges[v] = set(edges_v)
            else:
                v_cc_edges[v] = set()
        
        # Greedy set cover
        uncovered = set(all_cc_edges)
        selected = []
        V_v = {v: set() for v in range(k, n)}
        
        while uncovered and not _time_up():
            best_v = None
            best_count = 0
            for v in range(k, n):
                if v in [s for s in selected]:
                    continue
                cnt = len(uncovered & v_cc_edges[v])
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            if best_v is None or best_count == 0:
                break
            selected.append(best_v)
            uncovered -= v_cc_edges[best_v]
            # V_v[best_v] will be determined later
        
        # Now handle cross edges: for each cross edge (u,v), ensure u ∈ V_v[v]
        for (u, v) in cross_edges_norm:
            if u not in V_v[v]:
                V_v[v].add(u)
        
        # For selected v's, V_v[v] should include all C-vertices involved in covered C-C edges
        # But we also need V_v[v] ⊆ N_C[v]. Let's verify and fix.
        for v in selected:
            # V_v[v] must be ⊆ N_C[v]
            V_v[v] = V_v[v] & set(N_C[v])
        
        # Now for each v with V_v[v] non-empty, create clique {v} ∪ V_v[v]
        # But we need to ensure all C-C edges are covered. Let's verify.
        # Actually, the greedy set cover ensured uncovered is empty (or we broke early).
        # If uncovered is not empty, we need to fall back.
        
        if uncovered:
            # Fallback: use Regime 1
            return best_cliques, best_count
        
        # Build cliques
        cliques = []
        for v in range(k, n):
            if V_v[v]:
                cliques.append(sorted([v] + list(V_v[v])))
        
        # Verify validity
        # Each clique must be a clique: check all pairs
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i+1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edge_set:
                        return best_cliques, best_count  # Invalid, fall back
        
        # Verify all edges covered exactly once
        edge_cover_count = {}
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i+1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    edge_cover_count[(a, b)] = edge_cover_count.get((a, b), 0) + 1
        
        for e in edge_set:
            if edge_cover_count.get(e, 0) != 1:
                return best_cliques, best_count  # Invalid, fall back
        
        if len(cliques) < best_count:
            best_cliques = cliques
            best_count = len(cliques)
        
        return best_cliques, best_count

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append(sorted([u, v]))
            results.append(cliques)
            continue
        n, k, p, seed = spec
        cliques, _ = _solve_instance(n, k, p, seed)
        results.append(cliques)
    return results
