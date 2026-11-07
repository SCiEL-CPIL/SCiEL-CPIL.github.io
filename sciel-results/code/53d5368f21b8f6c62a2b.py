import time
from random import Random
from itertools import combinations

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

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)

        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # Separate C-C edges and cross edges
        cc_edges = []
        cross_edges = []
        for (a, b) in edge_set:
            if a < k and b < k:
                cc_edges.append((a, b))
            else:
                cross_edges.append((a, b))

        # The C-C edges form K_k, covered by one clique
        # For cross edges, we want to group them into cliques
        # A clique containing cross edges must be of the form:
        #   {u} ∪ S_v where u < k, all v in S_v >= k, and (u,v) is an edge for all v in S_v
        #   {v} ∪ S_u where v >= k, all u in S_u < k, and (u,v) is an edge for all u in S_u
        #   Or a single cross edge {u, v}

        # Build adjacency: for each cross edge, what other cross edges can be in the same clique?
        # Two cross edges (u1, v1) and (u2, v2) can be in the same clique if:
        #   - They share a C-vertex: u1 == u2 (clique {u1} ∪ {v1, v2, ...})
        #   - They share an I-vertex: v1 == v2 (clique {v1} ∪ {u1, u2, ...})
        #   - They form a triangle: u1 != u2, v1 != v2, and (u1,v2) and (u2,v1) are also edges
        #     In that case, clique {u1, u2, v1, v2} works if all 6 pairs are edges
        #     But this is complex; let's focus on stars first

        # Simplified approach: model as matching where each cross-edge is a node
        # Edge between two cross-edges if they can be in the same star
        # Weight = 1 if merging reduces clique count by 1 (i.e., two edges become one clique)

        # Actually, let's think differently. We want to partition cross edges into groups.
        # Each group must form a clique when combined with their shared vertex.
        # 
        # Group type 1: All edges share the same C-vertex u. Group = {(u, v1), (u, v2), ...}
        #   Clique = {u, v1, v2, ...}. Cost = 1.
        # Group type 2: All edges share the same I-vertex v. Group = {(u1, v), (u2, v), ...}
        #   Clique = {v, u1, u2, ...}. Cost = 1.
        # Group type 3: Single edge {(u, v)}. Cost = 1.
        #
        # We want to minimize total cost = number of groups.
        # This is equivalent to: maximize savings. Each group of size s saves (s-1) cliques.
        #
        # Let's use a different formulation. For each cross-edge e, initially it's its own clique (cost 1).
        # If we merge edges into a star centered at C-vertex u, we save (|S|-1) where S is the set of I-vertices.
        # If we merge edges into a star centered at I-vertex v, we save (|S|-1) where S is the set of C-vertices.
        #
        # But edges can't be in two stars simultaneously. So this is a set packing / matching problem.
        #
        # Let's model it as: create a bipartite-like structure.
        # For each C-vertex u, we can choose a subset of its incident cross-edges to form a star.
        # For each I-vertex v, we can choose a subset of its incident cross-edges to form a star.
        # No cross-edge can be in both a C-star and an I-star.
        #
        # This is complex. Let's try a simpler greedy approach with better heuristics.

        # Greedy: repeatedly pick the best merge (largest savings)
        # A C-star at u with |S| edges saves |S|-1
        # An I-star at v with |S| edges saves |S|-1

        # Track which cross-edges are already assigned
        ce_index = {}
        for i, e in enumerate(cross_edges):
            ce_index[e] = i

        assigned = [False] * len(cross_edges)

        # For each C-vertex, list of incident cross-edges
        c_stars = {u: [] for u in range(k)}
        # For each I-vertex, list of incident cross-edges
        i_stars = {v: [] for v in range(k, n)}

        for i, (u, v) in enumerate(cross_edges):
            c_stars[u].append(i)
            i_stars[v].append(i)

        cliques = [list(range(k))]  # C-C clique

        # Greedy: pick the star with maximum savings
        # Savings for C-star at u = len([e for e in c_stars[u] if not assigned[e]]) - 1
        # Savings for I-star at v = len([e for e in i_stars[v] if not assigned[e]]) - 1

        while True:
            if _time_up():
                break

            best_saving = 0
            best_type = None  # 'c' or 'i'
            best_vertex = None

            for u in range(k):
                available = [e for e in c_stars[u] if not assigned[e]]
                if len(available) >= 2:
                    saving = len(available) - 1
                    if saving > best_saving:
                        best_saving = saving
                        best_type = 'c'
                        best_vertex = u

            for v in range(k, n):
                available = [e for e in i_stars[v] if not assigned[e]]
                if len(available) >= 2:
                    saving = len(available) - 1
                    if saving > best_saving:
                        best_saving = saving
                        best_type = 'i'
                        best_vertex = v

            if best_saving <= 0:
                break

            if best_type == 'c':
                u = best_vertex
                available = [e for e in c_stars[u] if not assigned[e]]
                vertices = [u] + [cross_edges[e][1] for e in available]
                cliques.append(sorted(vertices))
                for e in available:
                    assigned[e] = True
            else:
                v = best_vertex
                available = [e for e in i_stars[v] if not assigned[e]]
                vertices = [v] + [cross_edges[e][0] for e in available]
                cliques.append(sorted(vertices))
                for e in available:
                    assigned[e] = True

        # Remaining unassigned cross-edges become individual cliques
        for i, (u, v) in enumerate(cross_edges):
            if not assigned[i]:
                cliques.append(sorted([u, v]))

        if not cliques:
            cliques = [list(range(k))]

        # Validate
        edge_count = {}
        for clique in cliques:
            if len(clique) < 2:
                continue
            for a, b in combinations(clique, 2):
                e = (a, b) if a < b else (b, a)
                edge_count[e] = edge_count.get(e, 0) + 1

        missing = [e for e in edge_set if edge_count.get(e, 0) == 0]
        over = [e for e, c in edge_count.items() if c > 1]

        if missing or over:
            cliques = [list(range(k))]
            for (u, v) in edge_set:
                if u >= k or v >= k:
                    cliques.append([u, v])

        results.append(cliques)

    return results
