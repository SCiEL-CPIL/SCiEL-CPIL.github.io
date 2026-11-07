"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from itertools import combinations

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        edges += cross_edges
        return edges

    def solve_split(n, k, p, seed):
        rng = Random(seed)
        # Determine neighbors in C for each vertex in I
        neighbors = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors[v] = nbrs

        # Build C-C edge set
        cc_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                cc_edges.add((u, w))

        # C-I edges per I-vertex
        ci_edges = {}
        for v in range(k, n):
            ci_edges[v] = set(neighbors[v])

        total_edges = len(cc_edges) + sum(len(s) for s in ci_edges.values())

        def greedy_cover(order, available_cc, seed_val):
            """Process I-vertices in given order, extract max cliques from available C edges."""
            cliques = []
            avail = set(available_cc)
            covered_ci = {v: set() for v in range(k, n)}

            for v in order:
                if v not in ci_edges or not ci_edges[v]:
                    continue
                remaining = set(ci_edges[v] - covered_ci[v])
                if not remaining:
                    continue

                # Greedily extract maximal cliques from remaining C-vertices
                while remaining:
                    # Find a maximal clique in 'remaining' using available C-C edges
                    # Sort by degree within remaining
                    cands = list(remaining)
                    cands.sort(key=lambda x: sum(1 for y in cands if x != y and (min(x,y), max(x,y)) in avail), reverse=True)

                    S = []
                    while cands:
                        u = cands[0]
                        S.append(u)
                        cands = [w for w in cands[1:] if (min(u, w), max(u, w)) in avail]

                    if len(S) >= 2:
                        clique = sorted(S + [v])
                        cliques.append(clique)
                        # Remove C-C edges within S from available
                        for i in range(len(S)):
                            for j in range(i + 1, len(S)):
                                avail.discard((min(S[i], S[j]), max(S[i], S[j])))
                        covered_ci[v] |= set(S)
                        remaining -= set(S)
                    elif len(S) == 1:
                        clique = sorted([S[0], v])
                        cliques.append(clique)
                        covered_ci[v] |= set(S[0])
                        remaining.discard(S[0])
                    else:
                        break

            # Cover remaining C-I edges with 2-vertex cliques
            for v in range(k, n):
                for u in ci_edges[v]:
                    if u not in covered_ci[v]:
                        cliques.append(sorted([u, v]))

            # Cover remaining C-C edges
            for edge in avail:
                cliques.append(sorted(list(edge)))

            return cliques

        def local_search(cliques, available_cc, ci_edges, max_iters=200, seed_val=42):
            """Try to improve by merging or replacing cliques."""
            rng_ls = Random(seed_val)
            current = [list(c) for c in cliques]
            best = current
            best_count = len(current)

            # Build edge-to-clique mapping
            def build_edge_map(clique_list):
                emap = {}
                for ci, c in enumerate(clique_list):
                    for a, b in combinations(sorted(c), 2):
                        emap[(a, b)] = ci
                return emap

            # Try removing redundant cliques and merging
            for _ in range(max_iters):
                emap = build_edge_map(current)

                # Find cliques that can be merged (share edges or are subsets)
                # Try: for each pair of cliques, check if their union is a clique
                # and if replacing them with the union reduces count
                improved = False
                for i in range(len(current)):
                    if improved:
                        break
                    for j in range(i + 1, len(current)):
                        union = set(current[i]) | set(current[j])
                        if len(union) <= max(len(current[i]), len(current[j])):
                            continue
                        # Check if union is a clique
                        is_clique = True
                        for a, b in combinations(sorted(union), 2):
                            if (a, b) not in emap:
                                is_clique = False
                                break
                        if is_clique:
                            # Check that all edges in union are covered by i or j
                            all_covered = True
                            for a, b in combinations(sorted(union), 2):
                                ci = emap.get((a, b), -1)
                                if ci != i and ci != j:
                                    all_covered = False
                                    break
                            if all_covered:
                                new_list = [c for idx, c in enumerate(current) if idx != i and idx != j]
                                new_list.append(sorted(union))
                                if len(new_list) < best_count:
                                    current = new_list
                                    best = [list(c) for c in current]
                                    best_count = len(current)
                                    improved = True
                                    break

                if not improved:
                    # Try: replace a 2-vertex C-I clique with a larger one
                    for i in range(len(current)):
                        if improved:
                            break
                        c = current[i]
                        if len(c) == 2:
                            u, v = c
                            # Try adding a C-vertex w that connects to both
                            # and whose C-C edges are already covered
                            emap = build_edge_map(current)
                            candidates = []
                            for w in range(k):
                                if w == u or w == v:
                                    continue
                                if (min(u, w), max(u, w)) in emap and (min(v, w), max(v, w)) in emap:
                                    candidates.append(w)
                            if candidates:
                                # Pick one that maximizes additional coverage
                                for w in candidates:
                                    new_clique = sorted([u, v, w])
                                    # Check it's a valid clique (all edges exist in graph)
                                    # and edges are covered
                                    valid = True
                                    for a, b in combinations(new_clique, 2):
                                        if (a, b) not in emap:
                                            valid = False
                                            break
                                    if valid:
                                        new_list = [c2 for idx, c2 in enumerate(current) if idx != i]
                                        new_list.append(new_clique)
                                        # Verify no edge is now uncovered
                                        new_emap = build_edge_map(new_list)
                                        ok = True
                                        for a, b in combinations(new_clique, 2):
                                            if (a, b) not in new_emap:
                                                ok = False
                                                break
                                        if ok:
                                            current = new_list
                                            if len(current) < best_count:
                                                best = [list(c) for c in current]
                                                best_count = len(current)
                                            improved = True
                                            break

            return best

        # Generate multiple orderings and pick the best
        I_vertices = [v for v in range(k, n) if ci_edges[v]]

        # Ordering strategies
        orderings = []

        # 1. By degree in C descending
        order1 = sorted(I_vertices, key=lambda v: len(ci_edges[v]), reverse=True)
        orderings.append(order1)

        # 2. By degree in C ascending
        order2 = sorted(I_vertices, key=lambda v: len(ci_edges[v]))
        orderings.append(order2)

        # 3. By number of C-C edges within N(v)
        def cc_degree(v):
            nbrs = ci_edges[v]
            cnt = 0
            for a, b in combinations(sorted(nbrs), 2):
                if (a, b) in cc_edges:
                    cnt += 1
            return cnt

        order3 = sorted(I_vertices, key=cc_degree, reverse=True)
        orderings.append(order3)

        # 4. Random shuffles
        for s in range(5):
            rng_r = Random(seed + 100 + s)
            order_r = I_vertices[:]
            rng_r.shuffle(order_r)
            orderings.append(order_r)

        # 5. Greedy set-cover-like: repeatedly pick I-vertex whose N(v) has max C-C edges
        order5 = []
        remaining_v = set(I_vertices)
        while remaining_v:
            best_v = max(remaining_v, key=lambda v: cc_degree(v))
            order5.append(best_v)
            remaining_v.discard(best_v)
        orderings.append(order5)

        # Evaluate each ordering
        best_cliques = None
        best_count = float('inf')

        for oi, order in enumerate(orderings):
            if time.time() > deadline - 10:
                break
            result = greedy_cover(order, cc_edges, seed + oi)
            count = len(result)
            if count < best_count:
                best_count = count
                best_cliques = result

        # Apply local search to the best
        if best_cliques is not None and time.time() < deadline - 20:
            improved = local_search(best_cliques, cc_edges, ci_edges, max_iters=100, seed_val=seed)
            if len(improved) < best_count:
                best_cliques = improved
                best_count = len(improved)

        if best_cliques is None:
            # Fallback: simple approach
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([v, u])
            if k >= 2:
                cliques.append(list(range(k)))
            best_cliques = cliques

        # Validate: every clique must be a valid clique in the graph
        # Build full edge set for validation
        full_edges = set(cc_edges)
        for v in range(k, n):
            for u in ci_edges[v]:
                full_edges.add((min(u, v), max(u, v)))

        validated = []
        for c in best_cliques:
            valid = True
            for a, b in combinations(sorted(c), 2):
                if (a, b) not in full_edges:
                    valid = False
                    break
            if valid and len(c) >= 2:
                validated.append(c)

        # Check that every edge is covered exactly once
        edge_count = {}
        for c in validated:
            for a, b in combinations(sorted(c), 2):
                edge_count[(a, b)] = edge_count.get((a, b), 0) + 1

        # If any edge is uncovered or over-covered, fall back to safe solution
        uncovered = False
        for e in full_edges:
            if e not in edge_count:
                uncovered = True
                break
        if uncovered or any(cnt != 1 for cnt in edge_count.values()):
            # Fallback to guaranteed correct solution
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append(sorted([u, v]))
            if k >= 2:
                cliques.append(list(range(k)))
            validated = cliques

        return validated

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 5:
            # Emergency fallback for remaining instances
            n, k, p, seed = spec
            rng = Random(seed)
            cliques = []
            for v in range(k, n):
                for u in range(k):
                    if rng.random() < p:
                        cliques.append(sorted([u, v]))
            if k >= 2:
                cliques.append(list(range(k)))
            result.append(cliques)
        else:
            result.append(solve_split(*spec))

    return result
# EVOLVE-BLOCK-END
