import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    start_time = time.time()
    deadline = start_time + 115.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        rng = random.Random(seed)
        neighbors = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        # Build edge sets
        E_CC = set()
        for u in range(k):
            for w in range(u + 1, k):
                E_CC.add((u, w))

        E_CI = {}
        for v in range(k, n):
            E_CI[v] = set(neighbors[v] & set(range(k)))

        total_cc = len(E_CC)
        total_ci = sum(len(s) for s in E_CI.values())

        # Try multiple strategies and keep the best
        best_cliques = None
        best_count = float('inf')

        # Strategy 1: Process I-vertices, greedily extract max clique from available CC edges
        def strategy_greedy_cc(order):
            available_cc = set(E_CC)
            cliques = []
            assigned_ci = {}

            for v in order:
                if not E_CI[v]:
                    continue
                neighbors_v = list(E_CI[v])

                # Find maximal clique in available_cc restricted to neighbors_v
                candidates = list(neighbors_v)
                candidates.sort(key=lambda x: sum(1 for w in candidates if x != w and (min(x,w), max(x,w)) in available_cc), reverse=True)

                S = []
                while candidates:
                    u = candidates[0]
                    S.append(u)
                    new_cands = [w for w in candidates[1:] if (min(u, w), max(u, w)) in available_cc]
                    candidates = new_cands

                if len(S) >= 2:
                    cliques.append(sorted(S + [v]))
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            available_cc.discard((min(S[i], S[j]), max(S[i], S[j])))
                    assigned_ci[v] = set(S)
                elif len(S) == 1:
                    cliques.append(sorted([S[0], v]))
                    assigned_ci[v] = set(S)

            # Cover remaining CI edges
            for v in E_CI:
                assigned = assigned_ci.get(v, set())
                for u in E_CI[v]:
                    if u not in assigned:
                        cliques.append(sorted([u, v]))

            # Cover remaining CC edges
            for edge in available_cc:
                cliques.append(sorted(list(edge)))

            return cliques

        # Strategy 2: Process I-vertices, cover ALL N(v) with multiple cliques
        def strategy_multi_clique(order):
            available_cc = set(E_CC)
            cliques = []
            assigned_ci = {}

            for v in order:
                if not E_CI[v]:
                    continue
                remaining = set(E_CI[v])
                v_cliques = []

                while remaining:
                    cands = list(remaining)
                    cands.sort(key=lambda x: sum(1 for w in cands if x != w and (min(x,w), max(x,w)) in available_cc), reverse=True)

                    S = []
                    while cands:
                        u = cands[0]
                        S.append(u)
                        new_cands = [w for w in cands[1:] if (min(u, w), max(u, w)) in available_cc]
                        cands = new_cands

                    if len(S) >= 2:
                        v_cliques.append(sorted(S + [v]))
                        for i in range(len(S)):
                            for j in range(i + 1, len(S)):
                                available_cc.discard((min(S[i], S[j]), max(S[i], S[j])))
                        remaining -= set(S)
                    elif len(S) == 1:
                        v_cliques.append(sorted([S[0], v]))
                        remaining.discard(S[0])
                    else:
                        break

                cliques.extend(v_cliques)
                assigned_ci[v] = set(E_CI[v])  # All covered

            # Cover remaining CC edges
            for edge in available_cc:
                cliques.append(sorted(list(edge)))

            return cliques

        # Strategy 3: Direct edge-based greedy (from inspiration)
        def strategy_edge_greedy():
            # Build all edges
            all_edges = set()
            for u in range(k):
                for w in range(u + 1, k):
                    all_edges.add((u, w))
            for v in range(k, n):
                for u in E_CI[v]:
                    all_edges.add((min(u, v), max(u, v)))

            rem_set = set(all_edges)
            cliques = []

            # Precompute adjacency for quick clique check
            adj = [set() for _ in range(n)]
            for u in range(k):
                for w in range(u + 1, k):
                    adj[u].add(w)
                    adj[w].add(u)
            for v in range(k, n):
                for u in E_CI[v]:
                    adj[u].add(v)
                    adj[v].add(u)

            while rem_set:
                if time.time() > deadline:
                    break
                # Pick an edge with maximum potential for extension
                # Score: number of common neighbors in rem_set that form a clique
                best_edge = None
                best_score = -1
                for e in list(rem_set)[:50]:  # Limit search for speed
                    u, w = e
                    # Count common neighbors x such that (u,x) and (w,x) are in rem_set
                    common = adj[u] & adj[w]
                    score = sum(1 for x in common if x != u and x != w and (min(u,x),max(u,x)) in rem_set and (min(w,x),max(w,x)) in rem_set)
                    if score > best_score:
                        best_score = score
                        best_edge = e

                if best_edge is None:
                    best_edge = next(iter(rem_set))

                u, w = best_edge
                clique_vertices = {u, w}

                # Try to extend
                candidates = list(adj[u] & adj[w])
                candidates = [x for x in candidates if x != u and x != w]
                candidates.sort(key=lambda x: sum(1 for y in clique_vertices if x in adj[y]), reverse=True)

                for x in candidates:
                    if all((min(x, y), max(x, y)) in rem_set for y in clique_vertices):
                        clique_vertices.add(x)

                # Remove covered edges
                for u1, u2 in combinations(sorted(clique_vertices), 2):
                    rem_set.discard((u1, u2))

                cliques.append(sorted(clique_vertices))

            return cliques

        # Generate different orderings
        I_vertices = list(range(k, n))

        # Order 1: by degree in C descending
        order1 = sorted(I_vertices, key=lambda v: len(E_CI[v]), reverse=True)

        # Order 2: by degree in C ascending
        order2 = sorted(I_vertices, key=lambda v: len(E_CI[v]))

        # Order 3: random
        rng2 = random.Random(seed + 1000)
        order3 = I_vertices[:]
        rng2.shuffle(order3)

        # Order 4: by number of edges in N(v) that are in E_CC
        def cc_degree(v):
            nv = E_CI[v]
            return sum(1 for u1 in nv for u2 in nv if u1 < u2 and (u1, u2) in E_CC)

        order4 = sorted(I_vertices, key=cc_degree, reverse=True)

        # Try all strategies with all orders
        strategies = [
            (strategy_greedy_cc, "greedy_cc"),
            (strategy_multi_clique, "multi_clique"),
            (strategy_edge_greedy, "edge_greedy"),
        ]

        orders = [order1, order2, order3, order4]

        for strat_func, strat_name in strategies:
            if strat_name == "edge_greedy":
                cliques = strat_func()
                count = len(cliques)
                if count < best_count:
                    best_count = count
                    best_cliques = cliques
            else:
                for order in orders:
                    if time.time() > deadline:
                        break
                    cliques = strat_func(order)
                    count = len(cliques)
                    if count < best_count:
                        best_count = count
                        best_cliques = cliques

        # If no strategy produced valid result, fallback
        if best_cliques is None:
            # Fallback: each edge as a clique
            best_cliques = []
            for u in range(k):
                for w in range(u + 1, k):
                    best_cliques.append([u, w])
            for v in range(k, n):
                for u in E_CI[v]:
                    best_cliques.append([u, v])

        # Validate and fix if needed
        # Check that every clique is actually a clique
        valid_cliques = []
        for cl in best_cliques:
            cl_set = set(cl)
            is_clique = True
            for u, w in combinations(sorted(cl), 2):
                if w not in neighbors[u]:
                    is_clique = False
                    break
            if is_clique and len(cl) >= 2:
                valid_cliques.append(sorted(cl))

        # Check coverage
        all_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_edges.add((u, w))
        for v in range(k, n):
            for u in E_CI[v]:
                all_edges.add((min(u, v), max(u, v)))

        covered = set()
        for cl in valid_cliques:
            for u, w in combinations(sorted(cl), 2):
                covered.add((u, w))

        uncovered = all_edges - covered

        # Add uncovered edges as individual cliques
        for u, w in uncovered:
            valid_cliques.append([u, w])

        # Final validation
        final_cliques = []
        for cl in valid_cliques:
            cl_set = set(cl)
            is_clique = True
            for u, w in combinations(sorted(cl), 2):
                if w not in neighbors[u]:
                    is_clique = False
                    break
            if is_clique and len(cl) >= 2:
                final_cliques.append(sorted(cl))

        results.append(final_cliques)

    return results
