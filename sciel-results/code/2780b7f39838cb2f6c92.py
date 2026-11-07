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

        # Adjacency for C vertices
        adj_C = [set() for _ in range(k)]
        for u in range(k):
            for w in range(u + 1, k):
                adj_C[u].add(w)
                adj_C[w].add(u)

        best_cliques = None
        best_count = float('inf')

        # Strategy 1: Greedy extraction of max clique from available CC edges for each I-vertex
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
                candidates.sort(key=lambda x: sum(1 for w in candidates if x != w and (min(x, w), max(x, w)) in available_cc), reverse=True)

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

        # Strategy 2: Multi-clique per I-vertex
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
                    cands.sort(key=lambda x: sum(1 for w in cands if x != w and (min(x, w), max(x, w)) in available_cc), reverse=True)

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
                assigned_ci[v] = set(E_CI[v])

            # Cover remaining CC edges
            for edge in available_cc:
                cliques.append(sorted(list(edge)))

            return cliques

        # Strategy 3: Edge-based greedy with extension
        def strategy_edge_greedy():
            all_edges = set()
            for u in range(k):
                for w in range(u + 1, k):
                    all_edges.add((u, w))
            for v in range(k, n):
                for u in E_CI[v]:
                    all_edges.add((min(u, v), max(u, v)))

            rem_set = set(all_edges)
            cliques = []

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
                best_edge = None
                best_score = -1
                for e in list(rem_set)[:50]:
                    u, w = e
                    common = adj[u] & adj[w]
                    score = sum(1 for x in common if x != u and x != w and (min(u, x), max(u, x)) in rem_set and (min(w, x), max(w, x)) in rem_set)
                    if score > best_score:
                        best_score = score
                        best_edge = e

                if best_edge is None:
                    best_edge = next(iter(rem_set))

                u, w = best_edge
                clique_vertices = {u, w}

                candidates = list(adj[u] & adj[w])
                candidates = [x for x in candidates if x != u and x != w]
                candidates.sort(key=lambda x: sum(1 for y in clique_vertices if x in adj[y]), reverse=True)

                for x in candidates:
                    if all((min(x, y), max(x, y)) in rem_set for y in clique_vertices):
                        clique_vertices.add(x)

                for u1, u2 in combinations(sorted(clique_vertices), 2):
                    rem_set.discard((u1, u2))

                cliques.append(sorted(clique_vertices))

            return cliques

        # Generate different orderings
        I_vertices = list(range(k, n))

        order1 = sorted(I_vertices, key=lambda v: len(E_CI[v]), reverse=True)
        order2 = sorted(I_vertices, key=lambda v: len(E_CI[v]))
        rng2 = random.Random(seed + 1000)
        order3 = I_vertices[:]
        rng2.shuffle(order3)

        # Try all strategies and orderings, keep best
        strategies = [
            lambda: strategy_greedy_cc(order1),
            lambda: strategy_greedy_cc(order2),
            lambda: strategy_greedy_cc(order3),
            lambda: strategy_multi_clique(order1),
            lambda: strategy_multi_clique(order2),
            lambda: strategy_multi_clique(order3),
            strategy_edge_greedy,
        ]

        for strat in strategies:
            if time.time() > deadline:
                break
            try:
                cliques = strat()
                count = len(cliques)
                if count < best_count:
                    best_count = count
                    best_cliques = cliques
            except Exception:
                pass

        if best_cliques is None:
            # Fallback: just pair up edges
            best_cliques = []
            all_edges = set()
            for u in range(k):
                for w in range(u + 1, k):
                    all_edges.add((u, w))
            for v in range(k, n):
                for u in E_CI[v]:
                    all_edges.add((min(u, v), max(u, v)))
            for e in all_edges:
                best_cliques.append(sorted(list(e)))

        # Safety net: verify all edges covered and add any missing
        all_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_edges.add((u, w))
        for v in range(k, n):
            for u in E_CI[v]:
                all_edges.add((min(u, v), max(u, v)))

        covered_all = set()
        for cl in best_cliques:
            for u1, u2 in combinations(cl, 2):
                covered_all.add((min(u1, u2), max(u1, u2)))

        uncovered = all_edges - covered_all
        unc_set = set(uncovered)
        while unc_set:
            if time.time() > deadline:
                break
            u, w = next(iter(unc_set))
            clique_vertices = {u, w}
            candidates = []
            for x in range(n):
                if x not in clique_vertices:
                    is_connected = True
                    for y in clique_vertices:
                        if (min(x, y), max(x, y)) not in unc_set:
                            is_connected = False
                            break
                    if is_connected:
                        candidates.append(x)
            clique_vertices.update(candidates)
            for u1, u2 in combinations(sorted(clique_vertices), 2):
                unc_set.discard((u1, u2))
            if len(clique_vertices) >= 2:
                best_cliques.append(sorted(clique_vertices))

        results.append([[int(x) for x in cl] for cl in best_cliques])

    return results
