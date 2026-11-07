"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

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

    def solve_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []

        # Build adjacency
        # C = {0,...,k-1}, I = {k,...,n-1}
        # E_CC: all edges within C (complete graph on k vertices)
        # E_CI: for each v in I, set of neighbors in C

        E_CC = set()
        for u in range(k):
            for v in range(u + 1, k):
                E_CC.add((u, v))

        E_CI = {}
        for u, v in edges:
            if u < k and v >= k:
                if v not in E_CI:
                    E_CI[v] = set()
                E_CI[v].add(u)
            elif v < k and u >= k:
                if u not in E_CI:
                    E_CI[u] = set()
                E_CI[u].add(v)

        # Total edges to cover
        total_cc = len(E_CC)
        total_ci = sum(len(s) for s in E_CI.values())
        total_edges = total_cc + total_ci

        # Baseline: each edge is its own clique
        baseline = total_edges

        def greedy_partition(order, max_clique_size=None):
            """
            Greedy: process I-vertices in given order.
            For each, find a large clique S in N(v) such that all C-edges in S are still available.
            Then cover remaining edges with edge-cliques.
            Returns (cliques, num_cliques).
            """
            available_cc = set(E_CC)
            cliques = []
            assigned_ci = {}  # v -> set of C-neighbors already covered

            for v in order:
                if v not in E_CI or not E_CI[v]:
                    continue
                neighbors = list(E_CI[v])
                if max_clique_size is not None:
                    neighbors = neighbors[:max_clique_size]

                # Greedy max clique in the subgraph induced by neighbors where edge exists iff in available_cc
                # Sort by degree in available_cc within neighbors
                def degree_in_avail(u, cand_list):
                    cnt = 0
                    for w in cand_list:
                        if u != w and (min(u, w), max(u, w)) in available_cc:
                            cnt += 1
                    return cnt

                candidates = list(neighbors)
                candidates.sort(key=lambda x: degree_in_avail(x, candidates), reverse=True)

                S = []
                while candidates:
                    u = candidates[0]
                    S.append(u)
                    new_cands = []
                    for w in candidates[1:]:
                        if (min(u, w), max(u, w)) in available_cc:
                            new_cands.append(w)
                    candidates = new_cands

                if len(S) >= 2:
                    cliques.append(S + [v])
                    # Remove C-edges consumed by S
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            available_cc.discard((min(S[i], S[j]), max(S[i], S[j])))
                    assigned_ci[v] = set(S)
                elif len(S) == 1:
                    # Single C-neighbor: just cover the cross edge
                    cliques.append([S[0], v])
                    assigned_ci[v] = set(S)

            # Cover remaining CI edges
            for v in E_CI:
                assigned = assigned_ci.get(v, set())
                for u in E_CI[v]:
                    if u not in assigned:
                        cliques.append([u, v])

            # Cover remaining CC edges
            for edge in available_cc:
                cliques.append([edge[0], edge[1]])

            return cliques, len(cliques)

        def randomized_greedy(order, max_clique_size=None, jitter=0.0):
            """Greedy with random tie-breaking."""
            available_cc = set(E_CC)
            cliques = []
            assigned_ci = {}

            for v in order:
                if v not in E_CI or not E_CI[v]:
                    continue
                neighbors = list(E_CI[v])
                if max_clique_size is not None:
                    neighbors = neighbors[:max_clique_size]

                # Randomize order with jitter
                def degree_in_avail(u, cand_list):
                    cnt = 0
                    for w in cand_list:
                        if u != w and (min(u, w), max(u, w)) in available_cc:
                            cnt += 1
                    return cnt

                candidates = list(neighbors)
                # Add jitter to break ties randomly
                keyed = [(degree_in_avail(x, candidates) + random.random() * jitter, x) for x in candidates]
                keyed.sort(reverse=True)
                candidates = [x for _, x in keyed]

                S = []
                while candidates:
                    u = candidates[0]
                    S.append(u)
                    new_cands = []
                    for w in candidates[1:]:
                        if (min(u, w), max(u, w)) in available_cc:
                            new_cands.append(w)
                    candidates = new_cands

                if len(S) >= 2:
                    cliques.append(S + [v])
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            available_cc.discard((min(S[i], S[j]), max(S[i], S[j])))
                    assigned_ci[v] = set(S)
                elif len(S) == 1:
                    cliques.append([S[0], v])
                    assigned_ci[v] = set(S)

            for v in E_CI:
                assigned = assigned_ci.get(v, set())
                for u in E_CI[v]:
                    if u not in assigned:
                        cliques.append([u, v])

            for edge in available_cc:
                cliques.append([edge[0], edge[1]])

            return cliques, len(cliques)

        def two_phase_partition(order):
            """
            Two-phase: Phase 1 - for each v, find a maximal clique in N(v) using available CC edges.
            Phase 2 - try to merge adjacent I-vertex cliques if they share C-vertices (can't, since different I vertices).
            Actually, let's try: for each v, instead of just one clique, try to cover ALL of N(v)
            by multiple cliques in N(v), each being a clique in the available CC subgraph.
            """
            available_cc = set(E_CC)
            cliques = []
            assigned_ci = {}

            for v in order:
                if v not in E_CI or not E_CI[v]:
                    continue
                neighbors = list(E_CI[v])

                # Try to cover all of N(v) with cliques in the available CC subgraph
                remaining = set(neighbors)
                v_cliques = []

                while remaining:
                    # Find a maximal clique in remaining
                    cands = list(remaining)
                    def degree_in_avail(u, cand_list):
                        cnt = 0
                        for w in cand_list:
                            if u != w and (min(u, w), max(u, w)) in available_cc:
                                cnt += 1
                        return cnt
                    cands.sort(key=lambda x: degree_in_avail(x, cands), reverse=True)

                    S = []
                    while cands:
                        u = cands[0]
                        S.append(u)
                        new_cands = []
                        for w in cands[1:]:
                            if (min(u, w), max(u, w)) in available_cc:
                                new_cands.append(w)
                        cands = new_cands

                    if S:
                        if len(S) >= 2:
                            v_cliques.append(S)
                            for i in range(len(S)):
                                for j in range(i + 1, len(S)):
                                    available_cc.discard((min(S[i], S[j]), max(S[i], S[j])))
                        else:
                            v_cliques.append(S)
                        remaining -= set(S)

                # Now form cliques: each C-clique S combined with v
                # But wait: if we have multiple C-cliques for the same v, each becomes a clique with v.
                # However, the cross edges (u, v) for u in different C-cliques are each covered by their respective clique.
                # The C-edges within each C-clique are covered.
                # This is valid.
                for S in v_cliques:
                    cliques.append(S + [v])
                    assigned_ci.setdefault(v, set()).update(S)

            # Cover remaining CI edges
            for v in E_CI:
                assigned = assigned_ci.get(v, set())
                for u in E_CI[v]:
                    if u not in assigned:
                        cliques.append([u, v])

            # Cover remaining CC edges
            for edge in available_cc:
                cliques.append([edge[0], edge[1]])

            return cliques, len(cliques)

        # Try multiple strategies and keep the best
        best_cliques = None
        best_count = baseline

        I_vertices = list(E_CI.keys())
        if not I_vertices:
            # Only CC edges
            return [[u, v] for u, v in E_CC]

        # Strategy 1: Greedy by degree descending
        order1 = sorted(I_vertices, key=lambda v: len(E_CI[v]), reverse=True)
        if time.time() < deadline:
            c, cnt = greedy_partition(order1)
            if cnt < best_count:
                best_count = cnt
                best_cliques = c

        # Strategy 2: Greedy by degree ascending
        order2 = sorted(I_vertices, key=lambda v: len(E_CI[v]))
        if time.time() < deadline:
            c, cnt = greedy_partition(order2)
            if cnt < best_count:
                best_count = cnt
                best_cliques = c

        # Strategy 3: Two-phase (cover all N(v) with multiple cliques)
        if time.time() < deadline:
            c, cnt = two_phase_partition(order1)
            if cnt < best_count:
                best_count = cnt
                best_cliques = c

        if time.time() < deadline:
            c, cnt = two_phase_partition(order2)
            if cnt < best_count:
                best_count = cnt
                best_cliques = c

        # Strategy 4: Randomized restarts
        num_restarts = 20
        elapsed = time.time() - start_time
        time_per_restart = max(0.1, (deadline - elapsed) / num_restarts)

        for r in range(num_restarts):
            if time.time() > deadline - 2:
                break
            random_order = I_vertices[:]
            random.shuffle(random_order)
            # Also try sorting by random perturbation of degree
            perturbed = sorted(I_vertices, key=lambda v: len(E_CI[v]) + random.gauss(0, 2))

            for order in [random_order, perturbed]:
                if time.time() > deadline - 1:
                    break
                for ms in [None, 8, 12, 15]:
                    if time.time() > deadline - 0.5:
                        break
                    c, cnt = randomized_greedy(order, max_clique_size=ms, jitter=1.0)
                    if cnt < best_count:
                        best_count = cnt
                        best_cliques = c

        # Strategy 5: Try to improve by local search - swap assignments
        # For each pair of I-vertices, try swapping their S assignments
        if best_cliques is not None and time.time() < deadline - 5:
            # Reconstruct from best_cliques is complex; skip local search for now
            pass

        if best_cliques is None:
            # Fallback: edge cliques
            best_cliques = [[u, v] for u, v in edges]

        # Validate and convert to proper types
        result = []
        for clique in best_cliques:
            result.append([int(x) for x in clique])

        return result

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[int(u), int(v)] for u, v in edges])
        else:
            res = solve_split_graph(*spec)
            results.append(res)

    return results
# EVOLVE-BLOCK-END
