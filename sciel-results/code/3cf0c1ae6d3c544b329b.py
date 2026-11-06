import time
import math
import random as _random
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = _random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def enumerate_maximal_cliques(adj, n):
        cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return

            # Pivoting to reduce branching
            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u

            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])

            for v in candidates:
                if time.time() - start_time > DEADLINE:
                    return
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)

        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        cliques = enumerate_maximal_cliques(adj, n)

        if not cliques:
            return []

        m = len(cliques)

        # Map vertex -> set of clique indices it belongs to
        vertex_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(ci)

        active_vertices = [v for v in range(n) if vertex_cliques[v]]
        if not active_vertices:
            return []

        # Dominance Pruning
        # Sort by number of cliques covered (descending)
        active_vertices_sorted = sorted(active_vertices, key=lambda v: -len(vertex_cliques[v]))
        keep_vertices = set()
        for v in active_vertices_sorted:
            dominated = False
            for u in keep_vertices:
                if vertex_cliques[v].issubset(vertex_cliques[u]):
                    dominated = True
                    break
            if not dominated:
                keep_vertices.add(v)

        candidates = list(keep_vertices)
        if not candidates:
            return []

        # Greedy Set Cover for initial upper bound
        def greedy_cover():
            selected = set()
            unc = set(range(m))
            while unc:
                best_v = None
                best_cov = 0
                for v in candidates:
                    if v in selected:
                        continue
                    cov = len(unc & vertex_cliques[v])
                    if cov > best_cov:
                        best_cov = cov
                        best_v = v
                if best_v is None:
                    break
                selected.add(best_v)
                unc -= vertex_cliques[best_v]
            return selected

        best_solution = greedy_cover()
        best_size = len(best_solution)

        # Branch and Bound with improved lower bound
        def branch_and_bound(selected, uncovered):
            nonlocal best_size, best_solution

            if not uncovered:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
                return

            if len(selected) >= best_size:
                return

            if time.time() - start_time > DEADLINE:
                return

            # Lower Bound 1: Max coverage bound
            max_cov = 0
            for v in candidates:
                if v in selected:
                    continue
                cov = len(uncovered & vertex_cliques[v])
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                return

            lb1 = len(selected) + math.ceil(len(uncovered) / max_cov)
            if lb1 >= best_size:
                return

            # Lower Bound 2: Disjoint Clique Packing Bound
            # Find a set of uncovered cliques that are pairwise disjoint.
            # Each such clique requires at least one unique vertex.
            # We greedily select uncovered cliques that are disjoint from previously selected ones.
            packed_cliques = []
            for ci in uncovered:
                c = cliques[ci]
                is_disjoint = True
                for pc in packed_cliques:
                    if c & pc:
                        is_disjoint = False
                        break
                if is_disjoint:
                    packed_cliques.append(c)
            
            lb2 = len(selected) + len(packed_cliques)
            if lb2 >= best_size:
                return

            # Branching: Choose an uncovered clique with fewest available candidates
            target_clique_idx = -1
            min_avail = float('inf')
            best_avail_verts = []
            
            for ci in uncovered:
                c = cliques[ci]
                avail = [v for v in c if v in candidates and v not in selected]
                if len(avail) < min_avail:
                    min_avail = len(avail)
                    target_clique_idx = ci
                    best_avail_verts = avail
                    if min_avail == 0:
                        # No candidate can cover this clique? Should not happen if candidates cover all
                        return
                    if min_avail == 1:
                        # Forced move
                        break

            if target_clique_idx == -1:
                return

            if not best_avail_verts:
                return

            # Order vertices by their coverage of the *current* uncovered set
            # This helps find good solutions earlier
            best_avail_verts.sort(key=lambda v: -len(uncovered & vertex_cliques[v]))

            for v in best_avail_verts:
                new_selected = selected | {v}
                new_uncovered = uncovered - vertex_cliques[v]
                branch_and_bound(new_selected, new_uncovered)

        initial_uncovered = set(range(m))
        branch_and_bound(set(), initial_uncovered)

        if best_solution is None:
            return []

        return sorted(best_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > DEADLINE - 10:
            # Not enough time, return empty or best effort? 
            # Task requires valid result. If we run out of time, we should probably return the greedy solution if we have it, 
            # but we compute per instance. 
            # Since we solve sequentially, if we run out of time mid-instance, we might return incomplete.
            # However, n=40 is the largest. If we finish 30 and 35 quickly, we have time for 40.
            # Let's just solve. The deadline check inside B&B will stop it and return best found so far.
            pass
        
        sol = solve_instance(n, p, seed)
        results.append(sol)

    return results
