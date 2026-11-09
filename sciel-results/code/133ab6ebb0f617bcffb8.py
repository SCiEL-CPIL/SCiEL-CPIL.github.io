import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        """Returns list of maximal cliques as frozensets"""
        cliques = []
        neighbors = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v and adj[u][v]:
                    neighbors[u].add(v)

        def expand(R, P, X):
            if not P and not X:
                cliques.append(frozenset(R))
                return
            # Pivot selection
            union = P | X
            if union:
                # Choose pivot with maximum |P ∩ N(u)|
                max_count = -1
                pivot = None
                for u in union:
                    count = len(P & neighbors[u])
                    if count > max_count:
                        max_count = count
                        pivot = u
                candidates = P - neighbors[pivot] if pivot is not None else P
            else:
                candidates = P

            for v in list(candidates):
                new_R = R | {v}
                new_P = P & neighbors[v]
                new_X = X & neighbors[v]
                expand(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)

        expand(frozenset(), set(range(n)), set())
        return cliques

    def is_valid(sol, cliques):
        sol_set = set(sol)
        for c in cliques:
            if not sol_set & c:
                return False
        return True

    def greedy_init(cliques, n):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques"""
        uncovered = [set(c) for c in cliques]
        covered = [False] * len(uncovered)
        selected = []

        while True:
            if all(covered):
                break
            hits = [0] * n
            for i, c in enumerate(uncovered):
                if not covered[i]:
                    for v in c:
                        hits[v] += 1
            max_hits = max(hits)
            if max_hits == 0:
                break
            # Pick vertex with max hits, tie-break by smaller index
            v = min(range(n), key=lambda x: (-hits[x], x))
            selected.append(v)
            for i, c in enumerate(uncovered):
                if not covered[i] and v in c:
                    covered[i] = True

        # Post-processing: try to remove vertices that are no longer needed
        sol_set = set(selected)
        for v in list(selected):
            new_set = sol_set - {v}
            ok = True
            for c in cliques:
                if not new_set & c:
                    ok = False
                    break
            if ok:
                selected.remove(v)
                sol_set = new_set

        return selected

    def local_search(sol, cliques, n, max_iters=5000):
        """Local search to minimize solution size. Returns improved solution."""
        sol_set = set(sol)
        best_sol = list(sol_set)
        best_size = len(best_sol)

        # Precompute for each vertex, which cliques it covers
        # And for each clique, which vertices can cover it
        cliques_list = list(cliques)
        num_cliques = len(cliques_list)

        # Precompute: for each vertex, the set of clique indices it covers
        vertex_covers = [set() for _ in range(n)]
        for i, c in enumerate(cliques_list):
            for v in c:
                vertex_covers[v].add(i)

        # Quick check if all cliques are covered
        def covers_all(s):
            s = set(s)
            for c in cliques_list:
                if not s & c:
                    return False
            return True

        if not covers_all(best_sol):
            # Need to fix - add vertices until valid
            for c in cliques_list:
                if not set(best_sol) & c:
                    best_sol.append(min(c))
            best_size = len(best_sol)

        rng = random.Random(42)
        ls_start = time.time()
        ls_deadline = ls_start + max(1.0, (deadline - (time.time() - start_time)) * 0.8)

        for it in range(max_iters):
            if time.time() > ls_deadline:
                break
            improved = False

            # Try removing a vertex
            sol_set = set(best_sol)
            for v in best_sol:
                new_set = sol_set - {v}
                # Check if all cliques still covered
                ok = True
                for c in cliques_list:
                    if not new_set & c:
                        ok = False
                        break
                if ok:
                    best_sol.remove(v)
                    best_size -= 1
                    improved = True
                    break

            if improved:
                continue

            # Try 1-swap: remove one vertex v, add one vertex w (w not in sol)
            # such that all cliques are still covered. Size stays same, but may enable
            # subsequent removals.
            sol_set = set(best_sol)
            found_swap = False
            for v in best_sol:
                # Cliques that would become uncovered if we remove v
                uncovered = []
                for i, c in enumerate(cliques_list):
                    # Check if c has any other vertex in sol_set
                    other = False
                    for u in c:
                        if u in sol_set and u != v:
                            other = True
                            break
                    if not other:
                        uncovered.append(i)
                if not uncovered:
                    # v is removable (should have been caught above)
                    continue

                # Find candidate w that covers all uncovered cliques
                # w must be in intersection of all uncovered cliques
                candidates = set(cliques_list[uncovered[0]])
                for idx in uncovered[1:]:
                    candidates &= set(cliques_list[idx])
                # Remove vertices already in sol
                candidates -= sol_set

                for w in candidates:
                    new_set = sol_set - {v} | {w}
                    # Verify all cliques covered
                    ok = True
                    for c in cliques_list:
                        if not new_set & c:
                            ok = False
                            break
                    if ok:
                        best_sol.remove(v)
                        best_sol.append(w)
                        found_swap = True
                        break
                if found_swap:
                    break

            if found_swap:
                # After a swap, try to remove vertices
                sol_set = set(best_sol)
                for v in list(best_sol):
                    new_set = sol_set - {v}
                    ok = True
                    for c in cliques_list:
                        if not new_set & c:
                            ok = False
                            break
                    if ok:
                        best_sol.remove(v)
                        best_size -= 1
                        break
                continue

            # Try 2-opt: remove two vertices, add one
            if len(best_sol) >= 2:
                sol_set = set(best_sol)
                found_2opt = False
                for i in range(len(best_sol)):
                    for j in range(i + 1, len(best_sol)):
                        v1, v2 = best_sol[i], best_sol[j]
                        new_set = sol_set - {v1, v2}
                        # Find uncovered cliques
                        uncovered = [c for c in cliques_list if not new_set & c]
                        if not uncovered:
                            best_sol.remove(v2)
                            best_sol.remove(v1)
                            best_size -= 2
                            found_2opt = True
                            break
                        # Check if a single vertex covers all uncovered
                        candidates = set(uncovered[0])
                        for c in uncovered[1:]:
                            candidates &= set(c)
                        candidates -= new_set
                        for w in candidates:
                            new_set2 = new_set | {w}
                            ok = True
                            for c in cliques_list:
                                if not new_set2 & c:
                                    ok = False
                                    break
                            if ok:
                                best_sol.remove(v2)
                                best_sol.remove(v1)
                                best_sol.append(w)
                                best_size -= 1
                                found_2opt = True
                                break
                        if found_2opt:
                            break
                    if found_2opt:
                        break
                if found_2opt:
                    continue

            # If no improvement, try a random perturbation
            if rng.random() < 0.05:
                candidates = [v for v in range(n) if v not in best_sol]
                if candidates:
                    v = rng.choice(candidates)
                    new_sol = best_sol + [v]
                    if is_valid(new_sol, cliques_list):
                        best_sol = new_sol
                        best_size = len(best_sol)
                        # Try to clean up
                        sol_set = set(best_sol)
                        for rv in list(best_sol):
                            new_set = sol_set - {rv}
                            ok = True
                            for c in cliques_list:
                                if not new_set & c:
                                    ok = False
                                    break
                            if ok:
                                best_sol.remove(rv)
                                best_size -= 1
                                break

        return best_sol

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]

        if not relevant_cliques:
            return []

        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 2:
            return greedy_init(relevant_cliques, n)

        # Get greedy solution
        sol = greedy_init(relevant_cliques, n)
        best_sol = sol
        best_size = len(sol)

        # Local search to improve
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        ls_iters = max(500, min(3000, int(time_left * 50)))
        sol2 = local_search(sol, relevant_cliques, n, max_iters=ls_iters)
        if len(sol2) < best_size:
            best_sol = sol2
            best_size = len(sol2)

        # Run local search again from the improved solution
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left > 5 and len(best_sol) > 0:
            ls_iters2 = max(200, min(1500, int(time_left * 30)))
            sol3 = local_search(best_sol, relevant_cliques, n, max_iters=ls_iters2)
            if len(sol3) < best_size:
                best_sol = sol3
                best_size = len(sol3)

        # Try MILP if time permits
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left > 10:
            try:
                from scipy.optimize import milp, LinearConstraint, Bounds
                m = len(relevant_cliques)
                A = np.zeros((m, n), dtype=np.int8)
                for i, clique in enumerate(relevant_cliques):
                    for v in clique:
                        A[i, v] = 1

                c = np.ones(n, dtype=np.float64)
                constraints = LinearConstraint(A, 1, np.inf)
                bounds_obj = Bounds(0, 1)
                integrality = np.ones(n, dtype=np.int8)

                # Use incumbent to help
                incumbent = np.zeros(n)
                for v in best_sol:
                    incumbent[v] = 1

                instance_time = min(time_left - 5, 20)
                res = milp(c=c, constraints=constraints, integrality=integrality,
                           bounds=bounds_obj,
                           options={"time_limit": instance_time, "presolve": True,
                                    "MIP_start": incumbent})

                if res.x is not None:
                    milp_sol = [i for i in range(n) if res.x[i] > 0.5]
                    if is_valid(milp_sol, relevant_cliques):
                        if len(milp_sol) < best_size:
                            best_sol = milp_sol
                            best_size = len(milp_sol)
            except Exception:
                pass

        return best_sol

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time, use greedy for remaining
            adj = random_graph(n, p, seed)
            all_cliques = bron_kerbosch(adj, n)
            relevant_cliques = [c for c in all_cliques if len(c) >= 2]
            sol = greedy_init(relevant_cliques, n) if relevant_cliques else []
            results.append(sorted(sol))
            continue

        sol = solve_instance(n, p, seed)
        results.append(sorted(sol))

    # Ensure we have 3 results
    while len(results) < 3:
        results.append([])

    return results[:3]
