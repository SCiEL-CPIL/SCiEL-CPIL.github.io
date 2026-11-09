import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145

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
            union = P | X
            if union:
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

    def is_valid(sol_set, cliques):
        for c in cliques:
            if not sol_set & c:
                return False
        return True

    def greedy_init(cliques, n, rng):
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
            candidates = [v for v in range(n) if hits[v] == max_hits]
            v = rng.choice(candidates)
            selected.append(v)
            for i, c in enumerate(uncovered):
                if not covered[i] and v in c:
                    covered[i] = True

        return selected

    def reduce_solution(sol, cliques, n):
        sol_set = set(sol)
        changed = True
        while changed:
            changed = False
            for v in list(sol_set):
                test_set = sol_set - {v}
                if is_valid(test_set, cliques):
                    sol_set = test_set
                    changed = True
        return sol_set

    def local_search(sol_set, cliques, n, rng, max_rounds=300):
        sol_set = set(sol_set)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1

            for v in list(sol_set):
                test_set = sol_set - {v}
                if is_valid(test_set, cliques):
                    sol_set = test_set
                    improved = True
                    break
            if improved:
                continue

            swap_found = False
            for v in list(sol_set):
                test_set = sol_set - {v}
                uncovered = [c for c in cliques if not test_set & c]
                if not uncovered:
                    sol_set = test_set
                    improved = True
                    swap_found = True
                    break
                candidate = None
                first = uncovered[0]
                for w in first:
                    if w in test_set:
                        continue
                    ok = True
                    for c in uncovered:
                        if w not in c:
                            ok = False
                            break
                    if ok:
                        candidate = w
                        break
                if candidate is not None:
                    sol_set = test_set | {candidate}
                    improved = True
                    swap_found = True
                    break

            if improved or swap_found:
                continue

            sol_list = list(sol_set)
            two_swap_found = False
            for i in range(len(sol_list)):
                for j in range(i + 1, len(sol_list)):
                    v1, v2 = sol_list[i], sol_list[j]
                    if v1 not in sol_set or v2 not in sol_set:
                        continue
                    test_set = sol_set - {v1, v2}
                    uncovered = [c for c in cliques if not test_set & c]
                    if not uncovered:
                        sol_set = test_set
                        improved = True
                        two_swap_found = True
                        break

                    best_pair = None
                    first_unc = uncovered[0]
                    for w1 in first_unc:
                        if w1 in test_set:
                            continue
                        remaining = [c for c in uncovered if w1 not in c]
                        if not remaining:
                            continue
                        for w2 in remaining[0]:
                            if w2 in test_set or w2 == w1:
                                continue
                            ok = True
                            for c in remaining:
                                if w2 not in c:
                                    ok = False
                                    break
                            if ok:
                                best_pair = (w1, w2)
                                break
                        if best_pair:
                            break
                    if best_pair:
                        sol_set = test_set | {best_pair[0], best_pair[1]}
                        improved = True
                        two_swap_found = True
                        break
                if two_swap_found:
                    break

            if two_swap_found:
                continue

        return sol_set

    def ilp_solve(cliques, n, time_limit=10.0):
        """
        Solve the hitting set ILP using branch-and-bound with LP relaxation.
        Minimize sum(x_v) s.t. for each clique c: sum_{v in c} x_v >= 1, x_v in {0,1}
        """
        m = len(cliques)
        if m == 0:
            return set()

        # Build incidence: for each clique, list of vertices
        clique_vertices = [list(c) for c in cliques]

        # LP relaxation: use greedy fractional solution as initial bound
        # Simple approach: use greedy as upper bound, then branch and bound

        best_solution = None
        best_bound = float('inf')

        # Initial solution from greedy
        rng = random.Random(42)
        greedy_sol = greedy_init(cliques, n, rng)
        greedy_set = set(greedy_sol)
        greedy_set = reduce_solution(greedy_set, cliques, n)
        best_solution = set(greedy_set)
        best_bound = len(best_solution)

        # Branch and bound
        # State: (fixed_vars dict, lower_bound)
        # fixed_vars: {vertex: 0 or 1}

        import heapq
        import math

        # Priority queue: (lower_bound, fixed_vars)
        # We'll use a simple recursive approach with pruning

        def compute_lp_lower_bound(fixed_vars, clique_vertices, n):
            """
            Compute a lower bound on the number of additional vertices needed.
            For each uncovered clique, we need at least one more vertex.
            Use a simple greedy lower bound: count cliques that can't be covered by fixed 1s,
            then divide by max clique overlap.
            """
            # Cliques not yet satisfied by fixed 1s
            unsatisfied = []
            for cv in clique_vertices:
                satisfied = False
                for v in cv:
                    if fixed_vars.get(v, 0) == 1:
                        satisfied = True
                        break
                if not satisfied:
                    unsatisfied.append(cv)

            if not unsatisfied:
                return 0

            # Lower bound: at least ceil(number of unsatisfied cliques / max coverage per vertex)
            # Max coverage: for each unfixed vertex, how many unsatisfied cliques it covers
            unfixed_vertices = [v for v in range(n) if v not in fixed_vars]
            if not unfixed_vertices:
                return float('inf')

            max_cov = 0
            for v in unfixed_vertices:
                cov = 0
                for cv in unsatisfied:
                    if v in cv:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov

            if max_cov == 0:
                return float('inf')

            return math.ceil(len(unsatisfied) / max_cov)

        def branch_and_bound(fixed_vars, depth=0):
            nonlocal best_solution, best_bound

            if time.time() - start_time > deadline - 5:
                return

            # Check if all cliques are satisfied
            unsatisfied = []
            for cv in clique_vertices:
                satisfied = False
                for v in cv:
                    if fixed_vars.get(v, 0) == 1:
                        satisfied = True
                        break
                if not satisfied:
                    unsatisfied.append(cv)

            if not unsatisfied:
                # All satisfied, compute solution size
                sol_size = sum(1 for v in range(n) if fixed_vars.get(v, 0) == 1)
                if sol_size < best_bound:
                    best_bound = sol_size
                    best_solution = set(v for v in range(n) if fixed_vars.get(v, 0) == 1)
                return

            # Lower bound
            lb = compute_lp_lower_bound(fixed_vars, clique_vertices, n)
            current_fixed_ones = sum(1 for v in range(n) if fixed_vars.get(v, 0) == 1)
            total_lb = current_fixed_ones + lb

            if total_lb >= best_bound:
                return

            # Choose a vertex to branch on
            # Pick vertex from the first unsatisfied clique with most coverage
            first_clique = unsatisfied[0]
            candidates = [v for v in first_clique if v not in fixed_vars]
            if not candidates:
                # Dead end
                return

            # Score candidates by how many unsatisfied cliques they cover
            def score(v):
                return sum(1 for cv in unsatisfied if v in cv)

            candidates.sort(key=score, reverse=True)
            v = candidates[0]

            # Branch: x_v = 1
            new_fixed_1 = dict(fixed_vars)
            new_fixed_1[v] = 1
            branch_and_bound(new_fixed_1, depth + 1)

            # Branch: x_v = 0
            new_fixed_0 = dict(fixed_vars)
            new_fixed_0[v] = 0
            branch_and_bound(new_fixed_0, depth + 1)

        branch_and_bound({})

        if best_solution is not None:
            return best_solution
        return set(greedy_init(cliques, n, random.Random(0)))

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]

        if not relevant_cliques:
            return []

        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 3:
            sol = greedy_init(relevant_cliques, n, random.Random(0))
            sol_set = reduce_solution(sol, relevant_cliques, n)
            return sorted(sol_set)

        # Try ILP first
        ilp_time_limit = min(30, time_left * 0.4)
        ilp_sol = ilp_solve(relevant_cliques, n, time_limit=ilp_time_limit)
        best_sol = set(ilp_sol)
        best_size = len(best_sol)

        # Polish with local search
        rng = random.Random(seed)
        best_sol = local_search(best_sol, relevant_cliques, n, rng, max_rounds=200)
        best_sol = reduce_solution(best_sol, relevant_cliques, n)
        best_size = len(best_sol)

        # Try multiple greedy restarts to potentially beat ILP
        num_restarts = min(20, max(5, int((time_left - 10) / 2)))
        for restart in range(num_restarts):
            if time.time() - start_time > deadline - 5:
                break
            rng = random.Random(1000 + restart * 7 + seed)
            sol = greedy_init(relevant_cliques, n, rng)
            sol_set = reduce_solution(set(sol), relevant_cliques, n)
            sol_set = local_search(sol_set, relevant_cliques, n, rng, max_rounds=100)
            sol_set = reduce_solution(sol_set, relevant_cliques, n)
            if len(sol_set) < best_size:
                best_size = len(sol_set)
                best_sol = set(sol_set)

        return sorted(best_sol)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            adj = random_graph(n, p, seed)
            all_cliques = bron_kerbosch(adj, n)
            relevant_cliques = [c for c in all_cliques if len(c) >= 2]
            if relevant_cliques:
                sol = greedy_init(relevant_cliques, n, random.Random(0))
                sol_set = reduce_solution(sol, relevant_cliques, n)
                results.append(sorted(sol_set))
            else:
                results.append([])
            continue

        sol = solve_instance(n, p, seed)
        results.append(sorted(sol))

    while len(results) < 3:
        results.append([])

    return results[:3]
