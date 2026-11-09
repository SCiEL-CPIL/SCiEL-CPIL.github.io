import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

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
        """Returns list of maximal cliques as frozensets using pivot rule."""
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
        """Greedy set cover with random tie-breaking."""
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
        """Remove any vertex that is redundant."""
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
        """Iterative local search with removals, 1-swaps, and 2-swaps."""
        sol_set = set(sol_set)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1

            # Try pure removals
            for v in list(sol_set):
                test_set = sol_set - {v}
                if is_valid(test_set, cliques):
                    sol_set = test_set
                    improved = True
                    break
            if improved:
                continue

            # Try 1-1 swap
            swap_found = False
            for v in list(sol_set):
                test_set = sol_set - {v}
                uncovered = [c for c in cliques if not test_set & c]
                if not uncovered:
                    sol_set = test_set
                    improved = True
                    swap_found = True
                    break
                # Find vertex w not in sol_set that covers all uncovered cliques
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

            # Try 2-swap: remove two vertices, add two new vertices
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

                    # Find two vertices w1, w2 not in test_set
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

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]

        if not relevant_cliques:
            return []

        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 2:
            sol = greedy_init(relevant_cliques, n, random.Random(0))
            sol_set = reduce_solution(sol, relevant_cliques, n)
            return sorted(sol_set)

        # Multiple random restarts with varied strategies
        best_sol = None
        best_size = float('inf')
        num_restarts = 50

        per_restart_time = max(0.3, (time_left - 5) / num_restarts)

        for restart in range(num_restarts):
            elapsed = time.time() - start_time
            if time.time() - start_time > deadline - 3:
                break

            rng = random.Random(1000 + restart * 7 + seed)
            t0 = time.time()

            # Greedy with random tie-breaking
            sol = greedy_init(relevant_cliques, n, rng)
            sol_set = set(sol)

            # Reduce
            sol_set = reduce_solution(sol_set, relevant_cliques, n)

            # Local search
            sol_set = local_search(sol_set, relevant_cliques, n, rng, max_rounds=200)

            # Final reduce
            sol_set = reduce_solution(sol_set, relevant_cliques, n)

            if len(sol_set) < best_size:
                best_size = len(sol_set)
                best_sol = set(sol_set)

            # Check time
            if time.time() - t0 > per_restart_time:
                pass

        if best_sol is None:
            sol = greedy_init(relevant_cliques, n, random.Random(0))
            best_sol = reduce_solution(sol, relevant_cliques, n)

        return sorted(best_sol)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
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
