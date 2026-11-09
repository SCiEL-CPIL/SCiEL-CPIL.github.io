import time
import random
import numpy as np
from itertools import combinations

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
            # Pick vertex with max hits, random tie-break
            candidates = [v for v in range(n) if hits[v] == max_hits]
            v = rng.choice(candidates)
            selected.append(v)
            for i, c in enumerate(uncovered):
                if not covered[i] and v in c:
                    covered[i] = True

        return selected

    def reduce_solution(sol, cliques, n):
        """Remove any vertex that is redundant (its removal keeps solution valid)."""
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

    def local_search_1swap(sol_set, cliques, n, rng, max_rounds=200):
        """Iteratively try 1-1 swaps: remove one vertex, if uncovered set is non-empty,
        try to cover all uncovered with a single new vertex (net -1 if uncovered was coverable by one).
        Also try pure removals (net -1 if any vertex is redundant)."""
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
            # Try 1-1 swap: for each v in sol, remove it, find uncovered cliques,
            # check if a single vertex w not in sol covers all uncovered
            for v in list(sol_set):
                test_set = sol_set - {v}
                uncovered = [c for c in cliques if not test_set & c]
                if not uncovered:
                    # This is a pure removal (should have been caught above, but safe)
                    sol_set = test_set
                    improved = True
                    break
                # Find vertices that cover all uncovered cliques
                # A vertex w covers all uncovered iff w is in every uncovered clique
                # (since we need test_set | {w} to intersect each uncovered clique,
                #  and test_set already misses each, so w must be in each)
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
                    # Size is same, but this may enable further removals
                    # To actually reduce size, we need this to enable a removal later.
                    # Mark improved to continue searching
                    improved = True
                    break
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

        # Multiple random restarts
        best_sol = None
        best_size = float('inf')
        num_restarts = 30  # tune based on time

        # Time budget per restart
        per_restart_time = max(0.5, (time_left - 5) / num_restarts)

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
            sol_set = local_search_1swap(sol_set, relevant_cliques, n, rng, max_rounds=100)

            # Final reduce
            sol_set = reduce_solution(sol_set, relevant_cliques, n)

            if len(sol_set) < best_size:
                best_size = len(sol_set)
                best_sol = set(sol_set)

            # Check time
            if time.time() - t0 > per_restart_time:
                # Continue with remaining restarts but tighter
                pass

        if best_sol is None:
            sol = greedy_init(relevant_cliques, n, random.Random(0))
            best_sol = reduce_solution(sol, relevant_cliques, n)

        return sorted(best_sol)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time, use greedy for remaining
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

    # Ensure we have 3 results
    while len(results) < 3:
        results.append([])

    return results[:3]
