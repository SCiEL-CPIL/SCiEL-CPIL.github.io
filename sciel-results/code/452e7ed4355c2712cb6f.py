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
                # Choose pivot that maximizes |P ∩ N(u)| to minimize branching
                max_count = -1
                pivot = None
                for u in union:
                    count = len(P & neighbors[u])
                    if count > max_count:
                        max_count = count
                        pivot = u
                if pivot is not None:
                    candidates = P - neighbors[pivot]
                else:
                    candidates = P
            else:
                candidates = P

            # Sort candidates by degree in P for better pruning
            for v in sorted(candidates, key=lambda x: len(P & neighbors[x]), reverse=True):
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
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques,
        with tie-breaking by fewer conflicts (vertices not in many other cliques)"""
        uncovered = [set(c) for c in cliques]
        covered = [False] * len(uncovered)
        selected = []
        
        # Precompute for each vertex, how many cliques it appears in
        vertex_clique_count = [0] * n
        for c in cliques:
            for v in c:
                vertex_clique_count[v] += 1

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
            # Pick vertex with max hits; tie-break by preferring vertices
            # that appear in fewer total cliques (less "wasteful")
            candidates = [v for v in range(n) if hits[v] == max_hits]
            # Secondary sort: fewer total clique appearances is better
            candidates.sort(key=lambda v: (vertex_clique_count[v], v))
            # Take from top few with randomness for diversity
            top_k = min(3, len(candidates))
            v = rng.choice(candidates[:top_k])
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

    def local_search(sol_set, cliques, n, rng, max_rounds=300):
        """Local search with removals, 1-1 swaps, and 2-1 swaps (remove 2, add 1)."""
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
            
            # Try 1-1 swap: remove v, add w where w covers all newly uncovered cliques
            for v in list(sol_set):
                test_set = sol_set - {v}
                uncovered = [c for c in cliques if not test_set & c]
                if not uncovered:
                    sol_set = test_set
                    improved = True
                    break
                # Find w in intersection of all uncovered cliques
                if len(uncovered) <= 10:  # Only do expensive check if few uncovered
                    intersection = set(uncovered[0])
                    for c in uncovered[1:]:
                        intersection &= c
                        if not intersection:
                            break
                    for w in intersection:
                        if w not in test_set:
                            sol_set = test_set | {w}
                            improved = True
                            break
                if improved:
                    break
            
            # Try 2-1 swap: remove two vertices, add one that covers all uncovered
            if not improved and len(sol_set) >= 3:
                sol_list = list(sol_set)
                for i in range(len(sol_list)):
                    for j in range(i + 1, len(sol_list)):
                        v1, v2 = sol_list[i], sol_list[j]
                        test_set = sol_set - {v1, v2}
                        uncovered = [c for c in cliques if not test_set & c]
                        if not uncovered:
                            sol_set = test_set
                            improved = True
                            break
                        if len(uncovered) <= 8:
                            intersection = set(uncovered[0])
                            for c in uncovered[1:]:
                                intersection &= c
                                if not intersection:
                                    break
                            for w in intersection:
                                if w not in test_set:
                                    sol_set = test_set | {w}
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
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
        
        # Dynamic number of restarts based on time
        num_restarts = 25

        for restart in range(num_restarts):
            if time.time() > deadline - 8:
                break

            rng = random.Random(1000 + restart * 7 + seed)

            # Greedy with random tie-breaking
            sol = greedy_init(relevant_cliques, n, rng)
            sol_set = set(sol)

            # Reduce
            sol_set = reduce_solution(sol_set, relevant_cliques, n)

            # Local search with 2-1 swaps
            sol_set = local_search(sol_set, relevant_cliques, n, rng, max_rounds=150)

            # Final reduce
            sol_set = reduce_solution(sol_set, relevant_cliques, n)

            if len(sol_set) < best_size:
                best_size = len(sol_set)
                best_sol = set(sol_set)

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
