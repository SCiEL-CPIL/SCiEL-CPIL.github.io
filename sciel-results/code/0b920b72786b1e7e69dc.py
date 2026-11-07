"""Simulated annealing metaheuristic for Maximum Diversity Problem on induced subgraphs."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']

    # Precompute adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute all subsets of size >= minv as frozensets
    # For n=17, minv=8: C(17,8)+...+C(17,17) = 24310+12376+6188+2380+714+161+27+1 = 46197
    all_subsets = []
    for size in range(minv, n + 1):
        for combo in itertools.combinations(range(n), size):
            all_subsets.append(frozenset(combo))

    # Precompute distinct degree counts for all subsets
    # Use bitmask-based degree computation
    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c

    subset_scores = {}
    for s in all_subsets:
        mask = 0
        for v in s:
            mask |= (1 << v)
        degs = set()
        for v in s:
            degs.add(popcount(adj[v] & mask))
        subset_scores[s] = len(degs)

    # Find the maximum possible score
    max_score = max(subset_scores.values())

    # Check if we already found the global optimum
    if max_score >= 7:
        # Find a subset achieving max_score
        for s, sc in subset_scores.items():
            if sc == max_score:
                return [int(v) for v in sorted(s)]

    # Simulated annealing to find high-scoring subsets
    # Start from multiple random initial states
    best_score = 0
    best_subset = None

    # Multi-start simulated annealing
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 2.0:
            break

        # Random initial subset of size minv to n
        size = random.randint(minv, n)
        current = frozenset(random.sample(range(n), size))
        current_score = subset_scores[current]

        if current_score > best_score:
            best_score = current_score
            best_subset = current

        if best_score >= max_score:
            break

        # Simulated annealing parameters
        T = 1.0
        T_min = 0.001
        cooling = 0.995
        steps_per_restart = 5000

        for step in range(steps_per_restart):
            if time.time() > deadline - 2.0:
                break

            # Generate neighbor: swap one vertex in with one out, or add/remove one
            r = random.random()
            if r < 0.5 and len(current) > minv:
                # Remove one vertex
                v_out = random.choice(list(current))
                new_set = frozenset(current - {v_out})
            elif r < 0.7 and len(current) < n:
                # Add one vertex
                available = [v for v in range(n) if v not in current]
                if not available:
                    continue
                v_in = random.choice(available)
                new_set = frozenset(current | {v_in})
            else:
                # Swap: remove one, add one
                if len(current) == 0 or len(current) == n:
                    continue
                v_out = random.choice(list(current))
                available = [v for v in range(n) if v not in current]
                if not available:
                    continue
                v_in = random.choice(available)
                new_set = frozenset(current - {v_out} | {v_in})

            new_score = subset_scores[new_set]
            delta = new_score - current_score

            if delta >= 0:
                # Accept if better or equal
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_subset = current
                    if best_score >= max_score:
                        break
            else:
                # Accept with probability exp(delta / T)
                if random.random() < np.exp(delta / T):
                    current = new_set
                    current_score = new_score

            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min

        if best_score >= max_score:
            break

    # If we found the global optimum, return it
    if best_score >= max_score:
        return [int(v) for v in sorted(best_subset)]

    # Otherwise, do a final exhaustive check around the best solution
    # Try all subsets of size minv to n and keep the best
    # But we already did exhaustive precomputation, so best_subset should be optimal
    # unless we didn't reach max_score due to time

    # As a safety net, if best_subset is None, fall back to exhaustive
    if best_subset is None:
        for s, sc in subset_scores.items():
            if sc > best_score:
                best_score = sc
                best_subset = s

    return [int(v) for v in sorted(best_subset)]
# EVOLVE-BLOCK-END
