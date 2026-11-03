"""Rewritten QAP solver for wil100 using Tabu Search with First-Improvement and ILS."""
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Precompute for fast delta computation
    # For swap (i, j) with locations u=p[i], v=p[j]:
    # delta = sum_{k!=i,j} (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]])
    #       + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
    #       + (a[i,j]-a[j,i])*(b[v,u]-b[u,v])

    # Precompute pair indices to avoid mask creation
    pair_indices = {}
    for i in range(n):
        for j in range(i + 1, n):
            idx = np.array([k for k in range(n) if k != i and k != j], dtype=np.int64)
            pair_indices[(i, j)] = idx

    def compute_cost(p):
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def compute_delta_swap(p, i, j):
        u = p[i]
        v = p[j]
        idx = pair_indices[(i, j)]
        pm = p[idx]

        term1 = np.dot(a[i, idx] - a[j, idx], b[v, pm] - b[u, pm])
        term2 = np.dot(a[idx, i] - a[idx, j], b[pm, v] - b[pm, u])
        term3 = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])

        return term1 + term2 + term3

    def first_improvement_tabu(p, tabu_list, tabu_tenure, best_cost, best_p):
        """Tabu search with first-improvement strategy."""
        n_local = n
        for iteration in range(500):
            if time.time() > deadline:
                break

            best_delta = 0
            best_pair = None
            best_tabu_age = 0

            for i in range(n_local):
                for j in range(i + 1, n_local):
                    d = compute_delta_swap(p, i, j)

                    # Check if this pair is tabu
                    is_tabu = False
                    tabu_age = 0
                    if (i, j) in tabu_list:
                        is_tabu = True
                        tabu_age = tabu_list[(i, j)]

                    # Accept if improving and not tabu, or if it's the best overall
                    if d < best_delta:
                        best_delta = d
                        best_pair = (i, j)
                        best_tabu_age = tabu_age
                        best_is_tabu = is_tabu

                    # First improvement: stop if we find an improving non-tabu move
                    if d < 0 and not is_tabu:
                        best_delta = d
                        best_pair = (i, j)
                        best_tabu_age = tabu_age
                        best_is_tabu = False
                        break

                if best_pair is not None and not best_is_tabu and best_delta < 0:
                    break

            if best_pair is None:
                break

            i, j = best_pair
            p[i], p[j] = p[j], p[i]

            # Update tabu list
            if (i, j) in tabu_list:
                del tabu_list[(i, j)]
            tabu_list[(i, j)] = iteration
            # Remove old entries
            old_keys = [k for k, v in tabu_list.items() if v < iteration - tabu_tenure]
            for k in old_keys:
                del tabu_list[k]

            cur_cost = compute_cost(p)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_p = p.copy()

        return best_cost, best_p

    def perturbation(p, num_swaps):
        """Random perturbation for ILS."""
        p = p.copy()
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        return p

    # Initial solution: greedy rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)

    best_cost = compute_cost(p)
    best_p = p.copy()

    np.random.seed(42)

    # Phase 1: Initial local search with tabu
    tabu_list = {}
    tabu_tenure = 10
    best_cost, best_p = first_improvement_tabu(p.copy(), tabu_list, tabu_tenure, best_cost, best_p)

    # Phase 2: Iterated Local Search with tabu
    num_ils_iter = 0
    while time.time() < deadline - 5.0:
        num_ils_iter += 1

        # Perturb the best solution
        num_swaps = min(2 + num_ils_iter // 20, 12)
        p = perturbation(best_p, num_swaps)

        # Run tabu search from perturbed solution
        tabu_list = {}
        tabu_tenure = 8 + (num_ils_iter % 5) * 2  # Vary tabu tenure
        cur_cost = compute_cost(p)
        cur_best_p = p.copy()

        best_cost_local, best_p_local = first_improvement_tabu(
            p, tabu_list, tabu_tenure, cur_cost, cur_best_p
        )

        if best_cost_local < best_cost:
            best_cost = best_cost_local
            best_p = best_p_local

    # Final polish: exhaustive 2-opt until no improvement
    if time.time() < deadline - 2.0:
        p = best_p.copy()
        while time.time() < deadline - 2.0:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    d = compute_delta_swap(p, i, j)
                    if d < 0:
                        p[i], p[j] = p[j], p[i]
                        improved = True
            if not improved:
                break
        cur_cost = compute_cost(p)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    return best_p.tolist()
