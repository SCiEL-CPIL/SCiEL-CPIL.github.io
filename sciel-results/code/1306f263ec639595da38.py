# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # --- Precompute helpers ---
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def two_opt_delta(p, i, j):
        """Compute delta cost for swapping facilities i and j in permutation p."""
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        # d[k] = b[lj, p[k]] - b[li, p[k]] for k != i,j
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        # a_diff[k] = a[i,k] - a[j,k] + a[k,i] - a[k,j] for k != i,j
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        return int(np.dot(a_diff, d))

    def linearized_cost_matrix(p):
        """
        Compute the linearized cost matrix for Hungarian algorithm.
        cost[i,l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        """
        C1 = b[:, p]  # C1[l, j] = b[l, p[j]]
        C2 = b[p, :]  # C2[j, l] = b[p[j], l]
        term1 = a @ C1.T  # term1[i,l] = sum_j a[i,j] * b[l, p[j]]
        term2 = a.T @ C2  # term2[i,l] = sum_j a[j,i] * b[p[j], l]
        return term1 + term2

    def two_opt_polish(p, max_time=None):
        """Greedy 2-opt local search. Returns (p, cost)."""
        cost = calc_cost_full(p)
        if max_time is None:
            max_time = deadline
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    # --- Phase 1: Initial Solution via Rank Matching ---
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    p = p.copy()

    # Initial polish
    p, current_cost = two_opt_polish(p, max_time=start_time + 12.0)
    best_p = p.copy()
    best_cost = current_cost

    # --- Phase 2: Hungarian Reassignment Loop with perturbation ---
    hungarian_count = 0
    no_improve_rounds = 0
    max_no_improve = 3
    rng = np.random.RandomState(42)

    while time.time() < deadline - 25.0:
        # Compute linearized cost and solve assignment
        cost_mat = linearized_cost_matrix(p)

        # Add small random perturbation to break ties and escape plateaus
        if hungarian_count > 0 and no_improve_rounds > 0:
            noise_scale = 1.0 + no_improve_rounds * 0.5
            cost_mat = cost_mat + rng.randn(n, n) * noise_scale

        try:
            row_ind, col_ind = linear_sum_assignment(cost_mat)
        except Exception:
            break

        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind

        new_cost = calc_cost_full(new_p)

        if new_cost < best_cost:
            p = new_p
            best_p = p.copy()
            best_cost = new_cost
            hungarian_count += 1
            no_improve_rounds = 0

            # Polish after a good Hungarian step
            p, p_cost = two_opt_polish(p, max_time=min(deadline - 12.0, time.time() + 8.0))
            if p_cost < best_cost:
                best_p = p.copy()
                best_cost = p_cost
        else:
            no_improve_rounds += 1

            # If stuck, try a perturbed restart from best
            if no_improve_rounds >= max_no_improve:
                p = best_p.copy()
                # Randomly swap a few pairs to escape local minimum
                num_swaps = rng.randint(5, 15)
                for _ in range(num_swaps):
                    i = rng.randint(n)
                    j = rng.randint(n)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                p, p_cost = two_opt_polish(p, max_time=min(deadline - 12.0, time.time() + 5.0))
                if p_cost < best_cost:
                    best_p = p.copy()
                    best_cost = p_cost
                no_improve_rounds = 0
                hungarian_count = 0

    # --- Phase 3: Final Polish ---
    p_polish = best_p.copy()
    p_polish, final_cost = two_opt_polish(p_polish, max_time=deadline - 2.0)
    if final_cost < best_cost:
        best_p = p_polish
        best_cost = final_cost

    return best_p.tolist()

# EVOLVE-BLOCK-END
