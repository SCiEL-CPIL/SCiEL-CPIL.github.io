# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

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
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
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
        """
        Vectorized 2-opt local search using R-matrix for O(n^2) swap evaluation.
        Returns (p, cost).
        """
        if max_time is None:
            max_time = deadline

        p = p.copy().astype(np.int64)
        current_cost = calc_cost_full(p)

        # Precompute R[i, l] = sum_j a[i,j] * b[l, p[j]]
        B_perm = b[np.ix_(np.arange(n), p)]  # B_perm[l, j] = b[l, p[j]]
        R = a @ B_perm.T  # R[i, l] = sum_j a[i,j] * b[l, p[j]]

        # Precompute C[i,k] = a[k,i] - a[i,k] + a[i,i] - a[k,k]
        diag_a = np.diag(a)
        C = (a.T - a) + diag_a[:, None] - diag_a[None, :]

        # B_diag = diag(b)
        B_diag = np.diag(b)

        idx = np.arange(n)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)  # upper triangle, i < k

        while time.time() < max_time:
            # Find the best swap (i, k) with i < k
            # Delta[i,k] = R[i, p[k]] - R[i, p[i]] + R[k, p[i]] - R[k, p[k]] + C[i,k] * (B_diag[p[k]] - B_diag[p[i]])

            # R[i, p[k]]: shape (n, n), entry (i,k) = R[i, p[k]]
            Ripk = R[idx[:, None], p[None, :]]

            # R[i, p[i]]: shape (n,), entry i = R[i, p[i]]
            Rii = R[idx, p]

            # R[k, p[i]]: entry (i,k) = R[k, p[i]]
            Rkpi = (R[:, p]).T  # Rkpi[i,k] = R[k, p[i]]

            # R[k, p[k]]: entry (i,k) = Rii[k]
            Rkk = np.tile(Rii[None, :], (n, 1))

            # B_diag[p[k]] - B_diag[p[i]]: entry (i,k) = B_diag[p[k]] - B_diag[p[i]]
            Bd_diff = B_diag[p[None, :]] - B_diag[p[:, None]]

            Delta = Ripk - Rii[:, None] + Rkpi - Rkk + C * Bd_diff
            Delta_masked = np.where(mask, Delta, np.inf)

            # Find the best swap
            best_idx = np.unravel_index(np.argmin(Delta_masked), (n, n))
            best_delta = Delta_masked[best_idx]

            if best_delta >= -1e-10:
                break  # No improving swap

            i, k = best_idx
            u, v = p[i], p[k]

            # Perform the swap
            p[i], p[k] = v, u

            # Update R
            a_diff = a[:, i] - a[:, k]  # shape (n,)
            b_diff = b[:, v] - b[:, u]  # shape (n,)
            R = R + np.outer(a_diff, b_diff)

            current_cost += best_delta

        return p, current_cost

    def spectral_init(a, b):
        """Generate an initial permutation using spectral embedding."""
        n = a.shape[0]
        M = a * b  # Hadamard product
        M_sym = (M + M.T) / 2.0
        try:
            eigvals, eigvecs = np.linalg.eigh(M_sym)
            idx = np.argsort(-np.abs(eigvals))
            eigvecs = eigvecs[:, idx]
            k = min(4, n)
            embed = eigvecs[:, :k]
            scores = embed[:, 0]
            order = np.argsort(scores, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[order] = np.arange(n)
            return p
        except Exception:
            heavy = np.argsort(-a.sum(axis=1), kind='stable')
            central = np.argsort(b.sum(axis=1), kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p

    # --- Phase 1: Initial Solution ---
    # Generate multiple initial permutations
    inits = []

    # 1. Spectral init
    p_spec = spectral_init(a, b)
    inits.append(('spectral', p_spec))

    # 2. Greedy init
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p_greedy = np.empty(n, dtype=np.int64)
    p_greedy[heavy_order] = central_order
    inits.append(('greedy', p_greedy))

    # 3. Random init
    np.random.seed(42)
    p_rand = np.random.permutation(n).astype(np.int64)
    inits.append(('random', p_rand))

    best_p = None
    best_cost = np.inf

    # Run local search on each initial permutation
    time_per_init = 15.0
    for name, p_init in inits:
        remaining = deadline - (time.time() - start_time)
        if remaining < 10.0:
            break

        t_limit = min(time_per_init, remaining - 5.0)
        if t_limit < 1.0:
            break

        p_local, cost = two_opt_polish(p_init, max_time=time.time() + t_limit)

        if cost < best_cost:
            best_cost = cost
            best_p = p_local.copy()

    # If no valid result yet, use greedy
    if best_p is None:
        best_p = p_greedy
        best_cost = calc_cost_full(best_p)

    # --- Phase 2: Hungarian Reassignment Loop ---
    # Alternating between Hungarian global reassignment and local polish
    hungarian_count = 0
    no_improve_rounds = 0
    max_no_improve = 3

    while time.time() < deadline - 20.0:
        # Compute linearized cost and solve assignment
        cost_mat = linearized_cost_matrix(best_p)

        # Add small random perturbation to break ties and escape plateaus
        if hungarian_count > 0 and no_improve_rounds > 0:
            noise_scale = 1.0 + no_improve_rounds * 0.5
            cost_mat = cost_mat + np.random.randn(n, n) * noise_scale

        try:
            row_ind, col_ind = linear_sum_assignment(cost_mat)
        except Exception:
            break

        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind

        new_cost = calc_cost_full(new_p)

        if new_cost < best_cost:
            best_p = new_p
            best_cost = new_cost
            hungarian_count += 1
            no_improve_rounds = 0

            # Polish after a good Hungarian step
            p_polished, p_cost = two_opt_polish(best_p, max_time=min(deadline - 10.0, time.time() + 10.0))
            if p_cost < best_cost:
                best_p = p_polished
                best_cost = p_cost
        else:
            no_improve_rounds += 1

            # If stuck, try a perturbed restart from best
            if no_improve_rounds >= max_no_improve:
                p = best_p.copy()
                # Randomly swap a few pairs to escape local minimum
                num_swaps = np.random.randint(5, 15)
                for _ in range(num_swaps):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                p, p_cost = two_opt_polish(p, max_time=min(deadline - 10.0, time.time() + 5.0))
                if p_cost < best_cost:
                    best_p = p
                    best_cost = p_cost
                no_improve_rounds = 0
                hungarian_count = 0

    # --- Phase 3: Final Polish ---
    p_final, final_cost = two_opt_polish(best_p, max_time=deadline - 2.0)
    if final_cost < best_cost:
        best_p = p_final
        best_cost = final_cost

    return best_p.tolist()
# EVOLVE-BLOCK-END
