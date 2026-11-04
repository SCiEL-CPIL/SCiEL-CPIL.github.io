"""Combined QAP solver for tai256c: ILS with vectorized 2-opt, Hungarian re-optimization,
eigen-based initialization, and MCMC-style block-reversal perturbations."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    def compute_cost(p):
        D = b[np.ix_(p, p)]
        return int(np.sum(a * D))

    def local_search(p, cost, max_passes=20):
        """Steepest Descent Local Search with vectorized delta calculation for speed."""
        improved = True
        passes = 0
        p = p.copy()

        # Precompute symmetric A for vectorized delta calculation
        A_sym = a + a.T
        A_diag = np.diag(a)
        D = b[np.ix_(p, p)].copy()

        while improved and passes < max_passes:
            improved = False
            passes += 1

            for i in range(n - 1):
                if time.time() > deadline:
                    return p, cost

                js = np.arange(i + 1, n)

                # Vectorized delta calculation
                diff_A = A_sym[i] - A_sym[js]
                diff_D = D[js] - D[i]

                S = np.sum(diff_A * diff_D, axis=1)

                # Correction terms
                coeff_i = A_sym[i, i] - A_sym[js, i]
                term_i = coeff_i * (D[js, i] - D[i, i])

                coeff_j = A_sym[i, js] - A_diag[js]
                term_j = coeff_j * (D[js, js] - D[i, js])

                diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
                cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])

                deltas = S - term_i - term_j + diag_term + cross_term

                min_idx = np.argmin(deltas)
                min_delta = deltas[min_idx]

                if min_delta < 0:
                    j = js[min_idx]
                    p[i], p[j] = p[j], p[i]

                    # Update D
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()

                    cost += int(min_delta)
                    improved = True

        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]

        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]

        C = M1 + M2

        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def hungarian_init():
        """Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix."""
        cost_matrix = a @ b + a.T @ b.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def eigen_init():
        """Generate a permutation using eigenvalue-based initialization.
        Projects facilities and locations onto principal eigenvectors of A and B,
        then aligns them by sorting."""
        A_sym = (a + a.T) / 2.0
        B_sym = (b + b.T) / 2.0

        try:
            w_a, v_a = np.linalg.eigh(A_sym)
            w_b, v_b = np.linalg.eigh(B_sym)
        except Exception:
            return np.random.permutation(n).astype(np.int64)

        # Take top k eigenvectors (largest eigenvalues)
        k = min(10, n)
        v_a_top = v_a[:, -k:]
        v_b_top = v_b[:, -k:]

        # Align signs so that the dot product is positive
        if np.dot(v_a_top[:, 0], v_b_top[:, 0]) < 0:
            v_b_top[:, 0] = -v_b_top[:, 0]

        # Use the first eigenvector for primary sort
        coords_a = v_a_top[:, 0]
        coords_b = v_b_top[:, 0]

        order_a = np.argsort(coords_a, kind="stable")
        order_b = np.argsort(coords_b, kind="stable")

        p = np.empty(n, dtype=np.int64)
        p[order_a] = order_b
        return p

    def rank_init():
        """Generate an initial assignment based on rank matching of flow and distance structures."""
        a_row_sums = a.sum(axis=1)
        b_col_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_col_sums, kind="stable")
        p_rank = np.empty(n, dtype=np.int64)
        p_rank[heavy] = central
        return p_rank

    def block_reverse_perturb(p, num_reversals=2, min_len=3, max_len=15):
        """MCMC-style perturbation: reverse random subsequences to escape local optima."""
        p = p.copy()
        for _ in range(num_reversals):
            if len(p) <= 2:
                break
            max_start = max(1, len(p) - min_len)
            start = np.random.randint(0, max_start + 1)
            max_end = min(len(p), start + max_len)
            min_end = min(len(p), start + min_len)
            if max_end <= min_end:
                continue
            end = np.random.randint(min_end, max_end + 1)
            p[start:end] = p[start:end][::-1]
        return p

    def random_swap_perturb(p, num_swaps):
        """Randomly swap pairs of positions."""
        p = p.copy()
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        return p

    # Phase 1: Hungarian Initialization
    p_hung = hungarian_init()
    best_p = p_hung.copy()
    best_cost = compute_cost(best_p)

    # Refine with local search
    current_p, current_cost = local_search(best_p, best_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Phase 2: Eigen initialization (alternative starting point)
    if time.time() < deadline - 60:
        p_eig = eigen_init()
        p_eig_ls, cost_eig = local_search(p_eig, compute_cost(p_eig))
        if cost_eig < best_cost:
            best_p = p_eig_ls.copy()
            best_cost = cost_eig

    # Phase 3: Rank matching initialization (alternative starting point)
    if time.time() < deadline - 40:
        p_rank = rank_init()
        p_rank_ls, cost_rank = local_search(p_rank, compute_cost(p_rank))
        if cost_rank < best_cost:
            best_p = p_rank_ls.copy()
            best_cost = cost_rank

    # Phase 4: Iterated Local Search with mixed perturbations
    iteration = 0
    consecutive_no_improve = 0
    base_perturb = 3

    while time.time() < deadline - 10:
        iteration += 1

        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb

        # Choose perturbation type: block reversal vs random swaps
        if iteration % 3 == 0:
            # MCMC-style block reversal
            p = block_reverse_perturb(best_p, num_reversals=max(1, perturb_k // 3),
                                      min_len=3, max_len=15)
        else:
            # Random swaps
            n_swaps = np.random.randint(2, perturb_k + 1)
            p = random_swap_perturb(best_p, n_swaps)

        # Local search from perturbed
        p, cost = local_search(p, compute_cost(p))

        # Every few iterations, try Hungarian re-optimization
        if iteration % 5 == 0 and time.time() < deadline - 25:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    return best_p.tolist()
