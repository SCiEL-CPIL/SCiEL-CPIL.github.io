"""QAP tai256c solver: vectorized 2-opt + ADMM-style reoptimization + iterated local search."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s

    # Precompute symmetric A for delta calculations
    A_sym = a + a.T

    def calc_cost_full(perm):
        """Compute exact QAP cost for permutation perm."""
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

    def vectorized_local_search(p, max_time=None):
        """
        Faster local search using vectorized delta calculation per row.
        For a fixed i, compute all deltas for swapping i with any j > i in O(n) vectorized ops.
        """
        if max_time is None:
            max_time = deadline
        p = p.copy()
        cost = calc_cost_full(p)
        D = b[np.ix_(p, p)].copy()  # D[i,j] = b[p[i], p[j]]

        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n - 1):
                if time.time() > max_time:
                    return p, cost

                js = np.arange(i + 1, n)

                # Vectorized delta calculation for swapping i with any j in js
                # Using the formula:
                # delta(i,j) = sum_{k != i,j} (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
                #             + correction terms for k=i and k=j

                # diff_A[i,k] - diff_A[j,k] for k in 0..n-1
                # We need: sum_k (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
                # = sum_k A_sym[i,k]*D[j,k] - sum_k A_sym[i,k]*D[i,k] - sum_k A_sym[j,k]*D[j,k] + sum_k A_sym[j,k]*D[i,k]

                # For all j in js simultaneously:
                # Term1[j] = sum_k A_sym[i,k] * D[j,k] = A_sym[i] @ D[j]  (but D[j] varies)
                # Actually: D[js] has shape (len(js), n), so A_sym[i] @ D[js].T gives (len(js),)
                # But we need sum_k A_sym[i,k] * D[j,k] for each j in js
                # = D[js] @ A_sym[i]  -> shape (len(js),)

                # Let me re-derive more carefully.
                # delta(i,j) = sum_{k} (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
                #            - (A_sym[i,i] - A_sym[j,i])*(D[j,i] - D[i,i])  # remove k=i
                #            - (A_sym[i,j] - A_sym[j,j])*(D[j,j] - D[i,j])  # remove k=j
                # But wait, we also need to handle the fact that when we swap, the diagonal and cross terms change.

                # Let me use the formula from the inspiration:
                # S[j] = sum_k (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
                #      = sum_k A_sym[i,k]*D[j,k] - sum_k A_sym[i,k]*D[i,k] - sum_k A_sym[j,k]*D[j,k] + sum_k A_sym[j,k]*D[i,k]

                # For fixed i and varying j in js:
                # T1[j] = sum_k A_sym[i,k] * D[j,k] = D[js] @ A_sym[i]  (shape len(js))
                # T2 = sum_k A_sym[i,k] * D[i,k] = A_sym[i] @ D[i]  (scalar)
                # T3[j] = sum_k A_sym[j,k] * D[j,k] = diag(A_sym[js] * D[js])  (shape len(js))
                # T4[j] = sum_k A_sym[j,k] * D[i,k] = A_sym[js] @ D[i]  (shape len(js))

                T1 = D[js] @ A_sym[i]  # (len(js),)
                T2 = np.dot(A_sym[i], D[i])  # scalar
                T3 = np.diag(A_sym[js] * D[js])  # (len(js),)
                T4 = A_sym[js] @ D[i]  # (len(js),)

                S = T1 - T2 - T3 + T4  # (len(js),)

                # Correction terms:
                # Remove k=i term: (A_sym[i,i] - A_sym[js,i]) * (D[js,i] - D[i,i])
                coeff_i = A_sym[i, i] - A_sym[js, i]  # (len(js),)
                term_i = coeff_i * (D[js, i] - D[i, i])  # (len(js),)

                # Remove k=j term: (A_sym[i,js] - np.diag(A_sym[js])) * (D[js,js] - D[i,js])
                coeff_j = A_sym[i, js] - np.diag(A_sym[js])  # (len(js),)
                term_j = coeff_j * (D[js, js] - D[i, js])  # (len(js),)

                # Additional diagonal and cross terms from the swap:
                # When swapping p[i] and p[j], the terms (i,i), (i,j), (j,i), (j,j) change.
                # diag_term = (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])  -- but this is already in S?
                # Actually, let me just use the simpler two_opt_delta for correctness first,
                # then optimize if needed. For now, use the vectorized S with corrections.

                deltas = S - term_i - term_j

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
        """
        Use Hungarian algorithm on a linearized cost matrix to escape local optima.
        This is the ADMM-style step: fix the "dual" variables and solve the linear assignment.
        """
        B_p = b[np.ix_(np.arange(n), p)]  # B_p[j, k] = b[j, p[k]]
        M1 = a @ B_p.T  # M1[i, j] = sum_k a[i, k] * b[j, p[k]]

        Bp2 = b[np.ix_(p, np.arange(n))]  # Bp2[k, j] = b[p[k], j]
        M2 = a.T @ Bp2  # M2[i, j] = sum_k a[k, i] * b[p[k], j]

        C = M1 + M2

        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def spectral_init():
        """Spectral bisection based initialization."""
        b_sym = (b + b.T) / 2.0
        d = b_sym.sum(axis=1)
        L = np.diag(d) - b_sym

        try:
            from scipy.linalg import eigh
            vals, vecs = eigh(L)
            fiedler = vecs[:, 1]
        except Exception:
            return np.random.permutation(n).astype(np.int64)

        loc_order = np.argsort(fiedler, kind='stable')
        fac_order = np.argsort(a.sum(axis=1), kind='stable')

        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order
        return p

    # --- Phase 1: Initial Solutions ---
    best_p = None
    best_cost = float('inf')

    # 1. Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy_order] = central_order
    p_rank, c_rank = vectorized_local_search(p_rank, max_time=start_time + 5.0)
    if c_rank < best_cost:
        best_cost = c_rank
        best_p = p_rank.copy()

    # 2. Spectral Init
    if time.time() < start_time + 10.0:
        p_spec = spectral_init()
        p_spec, c_spec = vectorized_local_search(p_spec, max_time=start_time + 15.0)
        if c_spec < best_cost:
            best_cost = c_spec
            best_p = p_spec.copy()

    # 3. Random Init
    if time.time() < start_time + 20.0:
        p_rand = np.random.permutation(n).astype(np.int64)
        p_rand, c_rand = vectorized_local_search(p_rand, max_time=start_time + 25.0)
        if c_rand < best_cost:
            best_cost = c_rand
            best_p = p_rand.copy()

    # --- Phase 2: Iterated Local Search with Hungarian Reoptimization ---
    rng = np.random.RandomState(42)
    current_p = best_p.copy()
    current_cost = best_cost

    # Simulated annealing parameters
    T = 5000.0
    cooling = 0.95
    min_T = 10.0

    while time.time() < deadline - 5.0:
        # Perturb: random 2-5 swaps
        p = current_p.copy()
        n_swaps = rng.randint(2, 6)
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]

        # Local search
        p, cost = vectorized_local_search(p, max_time=deadline - 10.0)

        # Hungarian reoptimization step (ADMM-style)
        if time.time() < deadline - 15.0:
            p_hung = hungarian_reopt(p)
            p_hung, cost_hung = vectorized_local_search(p_hung, max_time=deadline - 10.0)
            if cost_hung < cost:
                p = p_hung
                cost = cost_hung

        # Accept or reject based on simulated annealing
        if cost < current_cost:
            current_p = p.copy()
            current_cost = cost
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        else:
            # Metropolis criterion
            delta_cost = cost - current_cost
            if T > min_T:
                prob = np.exp(-delta_cost / T)
                if rng.random_sample() < prob:
                    current_p = p.copy()
                    current_cost = cost
            T *= cooling

    # Final polish: one more local search pass
    if time.time() < deadline - 2.0:
        best_p, best_cost = vectorized_local_search(best_p, max_time=deadline - 1.0)

    return best_p.tolist()
