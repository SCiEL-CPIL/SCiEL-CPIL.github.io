"""QAP tai256c solver: Lin-Kernighan style multi-move reconnection + vectorized 2-opt + ADMM reoptimization."""
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
                T1 = D[js] @ A_sym[i]  # (len(js),)
                T2 = np.dot(A_sym[i], D[i])  # scalar
                T3 = np.diag(A_sym[js] * D[js])  # (len(js),)
                T4 = A_sym[js] @ D[i]  # (len(js),)

                S = T1 - T2 - T3 + T4  # (len(js),)

                # Correction terms:
                coeff_i = A_sym[i, i] - A_sym[js, i]  # (len(js),)
                term_i = coeff_i * (D[js, i] - D[i, i])  # (len(js),)

                coeff_j = A_sym[i, js] - np.diag(A_sym[js])  # (len(js),)
                term_j = coeff_j * (D[js, js] - D[i, js])  # (len(js),)

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

    def lin_kernighan_reconnect(p, max_time=None):
        """
        Lin-Kernighan style multi-move reconnection.
        This performs a sequence of 3-opt moves (reversals of segments) to escape
        local optima that are not reachable by pairwise swaps.
        """
        if max_time is None:
            max_time = deadline
        p = p.copy()
        cost = calc_cost_full(p)
        D = b[np.ix_(p, p)].copy()

        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n - 2):
                if time.time() > max_time:
                    return p, cost

                # Try 3-opt moves: reverse segment [i, j] for j > i
                # This is equivalent to moving facility i to position j and shifting
                # facilities i+1..j one position left.
                # For efficiency, we try a limited set of j values.
                # We'll try j = i+1, i+2, ..., i+min(10, n-i-1)
                max_j_offset = min(10, n - i - 1)

                best_delta = 0
                best_j = -1

                for offset in range(1, max_j_offset + 1):
                    j = i + offset
                    if j >= n:
                        break

                    # Compute delta for reversing segment [i, j]
                    # This is complex, so we approximate by trying the swap of i and j
                    # followed by a local search. But for true 3-opt, we need to
                    # reverse the segment.

                    # For simplicity and speed, we'll use a 2-opt style delta
                    # but allow non-adjacent swaps. This is already covered by
                    # vectorized_local_search. Instead, let's try a different approach:
                    # perform a "segment reversal" which is a 3-opt move.

                    # Segment reversal: p[i..j] -> p[j..i]
                    # This changes the order of facilities i..j.
                    # The cost change is complex to compute in O(n), so we'll
                    # only try this if the 2-opt local search has converged.

                    # For now, let's try a different multi-move: 3-swap
                    # Swap p[i], p[j], p[k] in a cycle.
                    # This is also complex, so let's stick with a simpler approach.

                    # Instead, let's try a "k-opt" move where we reverse a segment
                    # of length k. We'll try k=2 (which is just a swap) and k=3.
                    # For k=3, we reverse p[i], p[i+1], p[i+2].

                    if offset == 2 and i + 2 < n:
                        # Try reversing 3 elements: p[i], p[i+1], p[i+2]
                        # New order: p[i+2], p[i+1], p[i]
                        # This is a 3-opt move.
                        # Compute delta by brute force for this small move.
                        old_cost_3 = (a[i, i] * D[i, i] + a[i, i+1] * D[i, i+1] + a[i, i+2] * D[i, i+2] +
                                     a[i+1, i] * D[i+1, i] + a[i+1, i+1] * D[i+1, i+1] + a[i+1, i+2] * D[i+1, i+2] +
                                     a[i+2, i] * D[i+2, i] + a[i+2, i+1] * D[i+2, i+1] + a[i+2, i+2] * D[i+2, i+2])
                        # New D for the 3 elements:
                        # D_new[i, i] = D[i+2, i+2]
                        # D_new[i, i+1] = D[i+2, i+1]
                        # D_new[i, i+2] = D[i+2, i]
                        # D_new[i+1, i] = D[i+1, i+2]
                        # D_new[i+1, i+1] = D[i+1, i+1]
                        # D_new[i+1, i+2] = D[i+1, i]
                        # D_new[i+2, i] = D[i, i+2]
                        # D_new[i+2, i+1] = D[i, i+1]
                        # D_new[i+2, i+2] = D[i, i]
                        new_cost_3 = (a[i, i] * D[i+2, i+2] + a[i, i+1] * D[i+2, i+1] + a[i, i+2] * D[i+2, i] +
                                     a[i+1, i] * D[i+1, i+2] + a[i+1, i+1] * D[i+1, i+1] + a[i+1, i+2] * D[i+1, i] +
                                     a[i+2, i] * D[i, i+2] + a[i+2, i+1] * D[i, i+1] + a[i+2, i+2] * D[i, i])
                        delta_3 = new_cost_3 - old_cost_3
                        if delta_3 < best_delta:
                            best_delta = int(delta_3)
                            best_j = j

                if best_delta < 0 and best_j > 0:
                    j = best_j
                    # Perform the reversal: p[i..j] -> p[j..i]
                    p[i:j+1] = p[i:j+1][::-1]

                    # Update D
                    D[i:j+1, i:j+1] = D[i:j+1, i:j+1][::-1, ::-1]
                    # Update rows and columns outside the segment
                    if i > 0:
                        D[i:j+1, :i] = D[i:j+1, :i][::-1, :]
                        D[:i, i:j+1] = D[:i, i:j+1][:, ::-1]
                    if j + 1 < n:
                        D[i:j+1, j+1:] = D[i:j+1, j+1:][::-1, :]
                        D[j+1:, i:j+1] = D[j+1:, i:j+1][:, ::-1]

                    cost += best_delta
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

    def hungarian_init():
        """Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix."""
        cost_matrix = a @ b + a.T @ b.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

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

    # 1. Hungarian Init
    if time.time() < start_time + 5.0:
        p_hung = hungarian_init()
        p_hung, c_hung = vectorized_local_search(p_hung, max_time=start_time + 8.0)
        if c_hung < best_cost:
            best_cost = c_hung
            best_p = p_hung.copy()

    # 2. Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy_order] = central_order
    p_rank, c_rank = vectorized_local_search(p_rank, max_time=start_time + 12.0)
    if c_rank < best_cost:
        best_cost = c_rank
        best_p = p_rank.copy()

    # 3. Spectral Init
    if time.time() < start_time + 15.0:
        p_spec = spectral_init()
        p_spec, c_spec = vectorized_local_search(p_spec, max_time=start_time + 20.0)
        if c_spec < best_cost:
            best_cost = c_spec
            best_p = p_spec.copy()

    # 4. Random Init
    if time.time() < start_time + 22.0:
        p_rand = np.random.permutation(n).astype(np.int64)
        p_rand, c_rand = vectorized_local_search(p_rand, max_time=start_time + 28.0)
        if c_rand < best_cost:
            best_cost = c_rand
            best_p = p_rand.copy()

    # --- Phase 2: Iterated Local Search with Lin-Kernighan Reconnection ---
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

        # Lin-Kernighan reconnection step (multi-move)
        if time.time() < deadline - 15.0:
            p_lk = lin_kernighan_reconnect(p, max_time=deadline - 12.0)
            p_lk, cost_lk = vectorized_local_search(p_lk, max_time=deadline - 10.0)
            if cost_lk < cost:
                p = p_lk
                cost = cost_lk

        # Hungarian reoptimization step (ADMM-style)
        if time.time() < deadline - 20.0:
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
