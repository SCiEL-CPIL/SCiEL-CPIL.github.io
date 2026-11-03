"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """
    Solve the Quadratic Assignment Problem for a n x n instance.
    Combines efficient 2-opt local search with Hungarian-algorithm based
    global reassignment moves to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin within 120s limit

    n = len(a)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def compute_cost(p):
        """Compute the exact QAP cost for assignment p."""
        B_p = b[np.ix_(p, p)]
        return int(np.sum(a * B_p))

    def greedy_init():
        """Rank-matching: high-flow facilities to low-distance locations."""
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def swap_delta_fast(p, i, j):
        """
        Fast O(n) swap delta:
        delta = (a[i,:]-a[j,:]) @ (b[pj]-b[pi])[p] + (a[:,i]-a[:,j]) @ (b[:,pj]-b[:,pi])[p]
        """
        pi = p[i]
        pj = p[j]
        row_diff = (b[pj] - b[pi])[p]
        col_diff = (b[:, pj] - b[:, pi])[p]
        delta = np.dot(a[i, :] - a[j, :], row_diff) + np.dot(a[:, i] - a[:, j], col_diff)
        return delta

    def local_search(p, max_passes=100, time_limit=None):
        """2-opt local search with first-improvement and random offset."""
        if time_limit is None:
            time_limit = deadline
        p = p.copy()
        cost = compute_cost(p)
        n_local = len(p)

        for _pass in range(max_passes):
            if time.time() > time_limit:
                break

            improved = False
            offset = np.random.randint(n_local)
            for ii in range(n_local):
                i = (offset + ii) % n_local
                for jj in range(ii + 1, n_local):
                    j = (offset + jj) % n_local
                    if j == i:
                        continue
                    delta = swap_delta_fast(p, i, j)
                    if delta < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if time.time() > time_limit:
                    return p, cost

            if not improved:
                break

        return p, cost

    def hungarian_reassign(p, num_facilities=None):
        """
        Reassign a subset of facilities using the Hungarian algorithm.
        Builds a cost matrix where the cost of assigning facility i to location l
        is the marginal contribution: sum_k a[i,k]*b[l,p[k]] + sum_k a[k,i]*b[p[k],l]
        (excluding k=i and the other facilities being reassigned).
        """
        if num_facilities is None:
            num_facilities = n

        # Select facilities to reassign: pick the ones with highest "pressure"
        # or just random subset
        if num_facilities >= n:
            fac_indices = np.arange(n)
        else:
            fac_indices = np.random.choice(n, size=num_facilities, replace=False)

        # Build cost matrix for Hungarian
        # For each facility i in fac_indices and each location l (0..n-1):
        # cost[i, l] = sum_{k not in fac_indices, k != i} a[i,k]*b[l, p[k]] + a[k,i]*b[p[k], l]
        # We can vectorize this:
        # Let S = fac_indices, K = all indices not in S
        # For facility i in S, location l:
        #   term1 = sum_{k in K} a[i,k] * b[l, p[k]]
        #   term2 = sum_{k in K} a[k,i] * b[p[k], l]
        # Note: b is not necessarily symmetric, so both terms matter.

        mask = np.ones(n, dtype=bool)
        mask[fac_indices] = False
        K = np.where(mask)[0]  # indices NOT being reassigned

        if len(K) == 0:
            # All facilities being reassigned: cost is just a[i,j]*b[l,m] for i,j in S, l,m in S
            # This is the full QAP on the subset - just use identity or random
            # Actually, if all are being reassigned, the Hungarian cost matrix should be:
            # cost[i_idx, l] = sum_{j_idx in S} a[i,j] * b[l, p[j]]  ... but p[j] is changing
            # This gets complicated. Let's just do a random permutation for this case.
            new_p = p.copy()
            new_p[fac_indices] = np.random.permutation(n)
            return new_p

        pk = p[K]  # locations of non-reassigned facilities

        # cost_matrix shape: (num_facilities, n)
        # For each facility i in fac_indices, for each location l:
        # cost = a[i, K] @ b[l, pk] + a[K, i] @ b[pk, l]
        # = sum_k a[i,k]*b[l,pk] + sum_k a[k,i]*b[pk,l]

        # Vectorized: 
        # a[i, K] has shape (num_fac, len(K))
        # b[l, pk] has shape (n, len(K))
        # We need: for each i_idx, l: sum_k a[fac_indices[i_idx], K] * b[l, pk]
        # This is: a_subset @ b_l_pk^T where a_subset = a[fac_indices, K] (num_fac x len(K))
        # and b_l_pk = b[:, pk] (n x len(K)), so b_l_pk.T is (len(K) x n)
        # Result: (num_fac x n)

        a_subset = a[np.ix_(fac_indices, K)]  # (num_fac, len(K))
        b_l_pk = b[:, pk]  # (n, len(K))
        # term1[i_idx, l] = sum_k a_subset[i_idx, k] * b_l_pk[l, k]
        # = a_subset @ b_l_pk.T
        term1 = a_subset @ b_l_pk.T  # (num_fac, n)

        # term2[i_idx, l] = sum_k a[K, fac_indices[i_idx]] * b[pk, l]
        # a_K_i = a[K, fac_indices]  # (len(K), num_fac)
        # b_pk_l = b[pk, :]  # (len(K), n)
        # term2[i_idx, l] = sum_k a_K_i[k, i_idx] * b_pk_l[k, l]
        # = a_K_i.T @ b_pk_l  ... no
        # term2[i_idx, l] = sum_k a[K, i_idx] * b[pk, l]
        # Let a_K_i = a[np.ix_(K, fac_indices)]  # (len(K), num_fac)
        # Let b_pk_l = b[np.ix_(pk, np.arange(n))]  # (len(K), n)
        # term2[i_idx, l] = sum_k a_K_i[k, i_idx] * b_pk_l[k, l]
        # = (a_K_i.T @ b_pk_l)[i_idx, l]  ... no, that's sum_k a_K_i[k,i_idx]*b_pk_l[k,l]
        # Actually: a_K_i.T has shape (num_fac, len(K)), b_pk_l has shape (len(K), n)
        # a_K_i.T @ b_pk_l has shape (num_fac, n)
        # Element [i_idx, l] = sum_k a_K_i.T[i_idx, k] * b_pk_l[k, l] = sum_k a_K_i[k, i_idx] * b_pk_l[k, l]
        # Yes! That's exactly term2.

        a_K_i = a[np.ix_(K, fac_indices)]  # (len(K), num_fac)
        b_pk_l = b[np.ix_(pk, np.arange(n))]  # (len(K), n)
        term2 = a_K_i.T @ b_pk_l  # (num_fac, n)

        cost_matrix = term1 + term2  # (num_fac, n)

        # Solve Hungarian
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build new permutation
        new_p = p.copy()
        for idx, loc in zip(row_ind, col_ind):
            new_p[fac_indices[idx]] = loc

        return new_p

    def perturb(p, num_swaps):
        """Apply random multi-swap perturbation."""
        p = p.copy()
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        return p

    # Initial solution via rank matching
    best_p = greedy_init()
    best_cost = compute_cost(best_p)

    # First local search to convergence
    best_p, best_cost = local_search(best_p, max_passes=200)

    # Iterated local search with adaptive perturbation and Hungarian moves
    stall_count = 0
    stall_threshold = 4
    hungarian_interval = 3  # try Hungarian every 3 iterations

    while time.time() < deadline - 5.0:
        iteration = stall_count + 1

        # Decide move type
        if iteration % hungarian_interval == 0 and time.time() < deadline - 15.0:
            # Hungarian reassignment move
            # Start with a perturbation, then Hungarian
            p_pert = perturb(best_p, 2)
            # Try Hungarian with varying subset sizes
            subset_size = min(n, 32 + 8 * (iteration // hungarian_interval))
            p_new = hungarian_reassign(p_pert, num_facilities=min(subset_size, n))
            p_new, cost_new = local_search(p_new, max_passes=40)
        else:
            # Standard ILS move
            if stall_count < stall_threshold:
                num_swaps = 2
            elif stall_count < 2 * stall_threshold:
                num_swaps = 4
            else:
                num_swaps = 6

            p_pert = perturb(best_p, num_swaps)
            p_new, cost_new = local_search(p_pert, max_passes=80)

        if cost_new < best_cost:
            best_p = p_new
            best_cost = cost_new
            stall_count = 0
        else:
            if stall_count > 3 * stall_threshold and np.random.random() < 0.15:
                best_p = p_new
                best_cost = cost_new
            stall_count += 1

        # If heavily stalled, try fresh initialization
        if stall_count > 12:
            p_fresh = greedy_init()
            p_new, cost_new = local_search(p_fresh, max_passes=80)
            if cost_new < best_cost:
                best_p = p_new
                best_cost = cost_new
            stall_count = 0

    # Final polish
    if time.time() < deadline - 2.0:
        best_p, best_cost = local_search(best_p, max_passes=30)

    return best_p.tolist()

# EVOLVE-BLOCK-END
