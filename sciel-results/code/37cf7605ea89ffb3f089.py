"""QAP solver for tho150: Linearized Greedy Construction + efficient 2-opt local search."""
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64, copy=False)
    b = b.astype(np.int64, copy=False)

    def compute_cost(p_arr):
        """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
        bp = b[p_arr]  # bp[i,j] = b[p[i], p[j]]
        return int(np.sum(a * bp))

    # --- Linearized Greedy Construction ---
    # Build permutation by assigning facilities one at a time.
    # Maintain a dynamic cost matrix C[i,j] = marginal cost of placing facility i at location j
    # given the current partial assignment.
    #
    # The full cost is sum_{i,j} a[i,j] b[p[i], p[j]].
    # When we assign facility i to location j, the contribution involving i is:
    #   sum_k a[i,k] b[j, p[k]] + sum_k a[k,i] b[p[k], j]
    # where p[k] is already assigned for previously placed facilities.
    #
    # We can maintain for each unassigned facility i and each location j:
    #   C[i,j] = sum_{k in assigned} a[i,k] * b[j, p[k]] + sum_{k in assigned} a[k,i] * b[p[k], j]
    #
    # When we assign facility f to location l, we update C for all remaining unassigned facilities i:
    #   C[i, j] += a[i, f] * b[j, l] + a[f, i] * b[l, j]
    #
    # This is O(n) per facility i update, and we do it for O(n) unassigned facilities,
    # so O(n^2) per assignment step. Total: O(n^3) for construction.
    # For n=150, that's 3.375M operations, very fast.

    def greedy_construct():
        """Build a permutation using linearized greedy construction."""
        assigned = np.zeros(n, dtype=bool)
        p = np.full(n, -1, dtype=np.int64)
        # C[i, j] = accumulated marginal cost of placing facility i at location j
        C = np.zeros((n, n), dtype=np.int64)

        for step in range(n):
            # Find the best (facility, location) pair among unassigned
            # Mask assigned facilities and locations
            C_masked = C.copy()
            C_masked[assigned, :] = np.iinfo(np.int64).max
            # Also mask already-used locations
            used_locs = p[p >= 0]
            if len(used_locs) > 0:
                C_masked[:, used_locs] = np.iinfo(np.int64).max

            # Find minimum
            min_idx = np.unravel_index(np.argmin(C_masked), C_masked.shape)
            f, l = min_idx[0], min_idx[1]

            p[f] = l
            assigned[f] = True

            # Update C for all unassigned facilities
            # C[i, j] += a[i, f] * b[j, l] + a[f, i] * b[l, j]
            # For all i not assigned, all j:
            unassigned_mask = ~assigned
            unassigned_mask[f] = False  # f is now assigned
            if np.any(unassigned_mask):
                # C[unassigned, :] += a[unassigned, f][:, None] * b[:, l][None, :] + a[f, unassigned][:, None] * b[l, :][None, :]
                C[unassigned_mask, :] += a[unassigned_mask, f][:, None] * b[:, l][None, :]
                C[unassigned_mask, :] += a[f, unassigned_mask][:, None] * b[l, :][None, :]

        return p

    # --- Efficient 2-opt (swap) local search ---
    # Precompute for fast delta computation.
    # For a swap of facilities i and j (locations old_i, old_j):
    # delta = sum_{k != i,j} [(a[i,k]-a[j,k])*(b[old_j,p[k]]-b[old_i,p[k]]) 
    #                        + (a[k,i]-a[k,j])*(b[p[k],old_j]-b[p[k],old_i])]
    #       + a[i,i]*(b[old_j,old_j]-b[old_i,old_i])
    #       + a[j,j]*(b[old_i,old_i]-b[old_j,old_j])
    #       + (a[i,j]-a[j,i])*(b[old_j,old_i]-b[old_i,old_j])

    def local_search(p_arr, cur_cost, time_limit):
        """Best-improvement swap local search until no improvement or time limit."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1

            for i in range(n):
                if time.time() > time_limit:
                    break
                old_i_loc = p_arr[i]
                # Precompute diff vectors for facility i
                # We'll compute delta for each j > i
                for j in range(i + 1, n):
                    old_j_loc = p_arr[j]

                    # Vectorized delta computation
                    # diff_b[k] = b[old_j_loc, k] - b[old_i_loc, k]
                    # diff_bt[k] = b[k, old_j_loc] - b[k, old_i_loc]
                    diff_b = b[old_j_loc] - b[old_i_loc]
                    diff_bt = b[:, old_j_loc] - b[:, old_i_loc]

                    # S1 = sum_k (a[i,k] - a[j,k]) * diff_b[p_arr[k]]
                    # S2 = sum_k (a[k,i] - a[k,j]) * diff_bt[p_arr[k]]
                    S1 = int(np.dot(a[i] - a[j], diff_b[p_arr]))
                    S2 = int(np.dot(a[:, i] - a[:, j], diff_bt[p_arr]))

                    # Subtract k=i and k=j contributions
                    S1_ii = (a[i, i] - a[j, i]) * diff_b[old_i_loc]
                    S1_ij = (a[i, j] - a[j, j]) * diff_b[old_j_loc]
                    S2_ii = (a[i, i] - a[i, j]) * diff_bt[old_i_loc]
                    S2_ij = (a[j, i] - a[j, j]) * diff_bt[old_j_loc]

                    delta = S1 + S2 - S1_ii - S1_ij - S2_ii - S2_ij

                    # Special terms
                    delta += a[i, i] * (b[old_j_loc, old_j_loc] - b[old_i_loc, old_i_loc])
                    delta += a[j, j] * (b[old_i_loc, old_i_loc] - b[old_j_loc, old_j_loc])
                    delta += (a[i, j] - a[j, i]) * (b[old_j_loc, old_i_loc] - b[old_i_loc, old_j_loc])

                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j

            if best_i >= 0:
                p_arr[best_i], p_arr[best_j] = p_arr[best_j], p_arr[best_i]
                cur_cost += best_delta
                improved = True

        return p_arr, cur_cost

    # --- Main algorithm ---
    # Phase 1: Greedy construction
    p = greedy_construct()
    cur_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = cur_cost

    # Phase 2: Local search from greedy solution
    p, cur_cost = local_search(p, cur_cost, deadline - 20)
    if cur_cost < best_cost:
        best_cost = cur_cost
        best_p = p.copy()

    # Phase 3: Multiple restarts with random perturbations
    if time.time() < deadline - 15:
        # Try several random restarts
        for restart in range(20):
            if time.time() > deadline - 10:
                break

            # Start from best solution and apply random perturbations
            p = best_p.copy()
            # Apply 5-10 random swaps
            n_swaps = 5 + restart // 4
            for _ in range(n_swaps):
                i, j = np.random.randint(0, n, 2)
                if i != j:
                    p[i], p[j] = p[j], p[i]

            cur_cost = compute_cost(p)

            # Local search
            p, cur_cost = local_search(p, cur_cost, deadline - 10)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_p = p.copy()

    # Phase 4: If time remains, do a final polish
    if time.time() < deadline - 5:
        best_p, best_cost = local_search(best_p.copy(), best_cost, deadline - 2)

    return best_p.tolist()
