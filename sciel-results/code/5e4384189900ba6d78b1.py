"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def compute_autocorr(h):
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        R = np.zeros(N)
        R[0] = acf[0]
        R[1:N] = acf[1:N]
        return R

    def get_max_autocorr(R):
        return np.max(np.abs(R[1:N]))

    def hill_climb(h, R, current_val, max_passes=300, time_limit=None):
        """Greedy local search with incremental O(N) updates."""
        if time_limit is None:
            time_limit = deadline - 5.0
        h = h.copy()
        R = R.copy()
        k = np.arange(1, N)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > time_limit:
                break
            improved = False
            passes += 1
            order = np.random.permutation(N)
            for i in order:
                if time.time() > time_limit:
                    break
                # Delta for flipping h[i]
                idx1 = i + k
                valid1 = idx1 < N
                term1 = np.zeros(N-1)
                term1[valid1] = h[idx1[valid1]]
                idx2 = i - k
                valid2 = idx2 >= 0
                term2 = np.zeros(N-1)
                term2[valid2] = h[idx2[valid2]]
                delta = -2.0 * h[i] * (term1 + term2)
                R[1:N] += delta
                new_val = get_max_autocorr(R)
                if new_val < current_val:
                    h[i] *= -1.0
                    current_val = new_val
                    improved = True
                else:
                    R[1:N] -= delta
        return h, R, current_val

    def anneal_segment(h, R, seg_start, seg_len, iters=3000, T0=5.0, T_min=0.05):
        """Simulated annealing restricted to a segment [seg_start, seg_start+seg_len)."""
        h = h.copy()
        R = R.copy()
        k = np.arange(1, N)
        T = T0
        cooling = (T_min / T0) ** (1.0 / max(iters, 1))
        current_val = get_max_autocorr(R)
        indices = np.arange(seg_start, seg_start + seg_len)
        for it in range(iters):
            if time.time() > deadline - 15.0:
                break
            i = indices[np.random.randint(seg_len)]
            idx1 = i + k
            valid1 = idx1 < N
            term1 = np.zeros(N-1)
            term1[valid1] = h[idx1[valid1]]
            idx2 = i - k
            valid2 = idx2 >= 0
            term2 = np.zeros(N-1)
            term2[valid2] = h[idx2[valid2]]
            delta = -2.0 * h[i] * (term1 + term2)
            R[1:N] += delta
            new_val = get_max_autocorr(R)
            delta_val = new_val - current_val
            if delta_val < 0:
                h[i] *= -1.0
                current_val = new_val
            else:
                if np.random.rand() < np.exp(-delta_val / T):
                    h[i] *= -1.0
                    current_val = new_val
                else:
                    R[1:N] -= delta
            T *= cooling
            if T < T_min:
                T = T_min
        return h, R, current_val

    def block_exchange(h, R, current_val, time_limit):
        """
        Structural change: treat the sequence as N/10 blocks of length 10.
        Build a cost matrix for permuting blocks, solve the assignment problem,
        apply the best permutation, then hill-climb to polish.
        """
        n_blocks = N // 10
        block_len = 10
        if n_blocks < 2:
            return h, R, current_val

        block_starts = np.arange(n_blocks) * block_len
        block_ends = block_starts + block_len

        # Cost matrix: cost[i][j] = how much the max sidelobe increases if block i goes to position j
        # Approximate: compute the autocorrelation contribution of each block to each lag
        # For efficiency, we'll use a simpler heuristic:
        # cost[i][j] = -sum of cross-correlation between block i and the target neighborhood
        # Actually, let's compute the actual change in objective for each swap.
        # Since N=400, n_blocks=40, the cost matrix is 40x40 = 1600 entries.
        # Each entry requires O(N) work to compute, so total O(N * n_blocks^2) = O(400 * 1600) = 640k ops.

        cost = np.zeros((n_blocks, n_blocks))
        base_h = h.copy()
        base_R = R.copy()
        base_val = current_val

        for i in range(n_blocks):
            for j in range(n_blocks):
                if i == j:
                    cost[i][j] = 0.0
                    continue
                # Simulate swapping blocks i and j
                h_trial = base_h.copy()
                # Extract blocks
                bi = h_trial[block_starts[i]:block_ends[i]]
                bj = h_trial[block_starts[j]:block_ends[j]]
                h_trial[block_starts[i]:block_ends[i]] = bj
                h_trial[block_starts[j]:block_ends[j]] = bi
                # Compute autocorrelation
                R_trial = compute_autocorr(h_trial)
                val_trial = get_max_autocorr(R_trial)
                cost[i][j] = val_trial - base_val

        # Solve the assignment problem: find permutation pi that minimizes sum of costs
        row_ind, col_ind = linear_sum_assignment(cost)
        # Apply the permutation
        h_new = h.copy()
        # The assignment tells us: block at row_ind[r] should go to position col_ind[r]
        # We need to construct the new order
        perm = np.zeros(n_blocks, dtype=int)
        perm[col_ind] = row_ind
        for new_pos in range(n_blocks):
            old_pos = perm[new_pos]
            h_new[block_starts[new_pos]:block_ends[new_pos]] = h[block_starts[old_pos]:block_ends[old_pos]]

        R_new = compute_autocorr(h_new)
        val_new = get_max_autocorr(R_new)

        # Polish with hill climbing
        h_new, R_new, val_new = hill_climb(h_new, R_new, val_new, max_passes=100, time_limit=time_limit)
        return h_new, R_new, val_new

    def multi_block_anneal(h, R, current_val, time_limit):
        """
        Structural change: divide sequence into 4 quarters.
        For each quarter, run segment-restricted annealing, then hill-climb.
        """
        q_len = N // 4
        h_new = h.copy()
        R_new = R.copy()
        val_new = current_val
        for q in range(4):
            if time.time() > time_limit:
                break
            seg_start = q * q_len
            seg_len = q_len if q < 3 else N - seg_start
            h_new, R_new, val_new = anneal_segment(h_new, R_new, seg_start, seg_len, iters=1500, T0=3.0)
        # Final polish
        h_new, R_new, val_new = hill_climb(h_new, R_new, val_new, max_passes=200, time_limit=time_limit)
        return h_new, R_new, val_new

    rng = np.random.RandomState(123)
    best_h = None
    best_R = None
    best_val = float('inf')

    # Phase 1: Random restarts with hill climbing
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        h = rng.choice([1, -1], N).astype(np.float64)
        R = compute_autocorr(h)
        val = get_max_autocorr(R)
        h, R, val = hill_climb(h, R, val, max_passes=100, time_limit=deadline - 20.0)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            best_R = R.copy()

    if best_h is None:
        best_h = rng.choice([1, -1], N).astype(np.float64)
        best_R = compute_autocorr(best_h)
        best_val = get_max_autocorr(best_R)

    # Phase 2: Block exchange structural moves
    if time.time() < deadline - 30.0:
        h2, R2, val2 = block_exchange(best_h, best_R, best_val, deadline - 10.0)
        if val2 < best_val:
            best_val = val2
            best_h = h2.copy()
            best_R = R2.copy()

    # Phase 3: Multi-block annealing
    if time.time() < deadline - 20.0:
        h3, R3, val3 = multi_block_anneal(best_h, best_R, best_val, deadline - 10.0)
        if val3 < best_val:
            best_val = val3
            best_h = h3.copy()
            best_R = R3.copy()

    # Phase 4: Another block exchange
    if time.time() < deadline - 15.0:
        h4, R4, val4 = block_exchange(best_h, best_R, best_val, deadline - 5.0)
        if val4 < best_val:
            best_val = val4
            best_h = h4.copy()
            best_R = R4.copy()

    # Final polish
    if time.time() < deadline - 5.0:
        h5, R5, val5 = hill_climb(best_h, best_R, best_val, max_passes=200, time_limit=deadline - 2.0)
        if val5 < best_val:
            best_val = val5
            best_h = h5.copy()

    return best_h
# EVOLVE-BLOCK-END
