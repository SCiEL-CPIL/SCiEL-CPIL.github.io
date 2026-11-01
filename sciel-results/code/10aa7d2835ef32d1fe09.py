"""Ternary covering code via vectorized simulated annealing."""
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 143.0  # 7s safety margin

    N = q ** n
    if N <= 0:
        return []

    # --- Precompute: all words as digits ---
    all_words = np.zeros((N, n), dtype=np.int32)
    for idx in range(N):
        rem = idx
        for j in range(n - 1, -1, -1):
            all_words[idx, j] = rem % q
            rem //= q

    # --- Precompute: ball matrix (N x N boolean, vectorized) ---
    # ball_matrix[c, w] = True iff w is within Hamming distance `radius` of c
    # For radius 1: ball(c) = {c} ∪ {neighbors}
    if radius == 1:
        # Vectorized: for each position, compute neighbors
        ball_matrix = np.zeros((N, N), dtype=bool)
        # Self
        np.fill_diagonal(ball_matrix, True)
        # Neighbors: change one position to another value
        for pos in range(n):
            power = q ** (n - 1 - pos)
            for dv in range(-q + 1, q):
                if dv == 0:
                    continue
                # For each c, new_idx = c + dv * power (if in range)
                # We need to check the digit at pos changes correctly
                c_vals = np.arange(N)
                digit_at_pos = (c_vals // power) % q
                new_digit = digit_at_pos + dv
                valid = (new_digit >= 0) & (new_digit < q)
                c_valid = c_vals[valid]
                new_indices = c_valid + dv * power
                # Verify new_indices are in range
                valid2 = (new_indices >= 0) & (new_indices < N)
                c_valid = c_valid[valid2]
                new_indices = new_indices[valid2]
                ball_matrix[c_valid, new_indices] = True
    else:
        # General radius: compute Hamming distance matrix
        # O(N^2 * n) but vectorized
        ball_matrix = np.zeros((N, N), dtype=bool)
        for pos in range(n):
            power = q ** (n - 1 - pos)
            digits = (np.arange(N) // power) % q
            # For each pair, count mismatches
            # This is O(N^2) per position - too slow for large N
            # Fallback: loop
            for c in range(N):
                for w in range(N):
                    if np.sum(all_words[c] != all_words[w]) <= radius:
                        ball_matrix[c, w] = True
                        break

    # Ball sizes
    ball_sizes = ball_matrix.sum(axis=1)

    # --- Initial solution: random over-cover ---
    # Start with a set of codewords that covers everything
    # Use a simple greedy to get a valid starting point, then perturb
    uncovered = np.ones(N, dtype=bool)
    initial_code = []
    while uncovered.any():
        # Find c that covers most uncovered
        gains = ball_matrix[:, uncovered].sum(axis=1)
        c = int(np.argmax(gains))
        initial_code.append(c)
        uncovered[ball_matrix[c]] = False

    code_arr = np.array(initial_code, dtype=np.int32)

    # --- Coverage count vector (vectorized) ---
    # cov_count[w] = number of codewords in code_arr that cover w
    cov_count = np.zeros(N, dtype=np.int32)
    for c in code_arr:
        cov_count += ball_matrix[c].astype(np.int32)

    # --- SA Objective ---
    # We want to minimize |code| while keeping all words covered.
    # Objective = |code| + BIG * (number of uncovered words)
    # BIG should be large enough that coverage is always prioritized
    BIG = 10000

    def objective(code_arr, cov_count):
        """Return (penalty, size) where penalty = uncovered_count * BIG + size"""
        uncovered_count = int(np.sum(cov_count == 0))
        return uncovered_count * BIG + len(code_arr)

    best_code = code_arr.copy()
    best_cov_count = cov_count.copy()
    best_obj = objective(best_code, best_cov_count)
    best_size = len(best_code)

    # --- SA Parameters ---
    T_start = 10.0
    T_end = 0.01
    # Number of iterations: we'll run until deadline
    # Each iteration: one move (add, remove, or swap)

    # Precompute: for each codeword c, which words does it uniquely cover?
    # Actually, we'll compute on the fly using cov_count

    rng = np.random.default_rng(42)
    random.seed(42)

    iter_count = 0
    stall_count = 0
    best_stall_obj = best_obj

    while time.time() < deadline - 5.0:
        # Annealing schedule: exponential cooling
        # We don't know total iterations, so use a time-based schedule
        elapsed = time.time() - start_time
        total_budget = deadline - start_time
        frac = elapsed / total_budget
        T = T_start * (T_end / T_start) ** frac

        # Choose move type
        move_type = rng.choice(3, p=[0.25, 0.35, 0.40])  # add, remove, swap

        if move_type == 0:  # ADD a random codeword
            if len(code_arr) >= N:
                continue
            new_c = int(rng.integers(0, N))
            if new_c in code_arr:
                continue
            # Evaluate: new cov_count = cov_count + ball_matrix[new_c]
            new_cov = cov_count + ball_matrix[new_c].astype(np.int32)
            new_obj = objective(code_arr, new_cov) + 1  # +1 for the added word
            delta_obj = new_obj - objective(code_arr, cov_count)
            # Accept if improves or with probability exp(-delta/T)
            if delta_obj <= 0 or rng.random() < np.exp(-delta_obj / max(T, 1e-10)):
                code_arr = np.append(code_arr, new_c)
                cov_count = new_cov
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_code = code_arr.copy()
                    best_cov_count = cov_count.copy()
                    best_size = len(best_code)
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                stall_count += 1

        elif move_type == 1:  # REMOVE a random codeword
            if len(code_arr) < 2:
                continue
            i = int(rng.integers(0, len(code_arr)))
            c = int(code_arr[i])
            # Check: can we remove c?
            # Words that c covers and no other codeword covers:
            # unique_to_c = ball_matrix[c] & (cov_count == 1)
            # If any word is uniquely covered by c, removal is invalid (penalty)
            unique_mask = ball_matrix[c] & (cov_count == 1)
            unique_count = int(np.sum(unique_mask))
            if unique_count == 0:
                # Safe removal
                new_cov = cov_count - ball_matrix[c].astype(np.int32)
                new_code = np.delete(code_arr, i)
                new_obj = objective(new_code, new_cov)
                delta_obj = new_obj - objective(code_arr, cov_count)
                if delta_obj <= 0 or rng.random() < np.exp(-delta_obj / max(T, 1e-10)):
                    code_arr = new_code
                    cov_count = new_cov
                    cur_obj = new_obj
                    if cur_obj < best_obj:
                        best_obj = cur_obj
                        best_code = code_arr.copy()
                        best_cov_count = cov_count.copy()
                        best_size = len(best_code)
                        stall_count = 0
                    else:
                        stall_count += 1
                else:
                    stall_count += 1
            else:
                # Removal would uncover `unique_count` words
                # Penalty increases by unique_count * BIG, size decreases by 1
                delta_obj = unique_count * BIG - 1
                if delta_obj <= 0 or rng.random() < np.exp(-delta_obj / max(T, 1e-10)):
                    new_cov = cov_count - ball_matrix[c].astype(np.int32)
                    new_code = np.delete(code_arr, i)
                    code_arr = new_code
                    cov_count = new_cov
                    cur_obj = objective(code_arr, cov_count)
                    if cur_obj < best_obj:
                        best_obj = cur_obj
                        best_code = code_arr.copy()
                        best_cov_count = cov_count.copy()
                        best_size = len(best_code)
                        stall_count = 0
                    else:
                        stall_count += 1
                else:
                    stall_count += 1

        else:  # SWAP: replace code_arr[i] with a random new_c
            if len(code_arr) < 2:
                continue
            i = int(rng.integers(0, len(code_arr)))
            old_c = int(code_arr[i])
            new_c = int(rng.integers(0, N))
            if new_c == old_c or new_c in code_arr:
                continue
            # Evaluate swap
            # New cov_count = cov_count - ball_matrix[old_c] + ball_matrix[new_c]
            new_cov = cov_count - ball_matrix[old_c].astype(np.int32) + ball_matrix[new_c].astype(np.int32)
            new_code = code_arr.copy()
            new_code[i] = new_c
            new_obj = objective(new_code, new_cov)
            cur_obj = objective(code_arr, cov_count)
            delta_obj = new_obj - cur_obj
            if delta_obj <= 0 or rng.random() < np.exp(-delta_obj / max(T, 1e-10)):
                code_arr = new_code
                cov_count = new_cov
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_code = code_arr.copy()
                    best_cov_count = cov_count.copy()
                    best_size = len(best_code)
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                stall_count += 1

        iter_count += 1

        # Periodic: if we've been stuck, try a targeted removal pass
        if stall_count > 500 and time.time() < deadline - 10:
            # Try to remove any redundant codeword
            for i in range(len(code_arr) - 1, -1, -1):
                c = int(code_arr[i])
                unique_mask = ball_matrix[c] & (cov_count == 1)
                if not unique_mask.any():
                    # Remove it
                    new_cov = cov_count - ball_matrix[c].astype(np.int32)
                    new_code = np.delete(code_arr, i)
                    code_arr = new_code
                    cov_count = new_cov
                    cur_obj = objective(code_arr, cov_count)
                    if cur_obj < best_obj:
                        best_obj = cur_obj
                        best_code = code_arr.copy()
                        best_cov_count = cov_count.copy()
                        best_size = len(best_code)
                    stall_count = 0
                    break

    # --- Final: ensure validity ---
    # best_code should be valid, but verify
    final_cov = np.zeros(N, dtype=np.int32)
    for c in best_code:
        final_cov += ball_matrix[c].astype(np.int32)

    if np.any(final_cov == 0):
        # Repair: add codewords to cover missing words
        missing = np.where(final_cov == 0)[0]
        for w in missing:
            # Find a codeword that covers w
            candidates = np.where(ball_matrix[:, w])[0]
            if len(candidates) > 0:
                c = int(candidates[0])
                if c not in best_code:
                    best_code = np.append(best_code, c)
                    final_cov += ball_matrix[c].astype(np.int32)

    return [list(all_words[i]) for i in best_code]
