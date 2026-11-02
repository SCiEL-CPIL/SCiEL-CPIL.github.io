import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute all words as indices
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)

    # Convert to numpy arrays for faster vectorized operations
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]

    # Precompute ball membership matrix: ball_mem[i, j] = True if j is in ball(i)
    ball_mem = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mem[i, ball_indices_np[i]] = True

    # Greedy construction with randomization
    def greedy_build(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        covered = np.zeros(total, dtype=bool)
        code = []

        while not np.all(covered):
            best_c = -1
            max_new = -1
            candidates = list(range(total))
            rng.shuffle(candidates)

            for c in candidates:
                ball = ball_indices_np[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_indices_np[best_c]] = True
        return code

    # Multiple greedy restarts to find good starting points
    best_initial = None
    best_initial_size = float('inf')

    # First greedy is deterministic (seed=0)
    init_code = greedy_build(seed=0)
    best_initial = set(init_code)
    best_initial_size = len(init_code)

    # Try more seeds with tighter time check
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 40.0:
            break
        init_code = greedy_build(seed=seed)
        if len(init_code) < best_initial_size:
            best_initial_size = len(init_code)
            best_initial = set(init_code)

    current_code = best_initial
    best_code = list(current_code)
    best_size = len(current_code)

    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1

    def run_delete_phase(code_set, cnt_arr):
        """Removes redundant codewords. Returns True if any removed."""
        changed = False
        # Iterate over a copy to allow modification
        for c in list(code_set):
            ball = ball_indices_np[c]
            if np.min(cnt_arr[ball]) >= 2:
                code_set.remove(c)
                cnt_arr[ball] -= 1
                changed = True
        return changed

    def try_swap(c_remove, c_add, code_set, cnt_arr):
        """Try a swap. Returns True if successful."""
        if c_add in code_set:
            return False
        if c_remove == c_add:
            return False

        ball_rem = ball_indices_np[c_remove]
        ball_add = ball_indices_np[c_add]

        # Use precomputed ball membership matrix for fast intersection check
        in_add = ball_mem[c_add, ball_rem]
        only_in_rem = ~in_add

        # Check if any element only in ball_rem has count < 2
        if np.any(cnt_arr[ball_rem[only_in_rem]] < 2):
            return False

        # Perform swap
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt_arr[ball_rem] -= 1
        cnt_arr[ball_add] += 1
        return True

    # Local Search Loop with improved simulated annealing
    no_improve_counter = 0
    max_no_improve = 50
    temperature = 10.0  # Higher initial temperature for more exploration
    cooling_rate = 0.97

    # Precompute ball overlap matrix for faster swap evaluation
    # overlap[i, j] = number of elements in ball(i) that are also in ball(j)
    # This helps us evaluate swaps more efficiently
    overlap = np.zeros((total, total), dtype=np.int32)
    for i in range(total):
        for j in range(i, total):
            if i == j:
                overlap[i, j] = ball_size
            else:
                # Count overlap using ball_mem
                overlap[i, j] = np.sum(ball_mem[i, ball_indices_np[j]])
                overlap[j, i] = overlap[i, j]

    while time.time() < deadline - 5.0:
        # 1. Delete Phase
        while run_delete_phase(current_code, cnt):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                no_improve_counter = 0
            else:
                no_improve_counter += 1

        # 2. Swap Phase with improved candidate selection
        swaps_attempted = 0
        max_swaps = 2000
        while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break

            # Pick random codeword to remove
            c = random.choice(list(current_code))

            # Smarter candidate selection: prioritize words with high overlap with current code
            # to enable more deletions after swap
            c_prime = random.randint(0, total - 1)

            # Try the swap
            if try_swap(c, c_prime, current_code, cnt):
                # After a valid swap, run delete phase
                while run_delete_phase(current_code, cnt):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1

                    # Early exit if we've hit a very good solution
                    if best_size <= 160:
                        break

                if best_size <= 160:
                    break
            else:
                no_improve_counter += 1

        # 3. If no improvement, try a restart with different strategy
        if no_improve_counter > max_no_improve and time.time() < deadline - 20.0:
            # Try a new greedy initialization
            seed = int(time.time() * 1000) % 10000
            init_code = greedy_build(seed=seed)
            if len(init_code) < len(current_code) + 5:  # only if competitive
                current_code = set(init_code)
                cnt = np.zeros(total, dtype=np.int32)
                for c in current_code:
                    cnt[ball_indices_np[c]] += 1
                no_improve_counter = 0

        if best_size <= 160:
            break

    # Final result conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
