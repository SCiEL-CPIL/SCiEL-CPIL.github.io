import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    total_space = q ** n

    # Precompute index -> word
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    all_words = [idx_to_word(i) for i in range(total_space)]

    # ---- Vectorized cover matrix ----
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    np.fill_diagonal(cover_matrix, True)

    if radius >= 1:
        powers = [q ** (n - 1 - pos) for pos in range(n)]
        for i in range(total_space):
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                power = powers[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    j = i + (val - orig_val) * power
                    cover_matrix[i, j] = True

    cm_int = cover_matrix.astype(np.int8)  # for fast dot products

    def greedy_cover_randomized():
        uncovered = np.ones(total_space, dtype=bool)
        code = []
        unc_int = uncovered.astype(np.int8)
        while np.any(uncovered):
            if time.time() > deadline - 8.0:
                break
            gains = cm_int.dot(unc_int)
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            code.append(best_idx)
            uncovered[cover_matrix[best_idx]] = False
            unc_int = uncovered.astype(np.int8)
        return code

    def local_search_reduce(code):
        if not code:
            return code
        current_code = code[:]
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            cover_count += cover_matrix[idx]

        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 4.0:
                break
            random.shuffle(current_code)
            n_cw = len(current_code)
            for i in range(n_cw):
                idx = current_code[i]
                mask = cover_matrix[idx]
                # removable if no word is covered only by this codeword
                if not np.any(cover_count[mask] == 1):
                    cover_count -= mask
                    current_code.pop(i)
                    improved = True
                    break
        return current_code

    def swap_polish(code):
        """Try to replace one codeword with another to enable removal of a second,
        reducing total size by 1 if possible."""
        if len(code) < 2:
            return code
        current_code = code[:]
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            cover_count += cover_matrix[idx]

        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 4.0:
                break
            random.shuffle(current_code)
            n_cw = len(current_code)
            for i in range(n_cw):
                a = current_code[i]
                mask_a = cover_matrix[a]
                # words uniquely covered by a
                unique_a = mask_a & (cover_count == 1)
                if not np.any(unique_a):
                    continue
                # Try to find a replacement codeword c (not in code) that covers unique_a
                # and after swapping a->c, some other codeword b becomes removable
                # We search for c such that:
                # 1. c covers all of unique_a
                # 2. After swap, there exists b != i where b's unique coverage becomes empty
                # For efficiency, just check if any c in the full space covers unique_a
                # and then check if removal of any b is possible.
                # This is expensive; limit to a small random sample.
                found = False
                for _ in range(50):
                    c = random.randint(0, total_space - 1)
                    if c in current_code:
                        continue
                    mask_c = cover_matrix[c]
                    if not np.all(mask_c[unique_a]):
                        continue
                    # Simulate swap: remove a, add c
                    new_cover_count = cover_count - mask_a + mask_c
                    # Check if any codeword b (b != i, b != c) is now removable
                    for j in range(n_cw):
                        if j == i:
                            continue
                        b = current_code[j]
                        mask_b = cover_matrix[b]
                        if not np.any(new_cover_count[mask_b] == 1):
                            # Perform the swap and removal
                            cover_count = new_cover_count
                            current_code[i] = c
                            current_code.pop(j)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        return current_code

    def simulated_annealing(code):
        if not code:
            return code
        current_code = code[:]
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            cover_count += cover_matrix[idx]

        best_code = current_code[:]
        best_energy = len(current_code)
        current_energy = best_energy

        T_start = 8.0
        T_end = 0.05
        T = T_start
        while T > T_end:
            if time.time() > deadline - 2.0:
                break
            move_type = random.random()
            if move_type < 0.35:
                # add a random codeword
                new_idx = random.randint(0, total_space - 1)
                if new_idx in current_code:
                    T *= 0.995
                    continue
                new_code = current_code + [new_idx]
                delta_e = 1
                if random.random() < np.exp(-delta_e / T):
                    cover_count += cover_matrix[new_idx]
                    current_code = new_code
                    current_energy = len(current_code)
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
            elif move_type < 0.7:
                # remove a codeword (only if it does not break coverage)
                if len(current_code) <= 1:
                    T *= 0.995
                    continue
                pos = random.randint(0, len(current_code) - 1)
                idx = current_code[pos]
                mask = cover_matrix[idx]
                if not np.any(cover_count[mask] == 1):
                    cover_count -= mask
                    new_code = current_code[:pos] + current_code[pos + 1:]
                    current_code = new_code
                    current_energy = len(current_code)
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
            else:
                # swap: replace one codeword with another while keeping coverage
                if not current_code:
                    T *= 0.995
                    continue
                pos = random.randint(0, len(current_code) - 1)
                old_idx = current_code[pos]
                new_idx = random.randint(0, total_space - 1)
                if new_idx == old_idx or new_idx in current_code:
                    T *= 0.995
                    continue
                mask_old = cover_matrix[old_idx]
                unique_old = mask_old & (cover_count == 1)
                if np.any(unique_old & ~cover_matrix[new_idx]):
                    T *= 0.995
                    continue
                cover_count -= mask_old
                cover_count += cover_matrix[new_idx]
                current_code[pos] = new_idx
            T *= 0.995

        return best_code

    # Initial greedy + reduce
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    best_code = swap_polish(best_code)

    # Restarts if time permits
    while time.time() < deadline - 15.0:
        cand = greedy_cover_randomized()
        cand = local_search_reduce(cand)
        cand = simulated_annealing(cand)
        cand = local_search_reduce(cand)
        cand = swap_polish(cand)
        if len(cand) < len(best_code):
            best_code = cand

    # Final high-precision polish
    best_code = local_search_reduce(best_code)
    if time.time() < deadline - 3.0:
        best_code = simulated_annealing(best_code)
    best_code = local_search_reduce(best_code)
    best_code = swap_polish(best_code)

    return [all_words[i] for i in best_code]
