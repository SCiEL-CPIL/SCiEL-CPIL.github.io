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

    # ---- Vectorized cover matrix (inspiration 1) ----
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

    def two_opt_polish(code):
        if len(code) < 2:
            return code
        current_code = code[:]
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            cover_count += cover_matrix[idx]
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
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
                for j in range(n_cw):
                    if j == i:
                        continue
                    b = current_code[j]
                    mask_b = cover_matrix[b]
                    # a and b must jointly cover unique_a without overlap issues:
                    # after removing a, unique_a words lose coverage unless b covers them
                    if np.any(unique_a & ~mask_b):
                        continue
                    # check that removing a and b and re-adding a,b (swap) is a no-op
                    # For a 2-opt swap we replace a with b and b with a? That's identity.
                    # Instead: try removing a and adding a new codeword c that covers unique_a.
                    # (handled by SA). Keep 2-opt as pure removal-then-refill check:
                    break
        return current_code

    def simulated_annealing(code):
        if not code:
            return code
        current_code = code[:]
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            cover_count += cover_matrix[idx]

        def energy(c_code):
            # minimal valid code: minimize size; validity guaranteed by moves
            return len(c_code)

        best_code = current_code[:]
        best_energy = energy(current_code)
        current_energy = best_energy

        T_start = 8.0
        T_end = 0.05
        T = T_start
        while T > T_end:
            if time.time() > deadline - 2.0:
                break
            move_type = random.random()
            if move_type < 0.35:
                # add a random codeword (may reduce nothing but can enable removals)
                new_idx = random.randint(0, total_space - 1)
                if new_idx in current_code:
                    T *= 0.995
                    continue
                new_code = current_code + [new_idx]
                delta_e = energy(new_code) - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    cover_count += cover_matrix[new_idx]
                    current_code = new_code
                    current_energy = energy(current_code)
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
                    current_energy = energy(current_code)
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
                # words that only old covers
                unique_old = mask_old & (cover_count == 1)
                if np.any(unique_old & ~cover_matrix[new_idx]):
                    T *= 0.995
                    continue
                # swap keeps size the same; accept with small probability to explore
                delta_e = 0
                if random.random() < np.exp(-delta_e / T):
                    cover_count -= mask_old
                    cover_count += cover_matrix[new_idx]
                    current_code[pos] = new_idx
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
            T *= 0.995

        return best_code

    # Initial greedy + reduce
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)

    # Restarts if time permits
    while time.time() < deadline - 15.0:
        cand = greedy_cover_randomized()
        cand = local_search_reduce(cand)
        cand = simulated_annealing(cand)
        cand = local_search_reduce(cand)
        if len(cand) < len(best_code):
            best_code = cand

    # Final high-precision polish
    best_code = local_search_reduce(best_code)
    if time.time() < deadline - 3.0:
        best_code = simulated_annealing(best_code)
    best_code = local_search_reduce(best_code)

    return [all_words[i] for i in best_code]
