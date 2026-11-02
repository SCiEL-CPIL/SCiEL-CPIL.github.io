"""Combined program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Specific optimization for the known task: q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)

    # Fallback for other parameters: simple greedy + pruning
    return _construct_general(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    N = 2187  # 3^7
    # Precompute powers of 3 for fast index conversion
    p3 = [1] * 7
    for i in range(1, 7):
        p3[i] = p3[i - 1] * 3

    # Precompute digits for each index
    digits_table = [[0] * 7 for _ in range(N)]
    for i in range(N):
        tmp = i
        for pos in range(7):
            digits_table[i][pos] = tmp % 3
            tmp //= 3

    # Precompute ball_masks: for each word index, bitmask of indices within Hamming distance 1
    ball_masks = [0] * N
    for i in range(N):
        mask = 1 << i
        for pos in range(7):
            orig = digits_table[i][pos]
            for val in range(3):
                if val != orig:
                    new_i = i + (val - orig) * p3[pos]
                    mask |= (1 << new_i)
        ball_masks[i] = mask

    # Precompute reverse_ball_masks: for each word w, bitmask of centers c' such that w is in ball(c')
    # This is the same as ball_masks because the relation is symmetric (dist <= 1 is symmetric)
    # So reverse_ball_masks[w] = ball_masks[w]
    # But let's verify: word w is covered by center c iff dist(c, w) <= 1 iff dist(w, c) <= 1
    # So yes, reverse_ball_masks = ball_masks

    # Helper: Greedy Set Cover initialization using bitset operations
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = (1 << N) - 1  # Full mask
        selected = []
        selected_set = set()

        while uncovered:
            best_idx = -1
            best_count = -1

            # Randomized scan to break ties and explore different paths
            candidates = list(range(N))
            rng.shuffle(candidates)

            for i in candidates:
                if i in selected_set:
                    continue

                # Count uncovered bits covered by i
                cnt = (uncovered & ball_masks[i]).bit_count()

                if cnt > best_count:
                    best_count = cnt
                    best_idx = i

            if best_idx == -1 or best_count == 0:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~ball_masks[best_idx]

            if time.time() > deadline - 5.0:
                break

        return selected

    # Helper: Local Search with Perturbation and Swaps (bitset-based)
    def local_search(code, seed):
        rng = random.Random(seed)
        code_set = set(code)

        # cnt[i] is the number of codewords in 'code' that cover word i
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            mask = ball_masks[c]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] += 1
                m ^= lsb

        best_code = code.copy()
        best_size = len(code)
        no_improve_count = 0
        max_no_improve = 15  # Trigger perturbation after this many failed cycles

        while time.time() < deadline - 2.0:
            improved = False

            # Phase 1: Deletion - remove codewords that don't uniquely cover any word
            to_remove = []
            for c in code:
                mask = ball_masks[c]
                m = mask
                is_essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cnt[idx] == 1:
                        is_essential = True
                        break
                    m ^= lsb
                if not is_essential:
                    to_remove.append(c)

            if to_remove:
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    mask = ball_masks[c]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cnt[idx] -= 1
                        m ^= lsb

                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
                improved = True
                no_improve_count = 0
                continue  # Restart loop as counts changed significantly

            # Phase 2: 1-Swap - replace one codeword with another
            if code:
                swaps_done = 0
                max_swaps = 80
                while swaps_done < max_swaps and time.time() < deadline - 2.0:
                    swaps_done += 1
                    c = rng.choice(code)

                    # Find words uniquely covered by c
                    unique_words = []
                    mask = ball_masks[c]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cnt[idx] == 1:
                            unique_words.append(idx)
                        m ^= lsb

                    if not unique_words:
                        # Redundant, handle just in case
                        code.remove(c)
                        code_set.remove(c)
                        mask = ball_masks[c]
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] -= 1
                            m ^= lsb
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code.copy()
                        improved = True
                        break

                    # Find candidates c' that cover all unique_words
                    # A candidate c' must be within distance 1 of all unique_words
                    # i.e., c' must be in the intersection of balls centered at each unique_word
                    cand_mask = ball_masks[unique_words[0]]
                    for w in unique_words[1:]:
                        cand_mask &= ball_masks[w]
                        if not cand_mask:
                            break

                    if not cand_mask:
                        continue

                    # Filter out candidates already in code
                    candidates = []
                    m = cand_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if idx not in code_set:
                            candidates.append(idx)
                        m ^= lsb

                    if not candidates:
                        continue

                    # Pick a random candidate
                    c_prime = rng.choice(candidates)

                    # Perform swap
                    code.remove(c)
                    code_set.remove(c)
                    mask = ball_masks[c]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cnt[idx] -= 1
                        m ^= lsb

                    code.append(c_prime)
                    code_set.add(c_prime)
                    mask = ball_masks[c_prime]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cnt[idx] += 1
                        m ^= lsb

                    improved = True

            # Phase 3: 2-for-1 Swap (Merge two codewords into one)
            if len(code) > 2:
                pair_swap_found = False
                num_pairs_to_check = 300
                for _ in range(num_pairs_to_check):
                    if time.time() > deadline - 2.0:
                        break
                    c1, c2 = rng.sample(code, 2)

                    # Collect unique words covered by c1 or c2
                    unique_words_set = set()
                    mask1 = ball_masks[c1]
                    m = mask1
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cnt[idx] == 1:
                            unique_words_set.add(idx)
                        m ^= lsb

                    mask2 = ball_masks[c2]
                    m = mask2
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cnt[idx] == 1:
                            unique_words_set.add(idx)
                        m ^= lsb

                    if not unique_words_set:
                        continue

                    unique_words = list(unique_words_set)

                    # Find candidates c' that cover all unique_words
                    cand_mask = ball_masks[unique_words[0]]
                    for w in unique_words[1:]:
                        cand_mask &= ball_masks[w]
                        if not cand_mask:
                            break

                    if not cand_mask:
                        continue

                    # Check if there is a candidate not in code
                    candidates = []
                    m = cand_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if idx not in code_set:
                            candidates.append(idx)
                        m ^= lsb

                    if candidates:
                        c_new = rng.choice(candidates)

                        # Perform 2-for-1 swap
                        code.remove(c1)
                        code_set.remove(c1)
                        mask = ball_masks[c1]
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] -= 1
                            m ^= lsb

                        code.remove(c2)
                        code_set.remove(c2)
                        mask = ball_masks[c2]
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] -= 1
                            m ^= lsb

                        code.append(c_new)
                        code_set.add(c_new)
                        mask = ball_masks[c_new]
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] += 1
                            m ^= lsb

                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code.copy()
                        improved = True
                        pair_swap_found = True
                        break  # Restart main loop to re-evaluate deletions

                if pair_swap_found:
                    no_improve_count = 0
                    continue

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Phase 4: Perturbation (Iterated Local Search)
            if no_improve_count >= max_no_improve:
                # Perturb: Remove 3-5 random codewords
                num_remove = rng.randint(3, 5)
                if len(code) > num_remove:
                    to_perturb = rng.sample(code, num_remove)
                    for c in to_perturb:
                        code.remove(c)
                        code_set.remove(c)
                        mask = ball_masks[c]
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] -= 1
                            m ^= lsb

                    # Repair: Greedily add codewords to cover any uncovered words
                    uncovered_mask = 0
                    for i in range(N):
                        if cnt[i] == 0:
                            uncovered_mask |= (1 << i)

                    # Greedy repair
                    while uncovered_mask and time.time() < deadline - 2.0:
                        best_idx = -1
                        best_count = -1
                        candidates = list(range(N))
                        rng.shuffle(candidates)
                        for i in candidates:
                            if i in code_set:
                                continue
                            cnt_cover = (uncovered_mask & ball_masks[i]).bit_count()
                            if cnt_cover > best_count:
                                best_count = cnt_cover
                                best_idx = i

                        if best_idx == -1 or best_count == 0:
                            break

                        code.append(best_idx)
                        code_set.add(best_idx)
                        mask = ball_masks[best_idx]
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] += 1
                            m ^= lsb
                        uncovered_mask &= ~ball_masks[best_idx]

                    no_improve_count = 0

        return best_code

    # Main Loop: Multi-start
    best_solution = None
    best_size = float('inf')

    seed = 42
    while time.time() < deadline - 2.0:
        # Get initial greedy solution
        init_code = greedy_init(seed)

        if not init_code:
            break

        # Refine with local search
        refined_code = local_search(init_code, seed)

        if len(refined_code) < best_size:
            best_size = len(refined_code)
            best_solution = refined_code.copy()

        seed += 1

        # Early exit if we hit the theoretical lower bound?
        if best_size <= 156:
            break

    if best_solution is None:
        return []

    # Convert indices to digit lists
    result = []
    for idx in best_solution:
        result.append(digits_table[idx][:])

    return result

def _construct_general(q, n, radius, deadline):
    # Simple fallback for other parameters
    all_words = []
    for i in range(q**n):
        digits = [0] * n
        tmp = i
        for pos in range(n):
            digits[pos] = tmp % q
            tmp //= q
        all_words.append(tuple(digits))

    balls = []
    for w in all_words:
        ball = set()
        for v in all_words:
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                ball.add(v)
        balls.append(ball)

    covered = set()
    selected = []
    remaining = set(all_words)

    while remaining:
        best_idx = -1
        best_count = -1
        for i, w in enumerate(all_words):
            if w in covered:
                continue
            count = len(balls[i] & remaining)
            if count > best_count:
                best_count = count
                best_idx = i

        if best_idx == -1:
            break

        selected.append(all_words[best_idx])
        covered.update(balls[best_idx])
        remaining -= set(balls[best_idx])

        if time.time() > deadline - 5.0:
            break

    return [list(w) for w in selected]
# EVOLVE-BLOCK-END
