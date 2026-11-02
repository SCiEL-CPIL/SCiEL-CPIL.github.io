"""Combined program: bit-parallel greedy init + batched deletion + coverer-intersection swaps."""
import time
import random
from itertools import product


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)

    # Fallback for other cases
    if n == 0:
        return [[]]
    if n == 1:
        return [[i] for i in range(q)]
    if n == 2:
        return [[a, b] for a in range(q) for b in range(q)]
    if n == 3:
        return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
    return [list(w) for w in product(range(q), repeat=n)]


def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187

    # Precompute powers of 3 for fast index arithmetic
    p3 = [1] * n
    for i in range(1, n):
        p3[i] = p3[i - 1] * 3

    # Precompute digits for each index
    digits_table = [[0] * n for _ in range(N)]
    for i in range(N):
        tmp = i
        for pos in range(n):
            digits_table[i][pos] = tmp % 3
            tmp //= 3

    # Precompute ball_masks: bitmask of all words within Hamming distance 1
    ball_masks = [0] * N
    for i in range(N):
        mask = 1 << i
        for pos in range(n):
            orig = digits_table[i][pos]
            for val in range(q):
                if val != orig:
                    new_i = i + (val - orig) * p3[pos]
                    mask |= (1 << new_i)
        ball_masks[i] = mask

    all_mask = (1 << N) - 1

    # Precompute coverers as bitmasks: coverers[point] = bitmask of codewords covering point
    coverers = [0] * N
    for cw_idx in range(N):
        m = ball_masks[cw_idx]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverers[idx] |= (1 << cw_idx)
            m ^= lsb

    # --- Greedy Set Cover ---
    def greedy_set_cover(rng):
        uncovered = all_mask
        selected = []
        selected_set = set()

        while uncovered:
            if time.time() > deadline - 30:
                break
            best_i = -1
            best_count = 0
            # Randomized scan
            order = list(range(N))
            rng.shuffle(order)
            for i in order:
                if i in selected_set:
                    continue
                new_bits = ball_masks[i] & uncovered
                if new_bits:
                    count = new_bits.bit_count()
                    if count > best_count:
                        best_count = count
                        best_i = i
            if best_i == -1:
                break
            selected.append(best_i)
            selected_set.add(best_i)
            uncovered &= ~ball_masks[best_i]

        # Ensure full coverage
        if uncovered:
            for i in range(N):
                if not uncovered:
                    break
                if i not in selected_set:
                    selected.append(i)
                    selected_set.add(i)
                    uncovered &= ~ball_masks[i]

        return selected

    # --- Batched redundancy removal using bitmask coverage ---
    def batched_removal(code, code_set, cov_mask):
        """Remove all non-essential codewords. cov_mask[i] = bitmask of codewords covering point i."""
        removed = 0
        to_remove = []
        for c in code:
            m = ball_masks[c]
            is_essential = False
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                # Check if any other codeword covers idx
                others = cov_mask[idx] & ~(1 << c)
                if others == 0:
                    is_essential = True
                    break
                m ^= lsb
            if not is_essential:
                to_remove.append(c)

        for c in to_remove:
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cov_mask[idx] &= ~(1 << c)
                m ^= lsb
            code.remove(c)
            code_set.discard(c)
            removed += 1
        return removed

    # --- Local search with swap ---
    def local_search(code, code_set, cov_mask, rng):
        best_code = list(code)
        best_size = len(code)
        no_improvement = 0
        max_no_improvement = 60

        while time.time() < deadline - 2:
            improved = False

            # Phase 1: Batched deletion
            removed = batched_removal(code, code_set, cov_mask)
            if removed > 0:
                improved = True
                no_improvement = 0
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                continue

            # Phase 2: Swap
            if not code:
                break

            # Try up to 30 swap attempts
            swap_found = False
            for _ in range(30):
                if time.time() > deadline - 2:
                    break
                c = rng.choice(code)
                m = ball_masks[c]
                unique_points = []
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cov_mask[idx] == (1 << c):  # only c covers this point
                        unique_points.append(idx)
                    m ^= lsb

                if not unique_points:
                    # Shouldn't happen after batched removal, but handle
                    m = ball_masks[c]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cov_mask[idx] &= ~(1 << c)
                        m ^= lsb
                    code.remove(c)
                    code_set.discard(c)
                    improved = True
                    no_improvement = 0
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    swap_found = True
                    break

                # Find candidates covering all unique points
                cand_mask = coverers[unique_points[0]]
                for p in unique_points[1:]:
                    cand_mask &= coverers[p]
                    if not cand_mask:
                        break

                if not cand_mask:
                    continue

                # Filter out existing codewords
                cand_mask &= ~_code_set_mask(code_set)

                if not cand_mask:
                    continue

                # Pick best candidate: minimize overlap with existing coverage
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb

                if len(candidates) > 20:
                    rng.shuffle(candidates)
                    candidates = candidates[:20]

                best_new = -1
                best_cost = float('inf')
                for cand in candidates:
                    m_cand = ball_masks[cand]
                    cost = 0
                    while m_cand:
                        lsb = m_cand & -m_cand
                        idx = lsb.bit_length() - 1
                        cost += cov_mask[idx].bit_count()
                        m_cand ^= lsb
                    if cost < best_cost:
                        best_cost = cost
                        best_new = cand

                if best_new != -1:
                    # Perform swap
                    m_old = ball_masks[c]
                    while m_old:
                        lsb = m_old & -m_old
                        idx = lsb.bit_length() - 1
                        cov_mask[idx] &= ~(1 << c)
                        m_old ^= lsb
                    m_new = ball_masks[best_new]
                    while m_new:
                        lsb = m_new & -m_new
                        idx = lsb.bit_length() - 1
                        cov_mask[idx] |= (1 << best_new)
                        m_new ^= lsb
                    code.remove(c)
                    code_set.discard(c)
                    code.append(best_new)
                    code_set.add(best_new)
                    improved = True
                    no_improvement = 0
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    swap_found = True
                    break

            if not improved:
                no_improvement += 1
                if no_improvement >= max_no_improvement:
                    break

        return best_code

    def _code_set_mask(code_set):
        """Build bitmask of all codewords in code_set."""
        mask = 0
        for c in code_set:
            mask |= (1 << c)
        return mask

    # --- Main search loop ---
    best_codewords = None
    best_size = N + 1

    # Deterministic greedy first
    rng0 = random.Random(42)
    code = greedy_set_cover(rng0)
    code_set = set(code)
    cov_mask = [0] * N
    for c in code:
        m = ball_masks[c]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cov_mask[idx] |= (1 << c)
            m ^= lsb

    batched_removal(code, code_set, cov_mask)
    result = local_search(code, code_set, cov_mask, rng0)
    if len(result) < best_size:
        best_size = len(result)
        best_codewords = result

    # Randomized restarts
    rng = random.Random(12345)
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        code = greedy_set_cover(rng)
        code_set = set(code)
        cov_mask = [0] * N
        for c in code:
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cov_mask[idx] |= (1 << c)
                m ^= lsb

        batched_removal(code, code_set, cov_mask)
        result = local_search(code, code_set, cov_mask, rng)

        if len(result) < best_size:
            best_size = len(result)
            best_codewords = result

    if best_codewords is None:
        # Fallback: return full space
        return [list(w) for w in product(range(q), repeat=n)]

    # Convert indices back to words
    return [digits_table[i][:] for i in best_codewords]
