import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before the 150 s kill

    if q != 3 or n != 6 or radius != 1:
        # Generic fallback: tetracode-style product (valid, not optimal)
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    total = 3 ** n
    powers = [3 ** p for p in range(n)]

    # --- Vectorized precompute of neighbor (ball) indices: shape (total, 13) ---
    base_idx = np.arange(total, dtype=np.int32)
    neighbors = np.empty((total, 13), dtype=np.int32)
    neighbors[:, 0] = base_idx
    k = 1
    for p in range(n):
        pw = powers[p]
        for alt in range(1, 3):
            d = (base_idx // pw) % 3
            alt_arr = np.full(total, alt, dtype=np.int32)
            neighbors[:, k] = base_idx - d * pw + alt_arr * pw
            k += 1

    def to_word(idx):
        return [(idx // powers[p]) % 3 for p in range(n)]

    def to_idx(word):
        idx = 0
        for w in word:
            idx = idx * 3 + w
        return idx

    # --- Initial code: tetracode direct sum (81 words) ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_idx = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    current_code = np.array(sorted(set(init_idx)), dtype=np.int32)
    code_set = set(int(x) for x in current_code)

    coverage = np.zeros(total, dtype=np.int32)
    for c in current_code:
        coverage[neighbors[c]] += 1

    best_code = current_code.copy()
    best_size = len(best_code)

    def greedy_remove(code_arr, cov, code_set_local):
        """Repeatedly remove any word whose whole ball is covered >=2 times."""
        while True:
            if time.time() > deadline:
                break
            mins = np.min(cov[neighbors[code_arr]], axis=1)
            removable = np.where(mins >= 2)[0]
            if removable.size == 0:
                break
            c = int(code_arr[removable[0]])
            cov[neighbors[c]] -= 1
            code_set_local.discard(c)
            code_arr = np.delete(code_arr, removable[0])
        return code_arr

    # Phase 1: aggressive greedy removal from the 81-word start
    current_code = greedy_remove(current_code, coverage, code_set)
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = current_code.copy()

    # Phase 2: directed local search with add-then-remove AND swap moves
    under = (coverage == 1).astype(np.int8)

    while time.time() < deadline - 5.0:
        # ---- Move A: add a high-gain word, then greedy-remove ----
        best_candidate = None
        best_candidate_size = len(current_code)

        gain = np.sum(under[neighbors], axis=1)
        gain_arr = gain.copy()
        for c in code_set:
            gain_arr[c] = -1

        top_k = min(5, total)
        top_indices = np.argsort(gain_arr)[-top_k:][::-1]

        for c2 in top_indices:
            c2 = int(c2)
            if gain_arr[c2] <= 0:
                continue

            saved_code = current_code.copy()
            saved_cov = coverage.copy()
            saved_set = set(code_set)

            coverage[neighbors[c2]] += 1
            code_set.add(c2)
            current_code = np.append(current_code, c2)
            current_code = greedy_remove(current_code, coverage, code_set)

            if len(current_code) < best_candidate_size:
                best_candidate_size = len(current_code)
                best_candidate = (current_code.copy(), coverage.copy(), set(code_set))

            current_code = saved_code
            coverage = saved_cov
            code_set = saved_set

        # ---- Move B: swap a codeword for a better one (escape local minima) ----
        if best_candidate is None:
            # find a redundant codeword to swap out
            mins = np.min(coverage[neighbors[current_code]], axis=1)
            swappable = np.where(mins >= 2)[0]
            if swappable.size > 0:
                # pick the one with the most slack (highest min coverage)
                slacks = mins[swappable]
                best_slack = slacks.max()
                best_slack_idx = swappable[slacks == best_slack]
                c_remove = int(current_code[best_slack_idx[random.randrange(len(best_slack_idx))]])

                saved_code = current_code.copy()
                saved_cov = coverage.copy()
                saved_set = set(code_set)

                coverage[neighbors[c_remove]] -= 1
                code_set.discard(c_remove)
                current_code = np.delete(current_code, np.where(current_code == c_remove)[0][0])

                # find best add candidate after removal
                under_local = (coverage == 1).astype(np.int8)
                gain_local = np.sum(under_local[neighbors], axis=1)
                gain_local_arr = gain_local.copy()
                for c in code_set:
                    gain_local_arr[c] = -1
                gain_local_arr[c_remove] = -1

                top_k2 = min(5, total)
                top_indices2 = np.argsort(gain_local_arr)[-top_k2:][::-1]

                best_swap = None
                best_swap_size = len(current_code)
                for c2 in top_indices2:
                    c2 = int(c2)
                    if gain_local_arr[c2] <= 0:
                        continue
                    coverage[neighbors[c2]] += 1
                    code_set.add(c2)
                    current_code = np.append(current_code, c2)
                    current_code = greedy_remove(current_code, coverage, code_set)
                    if len(current_code) < best_swap_size:
                        best_swap_size = len(current_code)
                        best_swap = (current_code.copy(), coverage.copy(), set(code_set))
                    # revert
                    coverage[neighbors[c2]] -= 1
                    code_set.discard(c2)
                    current_code = np.delete(current_code, np.where(current_code == c2)[0][0])

                if best_swap is not None:
                    current_code, coverage, code_set = best_swap
                else:
                    # revert swap
                    current_code = saved_code
                    coverage = saved_cov
                    code_set = saved_set

        # Apply best candidate if found
        if best_candidate is not None:
            current_code, coverage, code_set = best_candidate
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code.copy()
                under = (coverage == 1).astype(np.int8)
        else:
            # No beneficial move: try random perturbation
            safe = np.where(np.min(coverage[neighbors[current_code]], axis=1) >= 2)[0]
            if safe.size == 0:
                if len(current_code) > 70:
                    for _ in range(min(3, safe.size if safe.size > 0 else 0)):
                        if time.time() > deadline:
                            break
                        safe = np.where(np.min(coverage[neighbors[current_code]], axis=1) >= 2)[0]
                        if safe.size == 0:
                            break
                        c_drop = int(current_code[safe[random.randrange(safe.size)]])
                        coverage[neighbors[c_drop]] -= 1
                        code_set.discard(c_drop)
                        current_code = np.delete(current_code, safe[0])
                    under = (coverage == 1).astype(np.int8)
                else:
                    break
            else:
                c_drop = int(current_code[safe[random.randrange(safe.size)]])
                coverage[neighbors[c_drop]] -= 1
                code_set.discard(c_drop)
                current_code = np.delete(current_code, safe[0])
                under = (coverage == 1).astype(np.int8)

    # Final polish: one more greedy pass with tighter deadline
    if time.time() < deadline - 2.0:
        current_code = greedy_remove(current_code, coverage, code_set)
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code.copy()

    return [to_word(int(x)) for x in best_code]
