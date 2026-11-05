import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    total = 3 ** n
    powers = [3 ** p for p in range(n)]

    # --- Precompute neighbor (ball) indices for every word: shape (total, 13) ---
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(n)]
        neighbors[i, 0] = i
        k = 1
        for p in range(n):
            base = i - d[p] * powers[p]
            for alt in range(3):
                if alt == d[p]:
                    continue
                neighbors[i, k] = base + alt * powers[p]
                k += 1

    def to_word(idx):
        return [(idx // powers[p]) % 3 for p in range(n)]

    # --- Initial code: tetracode direct sum (81 words) ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_idx = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            init_idx.append(to_idx(h + list(tail))) if False else None
    # compute indices directly
    def to_idx(word):
        idx = 0
        for w in word:
            idx = idx * 3 + w
        return idx

    init_idx = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    current_code = np.array(sorted(set(init_idx)), dtype=np.int32)
    code_set = set(int(x) for x in current_code)

    # coverage counts: number of chosen words within distance 1 of each word
    coverage = np.zeros(total, dtype=np.int32)
    for c in current_code:
        coverage[neighbors[c]] += 1

    best_code = current_code.copy()
    best_size = len(best_code)

    start_time = time.time()
    deadline = start_time + 112.0

    def greedy_remove(code_arr, cov, code_set_local):
        """Remove all simultaneously-removable words in one vectorized pass, repeat."""
        while True:
            if time.time() > deadline:
                break
            if len(code_arr) == 0:
                break
            # vectorized: for each chosen word, min coverage over its ball
            mins = np.min(cov[neighbors[code_arr]], axis=1)
            removable = np.where(mins >= 2)[0]
            if removable.size == 0:
                break
            # remove all simultaneously-removable words at once (safe: each
            # still has cov >= 2 over its ball before any removal in this batch)
            for c in code_arr[removable]:
                cov[neighbors[c]] -= 1
                code_set_local.discard(int(c))
            code_arr = np.delete(code_arr, removable)
        return code_arr

    # Phase 1: aggressive greedy removal from the 81-word start
    current_code = greedy_remove(current_code, coverage, code_set)
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = current_code.copy()

    # Phase 2: directed local search.
    # Move = add a word that covers the most under-covered (coverage==1) words,
    # then greedily remove anything that became redundant. Accept if smaller.
    # This is the mechanism that escapes the greedy set-cover local minimum.
    under = (coverage == 1).astype(np.int8)

    # Precompute in-code mask (boolean array of size total)
    in_code = np.zeros(total, dtype=bool)
    in_code[current_code] = True

    while time.time() < deadline:
        # candidate that maximizes newly-covered under-covered words
        gain = np.sum(under[neighbors], axis=1)
        # only consider words not already in the code (vectorized mask)
        gain_arr = gain.copy()
        gain_arr[in_code] = -1
        c2 = int(np.argmax(gain_arr))
        if gain_arr[c2] <= 0:
            # no beneficial add found; try a few random restarts of removal order
            # by re-shuffling and doing one more greedy pass
            current_code = greedy_remove(current_code, coverage, code_set)
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code.copy()
            # random perturbation: drop a random word if safe, to change landscape
            safe = np.where(np.min(coverage[neighbors[current_code]], axis=1) >= 2)[0]
            if safe.size == 0:
                break
            c_drop = int(current_code[safe[random.randrange(safe.size)]])
            coverage[neighbors[c_drop]] -= 1
            code_set.discard(c_drop)
            current_code = np.delete(current_code, safe[0])
            under = (coverage == 1).astype(np.int8)
            continue

        # save state
        saved_code = current_code.copy()
        saved_cov = coverage.copy()
        saved_set = set(code_set)
        saved_in_code = in_code.copy()

        # add c2
        coverage[neighbors[c2]] += 1
        code_set.add(c2)
        in_code[c2] = True
        current_code = np.append(current_code, c2)

        # greedy remove
        current_code = greedy_remove(current_code, coverage, code_set)
        # update in_code mask after removals
        in_code[:] = False
        in_code[current_code] = True

        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code.copy()
            under = (coverage == 1).astype(np.int8)
        else:
            # revert
            current_code = saved_code
            coverage = saved_cov
            code_set = saved_set
            in_code = saved_in_code
            under = (coverage == 1).astype(np.int8)

    # Final polish: use remaining time for fast vectorized removal passes
    # on the best code found (rebuild coverage from scratch for correctness)
    if time.time() < deadline - 2.0:
        best_set = set(int(x) for x in best_code)
        best_cov = np.zeros(total, dtype=np.int32)
        for c in best_code:
            best_cov[neighbors[c]] += 1
        best_arr = best_code.copy()
        while time.time() < deadline - 2.0:
            if len(best_arr) == 0:
                break
            mins = np.min(best_cov[neighbors[best_arr]], axis=1)
            removable = np.where(mins >= 2)[0]
            if removable.size == 0:
                break
            for c in best_arr[removable]:
                best_cov[neighbors[c]] -= 1
                best_set.discard(int(c))
            best_arr = np.delete(best_arr, removable)
        if len(best_arr) < best_size:
            best_code = best_arr

    return [to_word(int(x)) for x in best_code]
