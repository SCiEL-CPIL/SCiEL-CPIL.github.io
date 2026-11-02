"""Fresh design: fast greedy covering + randomized local search with time guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 108.0  # safety margin under 120s

    total = q ** n
    pos_step = [q ** (n - 1 - i) for i in range(n)]

    # Precompute neighbor offsets for each position and symbol change
    # neighbor_offsets[pos][sym] = list of index offsets
    neighbor_offsets = []
    for i in range(n):
        step = pos_step[i]
        offsets = []
        for cur_sym in range(q):
            off_list = []
            for v in range(q):
                if v != cur_sym:
                    off_list.append((v - cur_sym) * step)
            offsets.append(off_list)
        neighbor_offsets.append(offsets)

    def word_to_idx(word):
        idx = 0
        for i, val in enumerate(word):
            idx += val * pos_step[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = (idx // pos_step[i]) % q
        return word

    def build_cover(codewords):
        """Return coverage count array."""
        cover = np.zeros(total, dtype=np.int32)
        for w in codewords:
            idx = word_to_idx(w)
            cover[idx] += 1
            if radius >= 1:
                for i in range(n):
                    cur_sym = w[i]
                    step = pos_step[i]
                    base = idx - cur_sym * step
                    for v in range(q):
                        if v != cur_sym:
                            cover[base + v * step] += 1
        return cover

    def is_valid_cover(cover):
        return cover.min() >= 1

    # Start with known good construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    seed = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    best = list(seed)
    best_len = len(seed)

    # Try aggressive deletion
    def try_delete(codewords, cover, max_iters=5000):
        deleted = 0
        for _ in range(max_iters):
            if time.time() > DEADLINE - 3:
                break
            found = -1
            for ci in range(len(codewords)):
                w = codewords[ci]
                idx = word_to_idx(w)
                if cover[idx] > 1:
                    # check all neighbors have cover >= 2
                    ok = True
                    for i in range(n):
                        cur_sym = w[i]
                        step = pos_step[i]
                        base = idx - cur_sym * step
                        for v in range(q):
                            if v != cur_sym:
                                if cover[base + v * step] < 2:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        found = ci
                        break
            if found < 0:
                break
            w = codewords[found]
            idx = word_to_idx(w)
            cover[idx] -= 1
            for i in range(n):
                cur_sym = w[i]
                step = pos_step[i]
                base = idx - cur_sym * step
                for v in range(q):
                    if v != cur_sym:
                        cover[base + v * step] -= 1
            codewords.pop(found)
            deleted += 1
        return deleted

    cover = build_cover(best)
    assert is_valid_cover(cover)
    try_delete(best, cover, 10000)
    if len(best) < best_len:
        best_len = len(best)

    # Randomized local search: try insert+delete cycles
    def randomized_ls(codewords, cover, time_budget):
        cur_len = len(codewords)
        end_time = time.time() + time_budget
        while time.time() < end_time:
            # Pick a random position to modify
            if random.random() < 0.5 and len(codewords) > 1:
                # Try removing a random codeword
                ci = random.randrange(len(codewords))
                w = codewords[ci]
                idx = word_to_idx(w)
                # Temporarily remove
                cover[idx] -= 1
                for i in range(n):
                    cur_sym = w[i]
                    step = pos_step[i]
                    base = idx - cur_sym * step
                    for v in range(q):
                        if v != cur_sym:
                            cover[base + v * step] -= 1
                if is_valid_cover(cover):
                    codewords.pop(ci)
                    cur_len = len(codewords)
                    # Try to delete more
                    try_delete(codewords, cover, 2000)
                    cur_len = len(codewords)
                else:
                    # Revert
                    cover[idx] += 1
                    for i in range(n):
                        cur_sym = w[i]
                        step = pos_step[i]
                        base = idx - cur_sym * step
                        for v in range(q):
                            if v != cur_sym:
                                cover[base + v * step] += 1
            else:
                # Try inserting a random codeword then deleting
                new_word = [random.randrange(q) for _ in range(n)]
                new_idx = word_to_idx(new_word)
                cover[new_idx] += 1
                for i in range(n):
                    cur_sym = new_word[i]
                    step = pos_step[i]
                    base = new_idx - cur_sym * step
                    for v in range(q):
                        if v != cur_sym:
                            cover[base + v * step] += 1
                codewords.append(new_word)
                # Try to delete more (including possibly the new one)
                try_delete(codewords, cover, 3000)
                if len(codewords) < cur_len:
                    cur_len = len(codewords)
                elif len(codewords) == cur_len:
                    # No improvement, revert the insert
                    # Remove the last added word (or find it)
                    # Simplified: just keep it if same length
                    pass
                else:
                    # Got worse, revert
                    # Find and remove the new_word
                    for ci in range(len(codewords) - 1, -1, -1):
                        if codewords[ci] == new_word:
                            w = codewords[ci]
                            idx = word_to_idx(w)
                            cover[idx] -= 1
                            for i in range(n):
                                cur_sym = w[i]
                                step = pos_step[i]
                                base = idx - cur_sym * step
                                for v in range(q):
                                    if v != cur_sym:
                                        cover[base + v * step] -= 1
                            codewords.pop(ci)
                            break
                    cur_len = len(codewords)
        return codewords, cover

    # Run local search with remaining time
    remaining = DEADLINE - time.time() - 2
    if remaining > 1:
        best, cover = randomized_ls(best, cover, remaining)

    return best
# EVOLVE-BLOCK-END
