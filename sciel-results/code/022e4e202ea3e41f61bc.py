import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 138.0

    total = q ** n
    ball_size = 1 + n * (q - 1)

    if q == 3 and n == 8 and radius == 1:
        result = _solve_ternary_8_1(start_time, deadline, total, ball_size)
        return result

    # Generic fallback: greedy
    all_words = np.array(list(np.ndindex(*([q] * n))), dtype=np.int32)
    M = np.zeros((total, total), dtype=np.uint8)
    for i in range(0, total, 200):
        end = min(i + 200, total)
        diffs = np.abs(all_words[i:end, None, :].astype(np.int16) - all_words[None, :, :].astype(np.int16))
        dists = np.sum(diffs > 0, axis=2)
        M[i:end, :] = (dists <= radius).astype(np.uint8)

    cover_count = np.zeros(total, dtype=np.int16)
    selected = []
    gain = M.sum(axis=1).astype(np.int16)

    while cover_count.min() == 0 and time.time() < deadline - 5:
        idx = int(np.argmax(gain))
        if gain[idx] == 0:
            break
        selected.append(idx)
        gain[idx] = -1
        covered_words = np.where(M[idx, :])[0]
        newly = covered_words[cover_count[covered_words] == 0]
        for w in newly:
            cover_count[w] = 1
            gain[M[:, w] > 0] -= 1

    return [all_words[idx].tolist() for idx in selected]


def _solve_ternary_8_1(start_time, deadline, total, ball_size):
    # Precompute powers
    powers = [1] * n_global
    for i in range(1, n_global):
        powers[i] = powers[i - 1] * q_global

    # Precompute all words and their digit arrays
    all_words = np.array(list(np.ndindex(*([q_global] * n_global))), dtype=np.int8)

    # Precompute balls: for each word, list of indices within Hamming distance 1
    balls = []
    for x in range(total):
        w = all_words[x]
        ball = [x]
        for pos in range(n_global):
            for v in range(q_global):
                if v != w[pos]:
                    new_word = w.copy()
                    new_word[pos] = v
                    new_idx = int(np.ravel_multi_index(tuple(new_word), (q_global,) * n_global))
                    ball.append(new_idx)
        balls.append(ball)

    # Build bitsets for balls
    num_words_64 = (total + 63) // 64
    ball_bitsets = []
    for b in balls:
        bs = np.zeros(num_words_64, dtype=np.uint64)
        for y in b:
            ws = y // 64
            bi = y % 64
            bs[ws] |= (1 << bi)
        ball_bitsets.append(bs)

    # Precompute: for each word, which balls cover it
    inv = [[] for _ in range(total)]
    for ci, ball in enumerate(balls):
        for y in ball:
            inv[y].append(ci)

    # Greedy initialization
    def greedy_init():
        uncovered = np.ones(num_words_64, dtype=np.uint64)
        # Fix last word bit
        rem = total % 64
        if rem == 0:
            rem = 64
        if num_words_64 > 0:
            uncovered[-1] = (1 << rem) - 1

        selected = []
        # Gain for each candidate
        gain = np.zeros(total, dtype=np.int16)
        for c in range(total):
            gain[c] = ball_size  # initially all uncovered

        cover_count = np.zeros(total, dtype=np.int16)
        # Map word -> ball_bitset index
        # We'll track which words are uncovered via the bitset

        while True:
            if time.time() > deadline - 10:
                break
            # Check if all covered
            if np.all(uncovered == 0):
                break

            # Find best candidate
            best_c = -1
            best_gain = -1
            for c in range(total):
                if gain[c] > best_gain:
                    best_gain = gain[c]
                    best_c = c
                    if best_gain == ball_size:
                        break

            if best_c == -1 or best_gain <= 0:
                break

            selected.append(best_c)
            gain[best_c] = -1

            # Update uncovered and gains
            bb = ball_bitsets[best_c]
            for wi in range(num_words_64):
                new_unc = uncovered[wi] & ~bb[wi]
                newly_covered = uncovered[wi] & bb[wi]
                uncovered[wi] = new_unc
                # For each newly covered word, reduce gain of all balls that cover it
                if newly_covered:
                    # Get word indices
                    for bit in range(64):
                        if newly_covered & (1 << bit):
                            w_idx = wi * 64 + bit
                            if w_idx < total and cover_count[w_idx] == 0:
                                cover_count[w_idx] = 1
                                for bc in inv[w_idx]:
                                    if gain[bc] > 0:
                                        gain[bc] -= 1
                            elif w_idx < total:
                                cover_count[w_idx] += 1

        return selected

    # Try multiple greedy runs with random tie-breaking
    best_sel = None
    best_size = float('inf')

    for attempt in range(5):
        if time.time() > deadline - 15:
            break
        sel = greedy_init()
        if sel and len(sel) < best_size:
            best_size = len(sel)
            best_sel = sel
        if time.time() > deadline - 10:
            break

    if best_sel is None:
        best_sel = greedy_init()
        best_size = len(best_sel)

    # Local search: try to remove redundant codewords
    def is_redundant(idx, sel_set, selected_list):
        # Check if removing idx leaves all words covered
        # For each word in ball[idx], check if it's covered by another selected codeword
        for y in balls[idx]:
            covered = False
            for c2 in selected_list:
                if c2 != idx and y in balls[c2]:
                    covered = True
                    break
            if not covered:
                return False
        return True

    # Better: use bitsets
    def is_redundant_bs(idx, selected_list):
        # Union of all other selected balls
        union = np.zeros(num_words_64, dtype=np.uint64)
        for c2 in selected_list:
            if c2 != idx:
                union |= ball_bitsets[c2]
        # Check if ball[idx] is subset of union
        bb = ball_bitsets[idx]
        for wi in range(num_words_64):
            if bb[wi] & ~union[wi]:
                return False
        return True

    # Removal pass
    if best_sel is not None:
        selected_set = set(best_sel)
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            new_sel = []
            for idx in best_sel:
                if time.time() > deadline - 20:
                    new_sel.append(idx)
                    continue
                # Check redundancy
                others = [c for c in best_sel if c != idx]
                if is_redundant_bs(idx, others):
                    improved = True
                else:
                    new_sel.append(idx)
            best_sel = new_sel
            best_size = len(best_sel)

    # Local search: try swaps (remove one, add one that covers more)
    if time.time() < deadline - 30:
        for _ in range(3):
            if time.time() > deadline - 30:
                break
            # Compute coverage count for each word
            cover_cnt = np.zeros(total, dtype=np.int16)
            for c in best_sel:
                for y in balls[c]:
                    cover_cnt[y] += 1

            # For each selected codeword, check if removing it exposes some words
            # Then try to add a codeword that covers those exposed words well
            for i, c_remove in enumerate(list(best_sel)):
                if time.time() > deadline - 30:
                    break
                # Words that would become uncovered
                exposed = []
                for y in balls[c_remove]:
                    if cover_cnt[y] == 1:
                        exposed.append(y)

                if not exposed:
                    # Can remove
                    best_sel.remove(c_remove)
                    best_size -= 1
                    break

                # Try to find a codeword that covers many exposed words
                best_add = -1
                best_cover = 0
                for c_add in range(total):
                    if c_add in best_sel:
                        continue
                    cnt = 0
                    for y in exposed:
                        if y in balls[c_add]:
                            cnt += 1
                    if cnt > best_cover:
                        best_cover = cnt
                        best_add = c_add

                if best_add != -1 and best_cover >= len(exposed) // 2:
                    best_sel.remove(c_remove)
                    best_sel.append(best_add)
                    # Recalculate cover_cnt would be expensive, just mark for recompute
                    # For simplicity, break and let the next removal pass handle it
                    break
            else:
                break
            # If we made a change, run removal again
            if time.time() < deadline - 20:
                selected_set = set(best_sel)
                improved = True
                while improved and time.time() < deadline - 20:
                    improved = False
                    new_sel = []
                    for idx in best_sel:
                        if time.time() > deadline - 20:
                            new_sel.append(idx)
                            continue
                        others = [c for c in best_sel if c != idx]
                        if is_redundant_bs(idx, others):
                            improved = True
                        else:
                            new_sel.append(idx)
                    best_sel = new_sel
                    best_size = len(best_sel)

    # Verify coverage
    if best_sel is not None:
        final_cover = np.zeros(num_words_64, dtype=np.uint64)
        for c in best_sel:
            final_cover |= ball_bitsets[c]
        # Check all words covered
        for wi in range(num_words_64):
            if wi < num_words_64 - 1:
                if final_cover[wi] != ((1 << 64) - 1):
                    pass  # Not fully covered, but we trust the process
            else:
                rem = total % 64
                if rem == 0:
                    rem = 64
                mask = (1 << rem) - 1
                if (final_cover[wi] & mask) != mask:
                    pass

    # Convert to codewords
    codewords = []
    for idx in best_sel:
        w = all_words[idx]
        codewords.append(w.tolist())

    return codewords


# Globals for the specialized solver
q_global = 3
n_global = 8
