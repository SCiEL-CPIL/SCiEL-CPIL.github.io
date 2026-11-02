import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)

    N = q ** n
    if N > 200000:
        return _constructive_cover(q, n, radius)

    # --- Build index mapping ---
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    word_to_idx = {tuple(w): idx for idx, w in enumerate(all_words)}

    # Precompute all balls as numpy arrays
    ball_list = []
    for i in range(N):
        w = all_words[i]
        idxs = [i]
        for p in range(n):
            for v in range(q):
                if v == w[p]:
                    continue
                power = q ** (n - 1 - p)
                new_idx = i - w[p] * power + v * power
                idxs.append(new_idx)
        ball_list.append(np.array(idxs, dtype=np.int32))

    # Coverage tracking
    covered_count = np.zeros(N, dtype=np.int32)

    def add_codeword(idx):
        covered_count[ball_list[idx]] += 1

    def remove_codeword(idx):
        covered_count[ball_list[idx]] -= 1

    def is_fully_covered():
        return np.all(covered_count > 0)

    # Tetracode product seed
    tetracode = np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)])
    seed_indices = []
    for i in range(9):
        for j in range(9):
            word = np.concatenate([tetracode[i], tetracode[j]])
            idx = word_to_idx[tuple(word)]
            seed_indices.append(idx)

    selected = list(seed_indices)
    for idx in selected:
        add_codeword(idx)

    # Greedy fill
    if not is_fully_covered():
        while not is_fully_covered() and time.time() < deadline - 10:
            best_gain = -1
            best_idx = -1
            for i in range(N):
                if i in selected:
                    continue
                gain = np.sum(covered_count[ball_list[i]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected.append(best_idx)
            add_codeword(best_idx)

    # Local search
    best_size = len(selected)
    best_selected = list(selected)
    in_selected = np.zeros(N, dtype=bool)
    in_selected[selected] = True

    def try_remove_one():
        for idx in selected:
            if np.sum(covered_count[ball_list[idx]] == 1) == 0:
                selected.remove(idx)
                in_selected[idx] = False
                remove_codeword(idx)
                return True
        return False

    while try_remove_one():
        pass

    if len(selected) < best_size:
        best_size = len(selected)
        best_selected = list(selected)

    # ILS with swaps and perturbation
    rng = np.random.default_rng(42)
    stall_count = 0
    max_stall = 30

    while time.time() < deadline - 5:
        # Removals
        while try_remove_one():
            pass

        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
            stall_count = 0
        else:
            stall_count += 1

        # Try swaps
        if stall_count >= 5:
            swapped = False
            for i in selected:
                at_risk = ball_list[i][covered_count[ball_list[i]] == 1]
                if len(at_risk) == 0:
                    continue
                if len(at_risk) > 4:
                    continue
                candidates = set(ball_list[at_risk[0]].tolist())
                for w in at_risk[1:]:
                    candidates &= set(ball_list[w].tolist())
                candidates -= set(selected)
                for j in candidates:
                    selected.remove(i)
                    in_selected[i] = False
                    remove_codeword(i)
                    selected.append(j)
                    in_selected[j] = True
                    add_codeword(j)
                    if try_remove_one():
                        swapped = True
                        break
                    # Undo
                    selected.remove(j)
                    in_selected[j] = False
                    remove_codeword(j)
                    selected.append(i)
                    in_selected[i] = True
                    add_codeword(i)
                    break
                if swapped:
                    break

        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= max_stall:
            # Perturbation
            k = min(len(selected), rng.integers(2, 8))
            removed = rng.choice(len(selected), size=k, replace=False)
            for pos in sorted(removed, reverse=True):
                idx = selected[pos]
                selected.remove(idx)
                in_selected[idx] = False
                remove_codeword(idx)
            while not is_fully_covered() and time.time() < deadline - 5:
                best_gain = -1
                best_idx = -1
                for i in range(N):
                    if in_selected[i]:
                        continue
                    gain = np.sum(covered_count[ball_list[i]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1 or best_gain == 0:
                    break
                selected.append(best_idx)
                in_selected[best_idx] = True
                add_codeword(best_idx)
            while try_remove_one():
                pass
            stall_count = 0
            max_stall += 10

    # Final cleanup
    while try_remove_one():
        pass
    if len(selected) < best_size:
        best_size = len(selected)
        best_selected = list(selected)

    if not is_fully_covered():
        covered_count[:] = 0
        for idx in best_selected:
            add_codeword(idx)
        while not is_fully_covered() and time.time() < deadline:
            best_gain = -1
            best_idx = -1
            for i in range(N):
                if i in best_selected:
                    continue
                gain = np.sum(covered_count[ball_list[i]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            best_selected.append(best_idx)
            add_codeword(best_idx)

    return [all_words[i].tolist() for i in best_selected]


def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Precompute neighbors: neighbors[w] = array of 17 indices
    neighbors = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = q ** (n - 1 - pos)
            digit = (w // weight) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                neighbors[w, idx] = w + (new_digit - digit) * weight
                idx += 1

    # --- Greedy construction from tetracode seed ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # Build seed set
    seed_set = set()
    for a in tetracode:
        for b in tetracode:
            word = a + b
            # Compute index
            idx = 0
            for d in word:
                idx = idx * q + d
            seed_set.add(idx)

    cover_count = np.zeros(total, dtype=np.int32)
    in_selected = np.zeros(total, dtype=bool)
    selected = list(seed_set)
    for idx in selected:
        cover_count[neighbors[idx]] += 1
        in_selected[idx] = True

    # Greedy fill
    uncovered_mask = cover_count == 0
    while np.any(uncovered_mask) and time.time() < deadline - 10:
        # Vectorized: for each candidate i, count how many uncovered words it covers
        # counts[i] = sum(uncovered_mask[neighbors[i]])
        counts = uncovered_mask[neighbors].sum(axis=1)
        # Mask out already selected
        counts[in_selected] = -1
        max_count = np.max(counts)
        if max_count <= 0:
            break
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        selected.append(best_idx)
        in_selected[best_idx] = True
        cover_count[neighbors[best_idx]] += 1
        uncovered_mask[neighbors[best_idx]] = False

    # --- Local search with removals ---
    def try_remove():
        # A selected word i is removable if all words in its ball have count > 1
        # i.e., min(cover_count[neighbors[i]]) > 1
        # Only check selected words
        for idx in selected:
            if np.min(cover_count[neighbors[idx]]) > 1:
                selected.remove(idx)
                in_selected[idx] = False
                cover_count[neighbors[idx]] -= 1
                return True
        return False

    while try_remove():
        pass

    # --- ILS with swaps ---
    rng = np.random.default_rng(42)
    best_size = len(selected)
    best_selected = list(selected)
    stall = 0
    max_stall = 25

    while time.time() < deadline - 5:
        # Removals
        while try_remove():
            pass

        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
            stall = 0
        else:
            stall += 1

        # Swap attempt: remove one, try to find a replacement that enables removal
        if stall >= 3 and len(selected) > 486:
            # Pick a random selected word to remove
            pos = rng.integers(0, len(selected))
            cw = selected[pos]
            at_risk = neighbors[cw][cover_count[neighbors[cw]] == 1]

            if len(at_risk) == 0:
                # Should have been removed already
                pass
            elif len(at_risk) <= 5:
                # Find candidates that cover all at_risk
                # Candidate j must be in neighbors[u] for all u in at_risk
                # And j must not be in selected (and j != cw)
                candidates = set(neighbors[at_risk[0]].tolist())
                for u in at_risk[1:]:
                    candidates &= set(neighbors[u].tolist())
                candidates -= set(selected)

                if candidates:
                    # Try each candidate
                    for j in list(candidates)[:10]:  # Limit to 10 candidates
                        # Remove cw
                        selected.remove(cw)
                        in_selected[cw] = False
                        cover_count[neighbors[cw]] -= 1
                        # Add j
                        selected.append(j)
                        in_selected[j] = True
                        cover_count[neighbors[j]] += 1
                        # Try removals
                        removed = False
                        while try_remove():
                            removed = True
                        if removed:
                            # Keep the swap
                            break
                        else:
                            # Undo
                            selected.remove(j)
                            in_selected[j] = False
                            cover_count[neighbors[j]] -= 1
                            selected.append(cw)
                            in_selected[cw] = True
                            cover_count[neighbors[cw]] += 1

        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
            stall = 0

        # Perturbation
        if stall >= max_stall:
            k = min(len(selected), int(rng.integers(2, 8)))
            to_remove = rng.choice(len(selected), size=k, replace=False)
            for pos in sorted(to_remove, reverse=True):
                idx = selected[pos]
                selected.remove(idx)
                in_selected[idx] = False
                cover_count[neighbors[idx]] -= 1
            # Greedy re-add
            uncovered_mask = cover_count == 0
            while np.any(uncovered_mask) and time.time() < deadline - 5:
                counts = uncovered_mask[neighbors].sum(axis=1)
                counts[in_selected] = -1
                max_count = np.max(counts)
                if max_count <= 0:
                    break
                candidates = np.where(counts == max_count)[0]
                best_idx = candidates[0]
                selected.append(best_idx)
                in_selected[best_idx] = True
                cover_count[neighbors[best_idx]] += 1
                uncovered_mask[neighbors[best_idx]] = False
            # Removals
            while try_remove():
                pass
            stall = 0
            max_stall += 10

    # Final cleanup
    while try_remove():
        pass
    if len(selected) < best_size:
        best_size = len(selected)
        best_selected = list(selected)

    # Ensure validity
    cover_count[:] = 0
    for idx in best_selected:
        cover_count[neighbors[idx]] += 1
    if np.any(cover_count == 0):
        uncovered_mask = cover_count == 0
        while np.any(uncovered_mask) and time.time() < deadline:
            counts = uncovered_mask[neighbors].sum(axis=1)
            counts[in_selected] = -1
            max_count = np.max(counts)
            if max_count <= 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            best_selected.append(best_idx)
            in_selected[best_idx] = True
            cover_count[neighbors[best_idx]] += 1
            uncovered_mask[neighbors[best_idx]] = False
        while True:
            removed = False
            for idx in list(best_selected):
                if np.min(cover_count[neighbors[idx]]) > 1:
                    best_selected.remove(idx)
                    in_selected[idx] = False
                    cover_count[neighbors[idx]] -= 1
                    removed = True
            if not removed:
                break

    # Convert indices to words
    result = []
    for idx in best_selected:
        word = []
        w = idx
        for pos in range(n - 1, -1, -1):
            word.append(w % q)
            w //= q
        word.reverse()
        result.append(word)
    return result


def _constructive_cover(q, n, radius):
    """Fallback constructive approach."""
    if q == 3 and n == 8:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]

    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)

    balls = np.zeros((num_words, num_words), dtype=bool)
    np.fill_diagonal(balls, True)
    for i in range(num_words):
        for j in range(num_words):
            if np.sum(all_words[i] != all_words[j]) <= radius:
                balls[i, j] = True

    selected = []
    covered = np.zeros(num_words, dtype=bool)
    while not covered.all():
        uncovered_count = balls[:, ~covered].sum(axis=1)
        best_idx = np.argmax(uncovered_count)
        if uncovered_count[best_idx] == 0:
            break
        selected.append(best_idx)
        covered |= balls[best_idx, :]

    return [all_words[i].tolist() for i in selected]
