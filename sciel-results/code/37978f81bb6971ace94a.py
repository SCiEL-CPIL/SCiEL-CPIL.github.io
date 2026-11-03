import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return _generic_greedy(q, n, radius)

    start_time = time.time()
    deadline = start_time + 142.0  # safety margin under 150s

    total_words = q ** n  # 729

    # All words and index map
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # sphere[c] = set of word indices within Hamming distance radius of c
    sphere = [None] * total_words
    for i, w in enumerate(all_words):
        cov = set([i])
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                cov.add(word_to_idx[tuple(new_w)])
        sphere[i] = cov

    # Conflict graph: edge between two space words at Hamming distance 2.
    # A valid code is an independent set in this graph.
    # For the 2-for-1 move: if we remove c1 and c2 and add c3, and c3 must cover
    # the union of sphere[c1] and sphere[c2], then c3 must be within distance 1 of
    # every word in that union. In particular c3 must be within distance 1 of c1
    # AND within distance 1 of c2 (since c1 in sphere[c1] and c2 in sphere[c2]).
    # Precompute for each c the set of candidates within distance 1.
    within1 = [set() for _ in range(total_words)]
    for c in range(total_words):
        within1[c].add(c)
        w = all_words[c]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                within1[c].add(word_to_idx[tuple(new_w)])

    # Distance-2 conflict sets: conflict[c] = set of words at distance 2 from c
    # (these are the ones that cannot both be in a valid code).
    # Actually for independent set we forbid distance-2 pairs.
    # But note: distance-0 and distance-1 pairs are fine (a word can be a codeword
    # and another codeword can be its neighbor). The constraint is: no two codewords
    # at distance 2.
    conflict = [set() for _ in range(total_words)]
    for c in range(total_words):
        w = all_words[c]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                nw = tuple(new_w)
                idx1 = word_to_idx[nw]
                # now change one more coordinate to get distance 2
                for pos2 in range(n):
                    for val2 in range(q):
                        if val2 == nw[pos2]:
                            continue
                        w2 = list(nw)
                        w2[pos2] = val2
                        idx2 = word_to_idx[tuple(w2)]
                        if abs(idx1 - c) > 0:  # distance 2 (not 0 or 1)
                            conflict[c].add(idx2)

    # Tetracode direct-sum seed -> 81 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial.append(word_to_idx[tuple(h + list(tail))])
    active = set(initial)

    cover_count = np.zeros(total_words, dtype=int)
    for idx in active:
        for w in sphere[idx]:
            cover_count[w] += 1

    def is_redundant(idx):
        for w in sphere[idx]:
            if cover_count[w] == 1:
                return False
        return True

    def remove_codeword(idx):
        nonlocal cover_count
        for w in sphere[idx]:
            cover_count[w] -= 1
        active.discard(idx)

    def add_codeword(idx):
        nonlocal cover_count
        for w in sphere[idx]:
            cover_count[w] += 1
        active.add(idx)

    def is_independent(code_set):
        # Check no two codewords are at distance 2
        lst = list(code_set)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[j] in conflict[lst[i]]:
                    return False
        return True

    # Phase 1: greedy redundancy removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in list(active):
            if is_redundant(idx):
                remove_codeword(idx)
                changed = True

    best_size = len(active)
    best_state = set(active)

    if best_size <= 73:
        return [list(all_words[i]) for i in sorted(best_state)]

    rng = random.Random(20240517)

    def find_2for1():
        """Find c1, c2, c3 such that sphere[c1] U sphere[c2] is covered by c3,
        c3 not active, and {c1,c2,c3} is independent (no distance-2 pairs among
        the two removed; c3 can be anything). Return (c1,c2,c3) or None."""
        if len(active) < 2:
            return None
        active_list = list(active)
        # Sample a few random pairs
        for _ in range(30):
            c1, c2 = rng.sample(active_list, 2)
            # c3 must be within1[c1] AND within1[c2]
            candidates = within1[c1] & within1[c2]
            if not candidates:
                continue
            union = sphere[c1] | sphere[c2]
            for c3 in candidates:
                if c3 in active:
                    continue
                # Check c3 covers union
                if union.issubset(sphere[c3]):
                    return (c1, c2, c3)
        return None

    def maxsat_local_search():
        """Clause-learning style: forbid distance-2 pairs. Try to remove a codeword
        and repair by adding a codeword that covers the lost words, ensuring the
        result is an independent set. This is a Max-SAT-style local search where
        clauses are (not x_a or not x_b) for each distance-2 pair (a,b)."""
        if len(active) < 1:
            return False
        # Try to remove a random codeword c1
        c1 = rng.choice(list(active))
        # Words uniquely covered by c1
        unique = set()
        for w in sphere[c1]:
            if cover_count[w] == 1:
                unique.add(w)
        if not unique:
            remove_codeword(c1)
            return True
        # Find c2 that covers unique, c2 not in conflict[c1] (distance 2 forbidden),
        # and c2 not already active (if active, c1 was redundant)
        # c2 must cover every word in unique, so c2 is within distance 1 of each.
        # Intersect within1 over all words in unique.
        candidates = set(range(total_words))
        for w in unique:
            # c2 must be within1 of w? No: c2 covers w means w in sphere[c2],
            # i.e., c2 in within1[w].
            candidates &= within1[w]
        candidates.discard(c1)
        candidates -= active  # if c2 active, removing c1 alone works (redundant)
        # Also c2 must not be at distance 2 from any other active codeword
        # (to keep independence) -- but actually we only need the new set to be
        # independent. Since we're removing c1 and adding c2, we need:
        #   - c2 not at distance 2 from any active - {c1}
        #   - c1 was not at distance 2 from c2 (but we're removing c1, so fine)
        for c2 in list(candidates):
            ok = True
            for other in active:
                if other == c1:
                    continue
                if other in conflict[c2]:
                    ok = False
                    break
            if ok:
                remove_codeword(c1)
                add_codeword(c2)
                return True
        return False

    # Main search loop
    temperature = 50.0
    cooling = 0.99
    steps = 0
    max_steps = 1000000

    while time.time() < deadline and steps < max_steps:
        steps += 1

        # Periodic 2-for-1
        if steps % 50 == 0:
            move = find_2for1()
            if move is not None:
                c1, c2, c3 = move
                remove_codeword(c1)
                remove_codeword(c2)
                add_codeword(c3)
                if len(active) < best_size:
                    best_size = len(active)
                    best_state = set(active)
                    if best_size <= 73:
                        break
                continue

        # Max-SAT clause-learning local search (swap maintaining independence)
        if maxsat_local_search():
            if len(active) < best_size:
                best_size = len(active)
                best_state = set(active)
                if best_size <= 73:
                    break
            continue

        # Random swap: remove c1, add c2 (size maintaining) to escape local minima
        if len(active) > 0:
            c1 = rng.choice(list(active))
            unique = set()
            for w in sphere[c1]:
                if cover_count[w] == 1:
                    unique.add(w)
            if not unique:
                remove_codeword(c1)
                if len(active) < best_size:
                    best_size = len(active)
                    best_state = set(active)
                    if best_size <= 73:
                        break
                continue
            # Find c2 covering unique, not at distance 2 from c1 or other actives
            candidates = set(range(total_words))
            for w in unique:
                candidates &= within1[w]
            candidates.discard(c1)
            valid_c2 = []
            for c2 in candidates:
                if c2 in active:
                    continue
                ok = True
                for other in active:
                    if other == c1:
                        continue
                    if other in conflict[c2]:
                        ok = False
                        break
                if ok and c1 not in conflict[c2]:
                    valid_c2.append(c2)
            if valid_c2:
                c2 = rng.choice(valid_c2)
                remove_codeword(c1)
                add_codeword(c2)

        # Occasionally do a full redundancy sweep
        if steps % 200 == 0:
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for idx in list(active):
                    if is_redundant(idx):
                        remove_codeword(idx)
                        changed = True
            if len(active) < best_size:
                best_size = len(active)
                best_state = set(active)
                if best_size <= 73:
                    break

        temperature *= cooling

    # Final redundancy sweep
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in list(active):
            if is_redundant(idx):
                remove_codeword(idx)
                changed = True
    if len(active) < best_size:
        best_state = set(active)
        best_size = len(active)

    # Verify best_state is valid and independent; if not, fall back to active
    result = best_state
    # Final validation
    counts = np.zeros(total_words, dtype=int)
    for idx in result:
        for w in sphere[idx]:
            counts[w] += 1
    if np.any(counts == 0) or not is_independent(result):
        result = active
        # Re-verify active
        counts = np.zeros(total_words, dtype=int)
        for idx in result:
            for w in sphere[idx]:
                counts[w] += 1
        if np.any(counts == 0):
            # absolute fallback: tetracode seed
            result = set(initial)

    return [list(all_words[i]) for i in sorted(result)]


def _generic_greedy(q, n, radius):
    import itertools

    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total = len(all_words)

    sphere = []
    for w in all_words:
        cov = set()
        for cand in all_words:
            dist = sum(1 for a, b in zip(w, cand) if a != b)
            if dist <= radius:
                cov.add(word_to_idx[cand])
        sphere.append(cov)

    uncovered = set(range(total))
    selected = []
    while uncovered:
        best_c = -1
        best_count = -1
        for i, cov in enumerate(sphere):
            if i in [word_to_idx[c] for c in selected]:
                continue
            count = len(cov & uncovered)
            if count > best_count:
                best_count = count
                best_c = i
        if best_c == -1:
            break
        selected.append(all_words[best_c])
        uncovered -= sphere[best_c]

    return selected
