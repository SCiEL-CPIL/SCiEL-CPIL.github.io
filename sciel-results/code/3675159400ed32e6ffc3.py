import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0

    if n == 0:
        return [[]]

    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Generic fallback
    return _generic_construct(q, n, radius, deadline)


def _solve_q3_n6_r1(deadline):
    q, n, r = 3, 6, 1
    total = q ** n  # 729

    # Precompute all words as indices
    all_words = list(itertools.product(range(q), repeat=n))
    idx_map = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbor lists (word + all words at Hamming distance <= 1)
    # Each word has 1 + n*(q-1) = 1 + 6*2 = 13 neighbors
    neighbors = [set() for _ in range(total)]
    for i, w in enumerate(all_words):
        neighbors[i].add(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                neighbors[i].add(idx_map[tuple(nw)])

    # Convert to lists for speed
    neighbors_list = [list(s) for s in neighbors]
    # Also keep as sets for membership tests
    neighbors_set = [set(s) for s in neighbors_list]

    # Build a numpy adjacency matrix for vectorized operations
    # adj[i, j] = 1 if i is a neighbor of j (i.e., word i is within distance 1 of word j)
    adj = np.zeros((total, total), dtype=np.uint8)
    for i in range(total):
        for j in neighbors_list[i]:
            adj[i, j] = 1

    # Start with tetracode direct sum construction
    # Tetracode: 9 codewords of length 4 over GF(3)
    # Direct sum with [3,2] repetition-like code of length 2
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Direct sum: for each tetracode word, combine with all 9 words of length 2
    # This gives 9 * 9 = 81 codewords
    code = set()
    for head in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = tuple(head + list(tail))
            code.add(idx_map[word])

    # Compute cover count using numpy
    code_arr = np.array(sorted(code), dtype=np.int32)
    cover_count = adj[:, code_arr].sum(axis=1)  # For each word, count how many codewords cover it

    def is_valid():
        return np.all(cover_count > 0)

    def try_remove(c, code_set, cc):
        """Try removing codeword c. Return True if successful (code still valid)."""
        # Check if any word would become uncovered
        for nb in neighbors_list[c]:
            if cc[nb] == 1:
                return False
        code_set.remove(c)
        for nb in neighbors_list[c]:
            cc[nb] -= 1
        return True

    def try_swap(c, new_c, code_set, cc):
        """Try replacing c with new_c. Return True if successful."""
        if new_c in code_set:
            return False
        # Words uniquely covered by c
        unique_to_c = []
        for nb in neighbors_list[c]:
            if cc[nb] == 1:
                unique_to_c.append(nb)

        # Check if new_c covers all unique_to_c
        for u in unique_to_c:
            if u not in neighbors_set[new_c]:
                return False

        # Do the swap
        code_set.remove(c)
        for nb in neighbors_list[c]:
            cc[nb] -= 1
        code_set.add(new_c)
        for nb in neighbors_list[new_c]:
            cc[nb] += 1
        return True

    def try_add(new_c, code_set, cc):
        """Add a new codeword. Return True if it covers at least one previously uncovered word."""
        if new_c in code_set:
            return False
        # Check if it covers any uncovered word
        for nb in neighbors_list[new_c]:
            if cc[nb] == 0:
                code_set.add(new_c)
                for nb2 in neighbors_list[new_c]:
                    cc[nb2] += 1
                return True
        return False

    # Local search: iteratively try to reduce code size
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1

        # Pass 1: try removing each codeword (multiple passes)
        for pass_num in range(3):
            code_list = list(code)
            random.shuffle(code_list)
            for c in code_list:
                if time.time() > deadline:
                    break
                if try_remove(c, code, cover_count):
                    improved = True
            if time.time() > deadline:
                break

        if time.time() > deadline:
            break

        # Pass 2: try swaps
        code_list = list(code)
        random.shuffle(code_list)
        for c in code_list:
            if time.time() > deadline:
                break
            # Find words uniquely covered by c
            unique_to_c = []
            for nb in neighbors_list[c]:
                if cover_count[nb] == 1:
                    unique_to_c.append(nb)

            if not unique_to_c:
                continue

            # Try to find a replacement
            # Candidate: words that cover all unique_to_c
            candidates = set()
            for u in unique_to_c:
                candidates.update(neighbors_list[u])
            candidates -= code

            for new_c in candidates:
                if time.time() > deadline:
                    break
                if try_swap(c, new_c, code, cover_count):
                    improved = True
                    break

        if time.time() > deadline:
            break

        # Pass 3: Try more aggressive moves - remove two codewords and add one
        # This can escape local minima
        if len(code) > 73:
            code_list = list(code)
            random.shuffle(code_list)
            for i in range(len(code_list)):
                if time.time() > deadline:
                    break
                c1 = code_list[i]
                for j in range(i + 1, len(code_list)):
                    if time.time() > deadline:
                        break
                    c2 = code_list[j]
                    # Find words uniquely covered by c1 or c2
                    unique_words = set()
                    for nb in neighbors_list[c1]:
                        if cover_count[nb] == 1:
                            unique_words.add(nb)
                    for nb in neighbors_list[c2]:
                        if cover_count[nb] == 1:
                            unique_words.add(nb)
                    
                    if not unique_words:
                        continue

                    # Find candidates that cover all unique_words
                    candidates = set()
                    for u in unique_words:
                        candidates.update(neighbors_list[u])
                    candidates -= code

                    for new_c in candidates:
                        if time.time() > deadline:
                            break
                        # Check if new_c covers all unique_words
                        ok = True
                        for u in unique_words:
                            if u not in neighbors_set[new_c]:
                                ok = False
                                break
                        if not ok:
                            continue

                        # Check that removing c1 and c2 doesn't uncover words
                        # that new_c doesn't cover
                        ok = True
                        for nb in neighbors_list[c1]:
                            if cover_count[nb] <= 2 and nb not in neighbors_set[new_c] and nb not in neighbors_list[c2]:
                                ok = False
                                break
                        if not ok:
                            continue
                        for nb in neighbors_list[c2]:
                            if cover_count[nb] <= 2 and nb not in neighbors_set[new_c] and nb not in neighbors_list[c1]:
                                ok = False
                                break
                        if not ok:
                            continue

                        # Do the move
                        code.remove(c1)
                        code.remove(c2)
                        for nb in neighbors_list[c1]:
                            cover_count[nb] -= 1
                        for nb in neighbors_list[c2]:
                            cover_count[nb] -= 1
                        code.add(new_c)
                        for nb in neighbors_list[new_c]:
                            cover_count[nb] += 1
                        improved = True
                        break
                    if improved:
                        break

    # Final greedy repair if needed
    while not is_valid() and time.time() < deadline:
        # Find uncovered words
        uncovered = [i for i in range(total) if cover_count[i] == 0]
        if not uncovered:
            break

        # Find best codeword to add
        best_c = -1
        best_cnt = -1
        for c in range(total):
            if c in code:
                continue
            cnt = sum(1 for u in uncovered if u in neighbors_set[c])
            if cnt > best_cnt:
                best_cnt = cnt
                best_c = c

        if best_c == -1 or best_cnt == 0:
            break

        code.add(best_c)
        for nb in neighbors_list[best_c]:
            cover_count[nb] += 1

    # Additional local search with random restarts of the removal phase
    while time.time() < deadline - 5:
        # Try a few random restarts of local search
        start_size = len(code)
        improved2 = True
        while improved2 and time.time() < deadline - 5:
            improved2 = False
            code_list = list(code)
            random.shuffle(code_list)
            for c in code_list:
                if time.time() > deadline - 5:
                    break
                if try_remove(c, code, cover_count):
                    improved2 = True
            # Try swaps
            code_list = list(code)
            random.shuffle(code_list)
            for c in code_list:
                if time.time() > deadline - 5:
                    break
                unique_to_c = []
                for nb in neighbors_list[c]:
                    if cover_count[nb] == 1:
                        unique_to_c.append(nb)
                if not unique_to_c:
                    continue
                candidates = set()
                for u in unique_to_c:
                    candidates.update(neighbors_list[u])
                candidates -= code
                for new_c in candidates:
                    if time.time() > deadline - 5:
                        break
                    if try_swap(c, new_c, code, cover_count):
                        improved2 = True
                        break

        if len(code) < start_size:
            continue
        else:
            break

    return [list(all_words[c]) for c in sorted(code)]


def _generic_construct(q, n, radius, deadline):
    """Simple greedy for other cases."""
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    idx_map = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbors
    neighbors = [set() for _ in range(total)]
    for i, w in enumerate(all_words):
        for r2 in range(radius + 1):
            for pos in itertools.combinations(range(n), r2):
                for vals in itertools.product(range(q), repeat=r2):
                    nw = list(w)
                    for p, v in zip(pos, vals):
                        nw[p] = v
                    neighbors[i].add(idx_map[tuple(nw)])

    neighbors_list = [list(s) for s in neighbors]

    code = set()
    cover_count = [0] * total

    while time.time() < deadline:
        uncovered = [i for i in range(total) if cover_count[i] == 0]
        if not uncovered:
            break

        best_c = -1
        best_cnt = -1
        for c in range(total):
            if c in code:
                continue
            cnt = sum(1 for u in uncovered if u in neighbors[c])
            if cnt > best_cnt:
                best_cnt = cnt
                best_c = c

        if best_c == -1 or best_cnt == 0:
            break

        code.add(best_c)
        for nb in neighbors_list[best_c]:
            cover_count[nb] += 1

    # Local search to reduce
    improved = True
    while improved and time.time() < deadline:
        improved = False
        code_list = list(code)
        random.shuffle(code_list)
        for c in code_list:
            if time.time() > deadline:
                break
            # Try removing
            ok = True
            for nb in neighbors_list[c]:
                if cover_count[nb] == 1:
                    ok = False
                    break
            if ok:
                code.remove(c)
                for nb in neighbors_list[c]:
                    cover_count[nb] -= 1
                improved = True

    return [list(all_words[c]) for c in sorted(code)]
