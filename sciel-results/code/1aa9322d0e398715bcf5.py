# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill

    # --- 1. Build a valid covering code via direct sum of ternary tetracode ---
    # Tetracode: 9 words of length 4, perfect 1-covering of {0,1,2}^4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    code = list(base)
    m = len(code)  # 9 * 3^(n-4)

    # --- 2. Precompute neighbor lists for each codeword (all words within Hamming dist <= radius) ---
    # For radius=1, neighbors are: self + n single-position changes
    def neighbors_of(word, q, n, radius):
        """Return set of word-indices (in full space) within Hamming distance <= radius from word."""
        # We use a frozenset of tuples for membership, but for efficiency we track
        # which codeword indices cover each position. Actually, let's track coverage counts.
        # For local search, we need: for each point in space, how many codewords cover it.
        # But space is 3^8 = 6561, which is small! So we can use an array.
        pass

    # Since q^n = 3^8 = 6561 is small, use a coverage-count array over the whole space.
    # Map each word to an index in [0, q^n)
    total_space = q ** n

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Coverage count: cov[idx] = number of codewords within Hamming distance <= radius of word idx
    cov = [0] * total_space

    def update_cov(word, delta):
        """Add delta (+1 or -1) to coverage of all points within Hamming distance <= radius of word."""
        # Enumerate all points within Hamming distance <= radius from word
        # For radius=1: the word itself + n neighbors
        # For general radius: enumerate all subsets of positions of size <= radius and all symbol changes
        # Since n=8, radius=1: 1 + 8*2 = 17 points (for q=3)
        # For radius=1, q=3: positions to change: 8, symbols to change to: 2 each
        # Total: 1 + 8*2 = 17
        if radius == 1:
            cov[word_to_idx(word)] += delta
            for pos in range(n):
                orig = word[pos]
                for sym in range(q):
                    if sym == orig:
                        continue
                    new_word = list(word)
                    new_word[pos] = sym
                    cov[word_to_idx(new_word)] += delta
        else:
            # General case: enumerate all words within Hamming distance <= radius
            # For n=8, radius up to 3, this is manageable
            from itertools import combinations, product
            for r in range(radius + 1):
                for positions in combinations(range(n), r):
                    if r == 0:
                        cov[word_to_idx(word)] += delta
                    else:
                        for syms in product(range(q), repeat=r):
                            new_word = list(word)
                            all_diff = True
                            for i, p in enumerate(positions):
                                if syms[i] == word[p]:
                                    all_diff = False
                                    break
                                new_word[p] = syms[i]
                            if all_diff:
                                cov[word_to_idx(new_word)] += delta

    # Initialize coverage
    for w in code:
        update_cov(w, 1)

    # --- 3. Local search: try to remove redundant codewords ---
    # A codeword is removable if after removing it, all points it covered are still covered by others.
    # i.e., for all points p in N(codeword), cov[p] > 1.

    def is_removable(ci):
        """Check if codeword at index ci can be removed while maintaining coverage."""
        w = code[ci]
        # Check all points in N(w) have cov >= 2
        if radius == 1:
            if cov[word_to_idx(w)] < 2:
                return False
            for pos in range(n):
                orig = w[pos]
                for sym in range(q):
                    if sym == orig:
                        continue
                    new_word = list(w)
                    new_word[pos] = sym
                    if cov[word_to_idx(new_word)] < 2:
                        return False
            return True
        else:
            from itertools import combinations, product
            for r in range(radius + 1):
                for positions in combinations(range(n), r):
                    if r == 0:
                        if cov[word_to_idx(w)] < 2:
                            return False
                    else:
                        for syms in product(range(q), repeat=r):
                            new_word = list(w)
                            all_diff = True
                            for i, p in enumerate(positions):
                                if syms[i] == w[p]:
                                    all_diff = False
                                    break
                                new_word[p] = syms[i]
                            if all_diff:
                                if cov[word_to_idx(new_word)] < 2:
                                    return False
            return True

    # Greedy removal pass
    removed_any = True
    while removed_any and time.time() < deadline - 30:
        removed_any = False
        # Try removing in random order to avoid bias
        indices = list(range(len(code)))
        random.shuffle(indices)
        for ci in indices:
            if time.time() > deadline - 30:
                break
            if is_removable(ci):
                update_cov(code[ci], -1)
                code.pop(ci)
                removed_any = True
        # After a full pass, do another pass since removals can enable more removals

    # --- 4. 2-opt: remove two codewords, add one ---
    # Try pairs of codewords to remove, then find if a single new codeword can cover all uncovered points.
    # This is expensive, so limit iterations.

    def try_two_opt():
        """Attempt to remove two codewords and add one to maintain coverage."""
        nonlocal code, cov
        m = len(code)
        if m < 3:
            return False

        # Sample random pairs
        for _ in range(200):
            if time.time() > deadline - 20:
                return False
            i, j = random.sample(range(m), 2)
            wi = code[i]
            wj = code[j]

            # Temporarily remove both
            update_cov(wi, -1)
            update_cov(wj, -1)

            # Find all points that are now uncovered (cov == 0)
            uncovered = []
            # Only need to check points that were covered by wi or wj
            # Collect all points in N(wi) union N(wj)
            points_to_check = set()
            if radius == 1:
                points_to_check.add(word_to_idx(wi))
                points_to_check.add(word_to_idx(wj))
                for pos in range(n):
                    for sym in range(q):
                        nw = list(wi)
                        nw[pos] = sym
                        if sym != wi[pos]:
                            points_to_check.add(word_to_idx(nw))
                        nw2 = list(wj)
                        nw2[pos] = sym
                        if sym != wj[pos]:
                            points_to_check.add(word_to_idx(nw2))
            else:
                from itertools import combinations, product
                for r in range(radius + 1):
                    for positions in combinations(range(n), r):
                        if r == 0:
                            points_to_check.add(word_to_idx(wi))
                            points_to_check.add(word_to_idx(wj))
                        else:
                            for syms in product(range(q), repeat=r):
                                nw = list(wi)
                                all_diff = True
                                for k, p in enumerate(positions):
                                    if syms[k] == wi[p]:
                                        all_diff = False
                                        break
                                    nw[p] = syms[k]
                                if all_diff:
                                    points_to_check.add(word_to_idx(nw))
                                nw2 = list(wj)
                                all_diff = True
                                for k, p in enumerate(positions):
                                    if syms[k] == wj[p]:
                                        all_diff = False
                                        break
                                    nw2[p] = syms[k]
                                if all_diff:
                                    points_to_check.add(word_to_idx(nw2))

            for pidx in points_to_check:
                if cov[pidx] == 0:
                    uncovered.append(pidx)

            if not uncovered:
                # Both can be removed!
                code.pop(j)  # pop higher index first
                code.pop(i)
                return True

            # Find a candidate replacement: a word that covers all uncovered points
            # The replacement must be within radius of every uncovered point.
            # This means the replacement must be in the intersection of N(p) for all p in uncovered.
            # For small uncovered sets, enumerate candidates.
            if len(uncovered) <= 10:
                # Find intersection of neighbor sets
                # Start with neighbors of first uncovered point
                def get_neighbors(pidx):
                    pw = idx_to_word(pidx)
                    result = set()
                    if radius == 1:
                        result.add(pidx)
                        for pos in range(n):
                            for sym in range(q):
                                nw = list(pw)
                                nw[pos] = sym
                                if sym != pw[pos]:
                                    result.add(word_to_idx(nw))
                    else:
                        from itertools import combinations, product
                        for r in range(radius + 1):
                            for positions in combinations(range(n), r):
                                if r == 0:
                                    result.add(pidx)
                                else:
                                    for syms in product(range(q), repeat=r):
                                        nw = list(pw)
                                        all_diff = True
                                        for k, p in enumerate(positions):
                                            if syms[k] == pw[p]:
                                                all_diff = False
                                                break
                                            nw[p] = syms[k]
                                        if all_diff:
                                            result.add(word_to_idx(nw))
                    return result

                candidate_set = get_neighbors(uncovered[0])
                for pidx in uncovered[1:]:
                    candidate_set &= get_neighbors(pidx)
                    if not candidate_set:
                        break

                if candidate_set:
                    # Pick a candidate that maximizes coverage of currently low-cov points
                    best_candidate = None
                    best_score = -1
                    for cand_idx in list(candidate_set)[:50]:  # limit
                        cand_w = idx_to_word(cand_idx)
                        # Score: sum of cov over neighbors of cand (higher is better for filling gaps)
                        score = 0
                        if radius == 1:
                            score += cov[cand_idx]
                            for pos in range(n):
                                for sym in range(q):
                                    nw = list(cand_w)
                                    nw[pos] = sym
                                    if sym != cand_w[pos]:
                                        score += cov[word_to_idx(nw)]
                        # Prefer candidates that cover more currently uncovered points
                        score *= 10  # weight
                        best_candidate = cand_idx
                        if score > best_score:
                            best_score = score
                            best_candidate = cand_idx

                    if best_candidate is not None:
                        cand_w = idx_to_word(best_candidate)
                        # Add the candidate
                        update_cov(cand_w, 1)
                        code.pop(j)
                        code.pop(i)
                        code.append(cand_w)
                        return True

            # Restore coverage
            update_cov(wi, 1)
            update_cov(wj, 1)

        return False

    # Run 2-opt several times
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        if not try_two_opt():
            # Shuffle and try again
            random.shuffle(code)
            # Rebuild cov
            cov = [0] * total_space
            for w in code:
                update_cov(w, 1)

    # --- 5. Final greedy removal pass ---
    removed_any = True
    while removed_any and time.time() < deadline - 5:
        removed_any = False
        indices = list(range(len(code)))
        random.shuffle(indices)
        for ci in indices:
            if time.time() > deadline - 5:
                break
            if is_removable(ci):
                update_cov(code[ci], -1)
                code.pop(ci)
                removed_any = True

    return code
# EVOLVE-BLOCK-END
