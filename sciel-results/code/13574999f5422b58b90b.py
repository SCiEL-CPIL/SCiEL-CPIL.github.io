"""Rewritten program: tetracode-product init + fast local search with deadline guard."""
import time
import itertools
import random

_DEADLINE = 145.0  # safety margin before 150 s kill


def _tetracode_words():
    """The 9 ternary tetracode words of length 4 (perfect code, radius 1)."""
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _neighbors_within_radius(word, q, n, radius):
    """Yield all words within Hamming distance <= radius of `word`."""
    # For radius 1: word itself + n*(q-1) neighbors
    if radius == 1:
        yield word
        for pos in range(n):
            for sym in range(q):
                if sym == word[pos]:
                    continue
                w = list(word)
                w[pos] = sym
                yield w
    else:
        # General radius (slow, but only used if radius != 1)
        for delta in itertools.product(range(q), repeat=n):
            dist = sum(1 for i in range(n) if delta[i] != word[i])
            if dist <= radius:
                yield [(word[i] + delta[i]) % q for i in range(n)]


def _word_to_index(word, q, n):
    """Convert a word (list of ints in 0..q-1) to a base-q integer index."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _index_to_word(idx, q, n):
    """Convert a base-q integer index back to a word."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def construct(q, n, radius):
    start_time = time.time()
    total_words = q ** n

    # --- Initialization: tetracode direct sum ---
    # If n >= 4, use tetracode on first 4 coords, all words on remaining.
    # This gives 9 * 3^(n-4) words.
    if n >= 4 and q == 3:
        tetra = _tetracode_words()
        codewords = []
        for h in tetra:
            for tail in itertools.product(range(q), repeat=n - 4):
                codewords.append(h + list(tail))
    else:
        # Fallback: greedy set cover
        codewords = _greedy_init(q, n, radius, start_time)

    # Remove duplicates (shouldn't happen, but safe)
    seen = set()
    unique = []
    for c in codewords:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    codewords = unique

    # --- Build coverage data structures ---
    # covered_count[w_idx] = number of codewords that cover word w_idx
    # For each codeword, we know which words it covers.
    # We'll maintain:
    #   covered_count: array of length total_words
    #   For removal checks, we need to know if removing codeword i leaves any word uncovered.
    #   A word is uniquely covered by codeword i if covered_count[w] == 1 and i covers w.

    # Precompute: for each codeword, the set of word indices it covers.
    # This is expensive to store fully, so we compute on-the-fly for the local search.

    # Build covered_count
    covered_count = [0] * total_words
    for c in codewords:
        for w in _neighbors_within_radius(c, q, n, radius):
            covered_count[_word_to_index(w, q, n)] += 1

    # Check initial validity
    initial_valid = all(cnt > 0 for cnt in covered_count)
    best_codewords = list(codewords)
    best_len = len(codewords)

    if not initial_valid:
        # Greedy init should be valid, but just in case
        return best_codewords

    # --- Local search: try to remove codewords ---
    # For each codeword i, check if removing it leaves all words covered.
    # A word w is uncovered after removing i iff covered_count[w] == 1 and i covers w.
    # So we need: for all w covered by i, covered_count[w] >= 2.

    # To make this fast, precompute for each codeword the list of words it covers.
    # For n=8, q=3, radius=1: each codeword covers 1 + 8*2 = 17 words.
    # 729 codewords * 17 = ~12393 entries. Manageable.

    cw_covered_words = []  # cw_covered_words[i] = list of word indices covered by codeword i
    for c in codewords:
        words = []
        for w in _neighbors_within_radius(c, q, n, radius):
            words.append(_word_to_index(w, q, n))
        cw_covered_words.append(words)

    # Try removals
    for i in range(len(codewords)):
        if time.time() - start_time > _DEADLINE:
            break
        # Check if codeword i can be removed
        removable = True
        for w_idx in cw_covered_words[i]:
            if covered_count[w_idx] == 1:
                removable = False
                break
        if removable:
            # Remove codeword i
            for w_idx in cw_covered_words[i]:
                covered_count[w_idx] -= 1
            codewords.pop(i)
            cw_covered_words.pop(i)
            best_codewords = list(codewords)
            best_len = len(codewords)
            # Restart removal loop (indices shifted)
            i = -1  # will be incremented to 0

    # --- Local search: try swaps ---
    # For each pair (i, candidate), check if removing i and adding candidate reduces size or maintains validity.
    # Actually, for swaps: remove i, check if valid. If not, find best candidate to add back.
    # Better: for each i, temporarily remove it. If invalid, find the candidate that covers the most
    #   newly-uncovered words. If that candidate covers all, we have a swap (same size, but may enable future removals).
    # If valid after removal, we already handled it above.

    # For swap: we want to replace codeword i with a better one that allows more removals later.
    # Simpler approach: try all (remove i, add candidate) pairs where candidate is not already in codewords.
    # A swap is beneficial if after the swap, we can remove more codewords.

    # Given time constraints, let's do a simpler local search:
    # Repeatedly try to remove any removable codeword (already done above).
    # Then try: for each i, remove it. If invalid, find the best single replacement.
    # Accept the replacement if it leads to a state where more removals are possible.

    # Actually, the most impactful move is: find a codeword whose removal makes some words uncovered,
    # then find a single new codeword that covers all those uncovered words AND is not already present.
    # This is a "swap" that maintains validity but changes the code structure, potentially enabling further removals.

    # Let's do iterative improvement:
    improved = True
    while improved and (time.time() - start_time < _DEADLINE):
        improved = False
        for i in range(len(codewords)):
            if time.time() - start_time > _DEADLINE:
                break
            # Temporarily remove codeword i
            for w_idx in cw_covered_words[i]:
                covered_count[w_idx] -= 1

            # Find uncovered words
            uncovered = [w_idx for w_idx in cw_covered_words[i] if covered_count[w_idx] == 0]

            if not uncovered:
                # Codeword i is redundant! Remove it for good.
                codewords.pop(i)
                cw_covered_words.pop(i)
                best_codewords = list(codewords)
                best_len = len(codewords)
                improved = True
                i = -1
                continue

            # Need to find a replacement that covers all uncovered words
            # The replacement must be within radius of all words in `uncovered`.
            # That means the replacement must be within distance <= radius of each word in uncovered.
            # For radius=1, this means the replacement is at distance <=1 from each uncovered word.
            # The intersection of these balls is small.

            # For radius=1: a word r covers all words in `uncovered` iff
            #   for each w in uncovered, dist(r, w) <= 1.
            # This means r is within distance 1 of every word in uncovered.
            # If |uncovered| >= 2, r must be very close to all of them.

            # Find candidate: try all words in the intersection of balls of radius 1 around each uncovered word.
            # For simplicity, take the first uncovered word u0. The candidate must be in ball(u0, 1).
            # Then check if it's also in ball(u, 1) for all other u in uncovered.

            if len(uncovered) == 1:
                u0 = _index_to_word(uncovered[0], q, n)
                candidates = list(_neighbors_within_radius(u0, q, n, radius))
            else:
                # Intersect balls
                u0 = _index_to_word(uncovered[0], q, n)
                candidates = list(_neighbors_within_radius(u0, q, n, radius))
                for u_idx in uncovered[1:]:
                    u = _index_to_word(u_idx, q, n)
                    ball = set(_word_to_index(w, q, n) for w in _neighbors_within_radius(u, q, n, radius))
                    candidates = [w for w in candidates if _word_to_index(w, q, n) in ball]

            # Check each candidate: is it already in codewords? If so, skip (no new info).
            # If not, try adding it and see if we get a better structure.
            # For now, accept the first valid candidate that isn't already present.
            existing_set = set(tuple(c) for c in codewords)
            found_replacement = False
            for cand in candidates:
                cand_tuple = tuple(cand)
                if cand_tuple in existing_set:
                    continue
                # Check if adding cand covers all uncovered words
                cand_covers = all(
                    sum(1 for a, b in zip(cand, _index_to_word(w_idx, q, n)) if a != b) <= radius
                    for w_idx in uncovered
                )
                if cand_covers:
                    # Perform swap: add cand, keep i removed
                    # Update covered_count for cand's covered words
                    for w in _neighbors_within_radius(cand, q, n, radius):
                        covered_count[_word_to_index(w, q, n)] += 1
                    # Replace codeword i with cand
                    codewords[i] = cand
                    cw_covered_words[i] = [_word_to_index(w, q, n) for w in _neighbors_within_radius(cand, q, n, radius)]
                    found_replacement = True
                    improved = True
                    break

            if not found_replacement:
                # Restore covered_count
                for w_idx in cw_covered_words[i]:
                    covered_count[w_idx] += 1

    # Final validity check
    final_valid = True
    for cnt in covered_count:
        if cnt == 0:
            final_valid = False
            break

    if final_valid:
        return codewords
    else:
        # Fallback to best valid solution found
        return best_codewords


def _greedy_init(q, n, radius, start_time):
    """Greedy set-cover initialization."""
    total_words = q ** n
    uncovered = [1] * total_words  # 1 if not yet covered
    codewords = []
    cw_covered = []

    while any(uncovered) and (time.time() - start_time < _DEADLINE):
        # Find the word that covers the most uncovered words
        best_count = -1
        best_word = None
        for w in range(total_words):
            if uncovered[w] == 0:
                continue
            # Count how many uncovered words are within radius of word w
            word = _index_to_word(w, q, n)
            count = 0
            for u in _neighbors_within_radius(word, q, n, radius):
                u_idx = _word_to_index(u, q, n)
                if uncovered[u_idx]:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = word
            if time.time() - start_time > _DEADLINE:
                break

        if best_word is None:
            break

        codewords.append(best_word)
        for u in _neighbors_within_radius(best_word, q, n, radius):
            u_idx = _word_to_index(u, q, n)
            uncovered[u_idx] = 0

    return codewords
