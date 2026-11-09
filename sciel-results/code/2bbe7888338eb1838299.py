"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _build_coverage_matrix(codewords, q, n, radius):
    """
    Build a coverage matrix: coverage_matrix[i] is the set of word-indices
    covered by codeword i.  Words are indexed 0..q^n-1 via base-q encoding.
    """
    total = q ** n
    m = len(codewords)
    # Precompute the list of neighbour indices for each codeword
    # A codeword at index c covers all words within Hamming distance <= radius.
    # For radius=1: covers c itself plus all words that differ in exactly one position.
    # For general radius: covers all words within that radius.

    # For efficiency with radius=1, we can compute neighbours via digit manipulation.
    # For general radius, we enumerate.
    coverage = []
    for c in codewords:
        # Convert codeword to integer index
        c_int = 0
        for d in reversed(c):
            c_int = c_int * q + d
        covered = set()
        _add_neighbours(c, q, n, radius, covered)
        coverage.append(covered)
    return coverage


def _add_neighbours(word, q, n, radius, out_set):
    """Add all words within Hamming distance <= radius to out_set (as int indices)."""
    # Convert word to integer
    base = 0
    for d in reversed(word):
        base = base * q + d
    out_set.add(base)
    if radius <= 0:
        return
    # Enumerate all changes of up to 'radius' positions
    # For radius=1: single position changes
    if radius == 1:
        for pos in range(n):
            digit = word[pos]
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                # Compute new index
                # word is stored with word[0] as most significant? 
                # In _int_to_word, word[i] = val % q for i=0..n-1, so word[0] is least significant.
                # Let's be consistent: treat word[0] as least significant digit.
                # base = sum(word[i] * q^i for i in 0..n-1)
                delta = (new_digit - digit) * (q ** pos)
                out_set.add(base + delta)
        return
    # General radius: enumerate combinations
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(q), repeat=r):
                new_word = word[:]
                ok = True
                for pos, val in zip(positions, values):
                    if val == word[pos]:
                        ok = False
                        break
                    new_word[pos] = val
                if not ok:
                    continue
                new_int = 0
                for d in reversed(new_word):
                    new_int = new_int * q + d
                out_set.add(new_int)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin

    # Tetracode direct-sum construction: 9 * 3^(n-4) codewords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        # Fallback: all words (small n)
        initial_codewords = [list(w) for w in itertools.product(range(q), repeat=n)]

    # Verify initial construction covers everything
    total_words = q ** n
    covered_set = set()
    for cw in initial_codewords:
        _add_neighbours(cw, q, n, radius, covered_set)
    if len(covered_set) < total_words:
        # Fallback: add missing words
        for w_int in range(total_words):
            if w_int not in covered_set:
                w = [0] * n
                v = w_int
                for i in range(n):
                    w[i] = v % q
                    v //= q
                initial_codewords.append(w)
                _add_neighbours(w, q, n, radius, covered_set)

    codewords = list(initial_codewords)

    # Build coverage sets
    coverage = _build_coverage_matrix(codewords, q, n, radius)

    # Iterative removal: try to remove codewords that are redundant
    # A codeword is redundant if every word it covers is also covered by at least one other codeword.
    # Use a count-based approach: for each word, count how many codewords cover it.
    # A codeword i can be removed if for every word w in coverage[i], count[w] >= 2.

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        m = len(codewords)
        if m == 0:
            break

        # Compute coverage counts
        # Use a dict or array of counts
        counts = np.zeros(total_words, dtype=np.int32)
        for i in range(m):
            for w in coverage[i]:
                counts[w] += 1

        # Try to remove codewords
        # A codeword i is removable if all words in coverage[i] have count >= 2
        for i in range(m):
            if time.time() > deadline - 5:
                break
            removable = True
            for w in coverage[i]:
                if counts[w] < 2:
                    removable = False
                    break
            if removable:
                # Remove codeword i
                codewords.pop(i)
                coverage.pop(i)
                # Decrement counts
                # We'll recompute counts in next iteration
                improved = True
                break  # Restart loop to recompute counts

    # Critical word heuristic: try to swap codewords to reduce count
    # Find words covered by exactly one codeword (critical words).
    # For each critical word, try to find an alternative codeword that covers it
    # and covers at least as many other words as the current one.
    # This is complex; instead, try a simpler approach:
    # For each codeword, check if there's another word (not in codewords) that covers
    # all the words the current codeword covers, plus possibly more.
    # If so, replace.

    # Simpler: try adding a random word and see if we can remove two codewords.
    # Or: for each codeword, try shifting one digit and see if coverage improves.

    # Let's do a local search: for each codeword, try all single-digit changes,
    # and if the new word covers a superset of the old word's unique coverage, accept.
    # This is O(m * n * q * (coverage size)) which might be expensive.
    # Instead, just do the removal pass more aggressively.

    # Second pass: more aggressive removal using a different order
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        m = len(codewords)
        if m == 0:
            break

        # Compute coverage counts
        counts = np.zeros(total_words, dtype=np.int32)
        for i in range(m):
            for w in coverage[i]:
                counts[w] += 1

        # Try removing from the end to the beginning
        for i in range(m - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            removable = True
            for w in coverage[i]:
                if counts[w] < 2:
                    removable = False
                    break
            if removable:
                codewords.pop(i)
                coverage.pop(i)
                improved = True
                break

    # Final verification
    covered_set = set()
    for cw in codewords:
        _add_neighbours(cw, q, n, radius, covered_set)
    if len(covered_set) < total_words:
        # Shouldn't happen, but fallback to initial
        return initial_codewords

    return codewords
# EVOLVE-BLOCK-END
