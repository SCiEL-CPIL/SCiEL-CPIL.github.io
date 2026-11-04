"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _covered_by(word, codewords, radius):
    for cw in codewords:
        if _hamming_dist(word, cw) <= radius:
            return True
    return False


def _build_coverage(codewords, q, n, radius):
    """Return a set of all words covered by the given codewords."""
    covered = set()
    for cw in codewords:
        # Add the codeword itself
        covered.add(tuple(cw))
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    covered.add(tuple(lst))
    return covered


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Greedy removal: repeatedly try to remove codewords that are redundant
    # (i.e., all words they cover are also covered by other codewords)
    # Use a coverage count array for efficiency
    total_words = q ** n

    # Map each word to an index
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute the covered words for each codeword pattern
    # For a codeword, the covered set is: itself + all words at distance 1
    # We can compute this efficiently

    def get_covered_indices(cw):
        """Return a list of indices of words covered by cw (including itself)."""
        indices = [word_to_idx(cw)]
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    indices.append(word_to_idx(lst))
        return indices

    # Build coverage count: for each word in the space, how many codewords cover it
    def build_coverage_count(cw_list):
        counts = [0] * total_words
        for cw in cw_list:
            covered_indices = get_covered_indices(cw)
            for idx in covered_indices:
                counts[idx] += 1
        return counts

    # Check if removing a codeword keeps coverage valid
    def is_redundant(cw, counts):
        # Find words covered by cw that would lose their last covering codeword
        covered_indices = get_covered_indices(cw)
        for idx in covered_indices:
            if counts[idx] <= 1:
                return False  # This word is only covered by cw
        return True

    # Iterative greedy removal with multiple passes and different orderings
    improved = True
    while improved and time.time() < deadline:
        improved = False
        counts = build_coverage_count(codewords)
        
        # Try removing in reverse order first
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            if is_redundant(codewords[i], counts):
                codewords.pop(i)
                improved = True
                # Rebuild counts after removal
                counts = build_coverage_count(codewords)
                break  # Restart the scan from the end
        
        if not improved and time.time() < deadline:
            # Try forward order
            counts = build_coverage_count(codewords)
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                if is_redundant(codewords[i], counts):
                    codewords.pop(i)
                    improved = True
                    # Rebuild counts after removal
                    counts = build_coverage_count(codewords)
                    break  # Restart the scan from the beginning
        
        if not improved and time.time() < deadline:
            # Try random order
            counts = build_coverage_count(codewords)
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                if is_redundant(codewords[i], counts):
                    codewords.pop(i)
                    improved = True
                    # Rebuild counts after removal
                    counts = build_coverage_count(codewords)
                    break  # Restart with new random order

    return codewords
# EVOLVE-BLOCK-END
