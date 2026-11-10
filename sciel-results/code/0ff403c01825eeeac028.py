"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(q, n, radius):
    deadline = time.time() + 110.0
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_size = 9 * (3 ** (n - 4))
    base = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    if len(base) <= 156:
        return base

    # Greedy set-cover: start with a random subset, repeatedly add the codeword that covers
    # the most currently-uncovered words.  This is a smooth surrogate for the 0/1 covering
    # objective; we use a soft-max-like weighting implicitly by counting uncovered words.
    import random
    import numpy as np

    rng = random.Random(42)
    total_words = q ** n
    if total_words > 500000:
        return base  # too large to enumerate

    # Precompute all words as indices
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)

    # Map each codeword to the set of words it covers (within Hamming distance <= radius)
    # For speed, precompute a lookup: for each word, which codewords cover it?
    # Instead, for each codeword, compute its covering set via a vectorized distance.
    # With base_size up to ~2187*9=19683 and total_words up to 2187, this is feasible.

    # Precompute the covering sets as bitsets for speed
    # Each codeword covers words within Hamming distance <= radius.
    # We'll represent the covered set as a boolean array of length total_words.

    # For memory, if base_size * total_words is too large, fall back
    if base_size * total_words > 200_000_000:
        return base

    # Compute covering sets
    # all_words: (total_words, n)
    # base: list of lists, convert to array
    base_arr = np.array(base, dtype=np.uint8)  # (base_size, n)

    # Compute Hamming distances: for each codeword, how many positions differ from each word
    # dist[i, j] = number of positions where base_arr[i] != all_words[j]
    # This is (base_size, total_words, n) which might be large.
    # For n=7, base_size=2187, total_words=2187: 2187*2187*7 ≈ 33M bytes as uint8, OK.

    # dist = np.sum(base_arr[:, np.newaxis, :] != all_words[np.newaxis, :, :], axis=2)
    # This creates a (base_size, total_words) array of uint8.

    # To save memory, process in chunks
    chunk_size = max(1, 200_000_000 // (total_words * n))
    covering_sets = []
    for i in range(0, base_size, chunk_size):
        chunk = base_arr[i:i + chunk_size]
        dist = np.sum(chunk[:, np.newaxis, :] != all_words[np.newaxis, :, :], axis=2)
        covered = dist <= radius  # (chunk_len, total_words)
        covering_sets.append(covered)

    covering_sets = np.concatenate(covering_sets, axis=0)  # (base_size, total_words)

    # Greedy set cover
    uncovered = np.ones(total_words, dtype=np.uint8)
    selected = []
    selected_mask = np.zeros(base_size, dtype=np.uint8)

    # Initial: pick a random codeword
    idx = rng.randint(0, base_size - 1)
    selected.append(idx)
    selected_mask[idx] = 1
    uncovered[covering_sets[idx]] = 0

    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
        # Find the codeword that covers the most uncovered words
        # gain[i] = sum(covering_sets[i] & uncovered)
        # Vectorized: (covering_sets * uncovered).sum(axis=1)
        gains = (covering_sets * uncovered).sum(axis=1)
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        selected.append(best)
        selected_mask[best] = 1
        uncovered[covering_sets[best]] = 0

    result = [base[i] for i in selected]
    return result
# EVOLVE-BLOCK-END
