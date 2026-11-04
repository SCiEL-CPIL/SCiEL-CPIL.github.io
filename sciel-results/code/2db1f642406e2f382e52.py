"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_start_time = time.time()
_DEADLINE = 115.0  # Safety margin before 120s kill (evaluator limit)


def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode (9 words len 4, perfect radius 1) with full tails."""
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _precompute_neighbor_indices(q, n, radius):
    """
    For each word index 0..q^n-1, compute the list of indices of words
    within Hamming distance <= radius.
    Returns a list of arrays, each of length 1 + n*(q-1) for radius=1.
    """
    space_size = q ** n
    # Precompute powers of q
    powers = [1] * (n + 1)
    for i in range(1, n + 1):
        powers[i] = powers[i - 1] * q

    neighbor_lists = [None] * space_size

    for idx in range(space_size):
        # Get the word digits
        digits = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q

        nbrs = [idx]
        for pos in range(n):
            orig = digits[pos]
            p = powers[n - 1 - pos]
            for v in range(q):
                if v == orig:
                    continue
                nbrs.append(idx + (v - orig) * p)

        neighbor_lists[idx] = np.array(nbrs, dtype=np.int32)

    return neighbor_lists


def _greedy_cover_vectorized(q, n, radius, deadline):
    """
    Vectorized greedy set cover using numpy.
    Returns list of selected indices.
    """
    space_size = q ** n
    nbrs = _precompute_neighbor_indices(q, n, radius)

    # Precompute a flat array of all neighbor indices for fast indexing
    # nbr_flat: 2D array where row i contains neighbors of word i
    # All rows have same length for radius=1: 1 + n*(q-1)
    nbr_len = 1 + n * (q - 1)
    nbr_flat = np.empty((space_size, nbr_len), dtype=np.int32)
    for i in range(space_size):
        nbr_flat[i] = nbrs[i]

    covered = np.zeros(space_size, dtype=np.bool_)
    selected = []
    # uncovered_count[i] = number of uncovered words in nbrs[i]
    uncovered_count = np.full(space_size, nbr_len, dtype=np.int32)

    while time.time() < deadline - 5:
        # Find candidate with max uncovered_count
        best_idx = int(np.argmax(uncovered_count))
        best_count = uncovered_count[best_idx]

        if best_count == 0:
            break

        selected.append(best_idx)
        covered[nbr_flat[best_idx]] = True

        # Update uncovered_count for all candidates
        # For each candidate j, count how many of its neighbors are now uncovered
        # Vectorized: for each j, uncovered_count[j] = sum(~covered[nbr_flat[j]])
        # This is O(space_size * nbr_len) per iteration, which is 6561 * 25 = 164k ops
        # With numpy this is fast.

        # More efficient: only update candidates whose neighbors include any newly covered word
        # But the full recompute is simpler and still fast enough
        # Let's use a batch approach:
        # uncovered_count = sum over j of (~covered[nbr_flat[j]]).sum(axis=1)

        # Actually, let's be smarter: only decrement for candidates that cover newly-covered words
        # The newly covered words are nbr_flat[best_idx]
        # For each such word w, all candidates j where w in nbrs[j] get their count decremented
        # We can precompute the reverse: for each word w, which candidates cover it
        # But that's expensive to build. Let's just do the full recompute.

        # Full recompute using numpy:
        # For each candidate j, uncovered_count[j] = number of k in nbr_flat[j] where not covered[k]
        # We can do this as: uncovered_count = (~covered[nbr_flat]).sum(axis=1)
        # This creates a (space_size, nbr_len) boolean array and sums along axis 1.
        # 6561 * 25 = 164k booleans, summing 6561 values. Very fast.

        uncovered_count = (~covered[nbr_flat]).sum(axis=1).astype(np.int32)

    return selected


def _prune_reference_count(q, n, radius, selected, deadline):
    """
    Prune redundant codewords using reference counting.
    A codeword is removable if all words it covers are covered by at least one other codeword.
    """
    if not selected:
        return selected

    space_size = q ** n
    nbrs = _precompute_neighbor_indices(q, n, radius)

    # Build cover sets for selected codewords
    cover_sets = [nbrs[i] for i in selected]

    # Reference count: how many codewords cover each word
    ref_count = np.zeros(space_size, dtype=np.int32)
    for cs in cover_sets:
        ref_count[cs] += 1

    # A codeword i is removable if for all w in cover_sets[i], ref_count[w] > 1
    keep = [True] * len(selected)

    for i in range(len(selected)):
        if time.time() > deadline - 3:
            break
        # Check if any word in cover_sets[i] has ref_count == 1
        cs = cover_sets[i]
        if np.any(ref_count[cs] == 1):
            keep[i] = True  # Not removable
        else:
            keep[i] = False  # Removable
            # Decrement ref_count for words covered by codeword i
            ref_count[cs] -= 1

    return [selected[i] for i in range(len(selected)) if keep[i]]


def _verify_coverage(q, n, radius, selected):
    """Verify that all words are covered by the selected codewords."""
    space_size = q ** n
    nbrs = _precompute_neighbor_indices(q, n, radius)
    covered = np.zeros(space_size, dtype=np.bool_)
    for idx in selected:
        covered[nbrs[idx]] = True
    return np.all(covered)


def _idx_to_word(idx, q, n):
    """Convert integer index to word list."""
    word = [0] * n
    temp = idx
    for i in range(n - 1, -1, -1):
        word[i] = temp % q
        temp //= q
    return word


def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    deadline = _start_time + 110.0

    # Fallback for non-standard parameters
    if q != 3 or n != 8 or radius != 1:
        tc = _tetracode_solution(q, n, radius)
        if tc:
            return tc
        # Generic fallback: return tetracode-like or empty
        return []

    # 1. Greedy cover (vectorized)
    selected = _greedy_cover_vectorized(q, n, radius, deadline)

    # 2. Prune
    selected = _prune_reference_count(q, n, radius, selected, deadline)

    # 3. Verify coverage
    if not _verify_coverage(q, n, radius, selected):
        # Fallback to tetracode
        tc = _tetracode_solution(q, n, radius)
        if tc:
            return tc
        return []

    # Convert indices to words
    result = [_idx_to_word(idx, q, n) for idx in selected]
    return result

# EVOLVE-BLOCK-END
