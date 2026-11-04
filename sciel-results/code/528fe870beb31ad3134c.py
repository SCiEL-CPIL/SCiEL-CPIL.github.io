"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill


def _word_to_idx(word, q, n):
    """Convert a word (tuple or list) to an integer index in 0..q^n-1."""
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _idx_to_word(idx, q, n):
    """Convert an integer index to a word (list)."""
    word = []
    for _ in range(n):
        word.append(idx % q)
        idx //= q
    word.reverse()
    return word


def _neighbors_indices(cw, q, n):
    """Return boolean numpy array of length q**n indicating words covered by cw within radius 1."""
    space_size = q ** n
    idx = np.zeros(space_size, dtype=bool)
    base = np.array(cw, dtype=np.int64)
    idx[int(np.ravel_multi_index(tuple(base), [q] * n))] = True
    for pos in range(n):
        for d in range(q):
            if d == base[pos]:
                continue
            cand = base.copy()
            cand[pos] = d
            idx[int(np.ravel_multi_index(tuple(cand), [q] * n))] = True
    return idx


def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode (9 words len 4, perfect radius 1) with full tails."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _greedy_cover(q, n, radius, seed_words=None, time_limit=120.0):
    """Greedy set cover: repeatedly pick the codeword covering the most uncovered words."""
    space_size = q ** n
    all_indices = np.arange(space_size, dtype=np.int64)
    
    # Precompute index for each word
    idx_of = {}
    for i, w in enumerate(itertools.product(range(q), repeat=n)):
        idx_of[w] = i

    # Start with all uncovered
    uncovered = np.ones(space_size, dtype=bool)

    # Precompute coverage vectors for all words
    coverings = []
    for i in range(space_size):
        w = [int(d) for d in np.unravel_index(i, [q] * n)]
        vec = _neighbors_indices(w, q, n)
        coverings.append(vec)
        if time.time() - _start_time > time_limit - 10:
            break

    if len(coverings) < space_size:
        # Fallback: just use tetracode
        return _tetracode_solution(q, n, radius)

    # Convert to numpy array for fast coverage counting
    cov_matrix = np.array(coverings, dtype=bool)  # shape (space_size, space_size)

    # If we have seed words, mark them as initially "covered by chosen"
    chosen_indices = []
    if seed_words is not None:
        for cw in seed_words:
            ci = idx_of[tuple(cw)]
            if ci < space_size:
                chosen_indices.append(ci)
                uncovered[cov_matrix[ci]] = False
        # Remove chosen from candidate pool conceptually (we won't pick them again)

    selected = list(chosen_indices)
    remaining_count = space_size - int(np.sum(~uncovered))

    while remaining_count > 0:
        # Find candidate covering most uncovered
        gains = np.sum(cov_matrix & uncovered[np.newaxis, :], axis=1)
        # Exclude already selected
        if selected:
            gains[np.array(selected, dtype=np.int64)] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        selected.append(best)
        uncovered[cov_matrix[best]] = False
        remaining_count -= int(gains[best])
        if time.time() - _start_time > time_limit - 5:
            break

    # Convert selected indices back to words
    result = []
    for i in selected:
        w = [int(d) for d in np.unravel_index(i, [q] * n)]
        result.append(w)
    return result


def _greedy_prune(q, n, radius, words, time_limit=120.0):
    """Given an initial set of words, try to remove words whose coverage is redundant."""
    space_size = q ** n
    idx_of = {}
    for i, w in enumerate(itertools.product(range(q), repeat=n)):
        idx_of[w] = i

    covs = []
    for cw in words:
        ci = idx_of[tuple(cw)]
        covs.append(_neighbors_indices(cw, q, n))
        if time.time() - _start_time > time_limit - 10:
            break

    if len(covs) < len(words):
        return words

    cov_arr = np.array(covs, dtype=bool)
    n_words = len(words)

    # Try removing each word in a greedy pass
    keep = [True] * n_words
    order = list(range(n_words))
    # Sort by coverage size (smaller coverage first are easier to remove)
    cov_sizes = np.sum(cov_arr, axis=1)
    order.sort(key=lambda i: cov_sizes[i])

    for ci in order:
        if not keep[ci]:
            continue
        if time.time() - _start_time > time_limit - 5:
            break
        # Check if cov_arr[ci] is covered by union of other kept codewords
        # Build union of others
        others = np.zeros(space_size, dtype=bool)
        for j in range(n_words):
            if j == ci or not keep[j]:
                continue
            others |= cov_arr[j]
        if np.all(others[cov_arr[ci]]):
            keep[ci] = False

    result = [words[i] for i in range(n_words) if keep[i]]
    return result


def _iterative_prune(q, n, radius, words, time_limit=120.0):
    """Iteratively prune until no more words can be removed."""
    current_words = words
    while True:
        if time.time() - _start_time > time_limit - 5:
            break
        pruned = _greedy_prune(q, n, radius, current_words, time_limit)
        if len(pruned) >= len(current_words):
            break
        current_words = pruned
    return current_words


def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    time_limit = _DEADLINE - 5  # 5s safety margin

    # If parameters don't match the ternary length-8 radius-1 case, fall back to tetracode
    if q != 3 or n != 8 or radius != 1:
        return _tetracode_solution(q, n, radius)

    # Start with the known tetracode-based solution (729 words)
    seed_words = _tetracode_solution(q, n, radius)

    # Try greedy pruning of the seed solution
    pruned = _iterative_prune(q, n, radius, seed_words, time_limit)

    # If time remains, try a greedy-from-scratch which may find a smaller code
    elapsed = time.time() - _start_time
    if elapsed < time_limit - 30 and len(pruned) > 500:
        greedy_fresh = _greedy_cover(q, n, radius, seed_words=None, time_limit=time_limit)
        if len(greedy_fresh) < len(pruned):
            pruned = greedy_fresh
        # Try to prune the greedy result as well
        pruned = _iterative_prune(q, n, radius, pruned, time_limit)

    # Verify coverage (sanity check)
    space_size = q ** n
    covered = np.zeros(space_size, dtype=bool)
    for cw in pruned:
        covered |= _neighbors_indices(cw, q, n)
    if not np.all(covered):
        # Fallback to seed
        return seed_words

    # Return best
    return pruned
# EVOLVE-BLOCK-END
