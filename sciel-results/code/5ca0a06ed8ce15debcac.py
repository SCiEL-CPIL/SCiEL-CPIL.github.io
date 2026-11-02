"""Greedy set-covering for ternary covering codes K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np


def _hamming_neighborhood(word, q, n, radius):
    """Return all words within Hamming distance `radius` of `word`."""
    if radius == 0:
        return [tuple(word)]
    if radius == 1:
        nbrs = [tuple(word)]
        for pos in range(n):
            old = word[pos]
            for v in range(q):
                if v != old:
                    w = list(word)
                    w[pos] = v
                    nbrs.append(tuple(w))
        return nbrs
    # general radius via itertools.product of per-position options
    import itertools
    opts = []
    for pos in range(n):
        opts.append(list(range(q)))
    result = []
    for combo in itertools.product(*opts):
        d = sum(1 for a, b in zip(word, combo) if a != b)
        if d <= radius:
            result.append(tuple(combo))
    return result


def _enumerate_sphere(word, q, n, radius):
    """Efficiently enumerate all words within Hamming distance radius of word."""
    if radius == 0:
        return [tuple(word)]
    result = [tuple(word)]
    # radius 1
    for pos in range(n):
        old = word[pos]
        for v in range(q):
            if v != old:
                w = list(word)
                w[pos] = v
                result.append(tuple(w))
    # radius 2
    if radius >= 2:
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                old1 = word[pos1]
                old2 = word[pos2]
                for v1 in range(q):
                    if v1 == old1:
                        continue
                    for v2 in range(q):
                        if v2 == old2:
                            continue
                        w = list(word)
                        w[pos1] = v1
                        w[pos2] = v2
                        result.append(tuple(w))
    return result


def construct(q, n, radius):
    total = q ** n
    # Map each word to an integer index
    def to_idx(word):
        idx = 0
        mul = 1
        for i in range(n):
            idx += word[i] * mul
            mul *= q
        return idx

    def from_idx(idx):
        word = [0] * n
        mul = 1
        for i in range(n):
            word[i] = idx % q
            idx //= q
            mul *= q
        return word

    # Precompute neighborhoods for all words (for radius 1, each has 1 + n*(q-1) neighbors)
    # To save memory/time, compute on the fly but cache
    all_words = list(range(total))
    
    # Greedy: repeatedly pick the uncovered word whose neighborhood covers the most uncovered words
    uncovered = set(all_words)
    code = []
    
    # Precompute neighborhood for each word lazily
    nbr_cache = {}
    
    def get_nbrs(idx):
        if idx not in nbr_cache:
            word = from_idx(idx)
            nbrs = _enumerate_sphere(word, q, n, radius)
            nbr_cache[idx] = [to_idx(nb) for nb in nbrs]
        return nbr_cache[idx]
    
    while uncovered:
        # Pick the word in uncovered whose neighborhood covers the most uncovered words
        best_idx = None
        best_count = -1
        # Sample if too large, but for 7^7=823543 it's fine to scan a subset
        # To speed up, iterate over uncovered
        for idx in list(uncovered):
            nbrs = get_nbrs(idx)
            count = sum(1 for nb in nbrs if nb in uncovered)
            if count > best_count:
                best_count = count
                best_idx = idx
                if best_count == len(uncovered):
                    break
        code.append(from_idx(best_idx))
        # Remove all covered words
        nbrs = get_nbrs(best_idx)
        for nb in nbrs:
            uncovered.discard(nb)
    
    return code
# EVOLVE-BLOCK-END
