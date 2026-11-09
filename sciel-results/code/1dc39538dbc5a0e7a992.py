"""Greedy + local search for ternary covering code K_3(7,1)."""
import time
import random


def _neighbors1(word, n, q):
    """All words at Hamming distance 0 or 1 from word."""
    result = [word]
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                w = word.copy()
                w[i] = v
                result.append(tuple(w))
    return result


def _neighbors2(word, n, q):
    """All words at Hamming distance 2 from word."""
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            for vi in range(q):
                if vi == word[i]:
                    continue
                for vj in range(q):
                    if vj == word[j]:
                        continue
                    w = list(word)
                    w[i] = vi
                    w[j] = vj
                    result.append(tuple(w))
    return result


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin within 150s limit

    # Precompute all words as tuples
    all_words = list(_generate_words(q, n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total_words = len(all_words)

    # Precompute neighbors (distance 0 and 1) and distance-2 neighbors
    neigh1 = [None] * total_words
    neigh2 = [None] * total_words
    for i, w in enumerate(all_words):
        neigh1[i] = _neighbors1(w, n, q)
        neigh2[i] = _neighbors2(w, n, q)

    # Convert to index-based lists for speed
    neigh1_idx = [None] * total_words
    neigh2_idx = [None] * total_words
    for i in range(total_words):
        neigh1_idx[i] = [word_to_idx[w] for w in neigh1[i]]
        neigh2_idx[i] = [word_to_idx[w] for w in neigh2[i]]

    # Greedy construction
    uncovered = set(range(total_words))
    code = []
    code_set = set()

    while uncovered:
        if time.time() > deadline - 30:
            break
        best_idx = None
        best_gain = -1
        for idx in uncovered:
            gain = sum(1 for nb in neigh1_idx[idx] if nb in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                if gain == len(uncovered):
                    break
        if best_idx is None:
            break
        code.append(best_idx)
        code_set.add(best_idx)
        for nb in neigh1_idx[best_idx]:
            uncovered.discard(nb)

    # If we ran out of time before covering everything, add remaining uncovered as codewords
    if uncovered and time.time() < deadline - 5:
        for idx in sorted(uncovered):
            code.append(idx)
            code_set.add(idx)
        uncovered = set()

    # Local search: deletion
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        new_code = []
        for i, idx in enumerate(code):
            # Check if removing idx still covers all words it covers
            # A word w is uniquely covered by idx if w in neigh1_idx[idx] and
            # no other codeword in code also covers w
            can_remove = True
            for nb in neigh1_idx[idx]:
                # Check if nb is covered by another codeword
                covered_by_other = False
                for j, other in enumerate(code):
                    if j == i:
                        continue
                    if nb in neigh1_idx[other]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    can_remove = False
                    break
            if can_remove:
                improved = True
            else:
                new_code.append(idx)
        code = new_code
        code_set = set(code)

    # Local search: exchange
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        new_code = list(code)
        for i, idx in enumerate(code):
            # Find words only covered by idx (among the current code)
            only_covered = set()
            for nb in neigh1_idx[idx]:
                covered_by_other = False
                for j, other in enumerate(code):
                    if j == i:
                        continue
                    if nb in neigh1_idx[other]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    only_covered.add(nb)
            if not only_covered:
                continue
            # Try to find a candidate that covers all of only_covered
            for cand in all_words:
                if cand in code_set:
                    continue
                cand_set = set(neigh1_idx[word_to_idx[cand]])
                if only_covered.issubset(cand_set):
                    # Check that replacing idx with cand doesn't uncover anything
                    # Words covered by idx but not by cand must be covered by others
                    for nb in neigh1_idx[idx]:
                        if nb not in cand_set:
                            covered_by_other = False
                            for j, other in enumerate(code):
                                if j == i:
                                    continue
                                if nb in neigh1_idx[other]:
                                    covered_by_other = True
                                    break
                            if not covered_by_other:
                                break
                    else:
                        new_code[i] = word_to_idx[cand]
                        code_set.discard(idx)
                        code_set.add(word_to_idx[cand])
                        improved = True
                        break
        code = new_code
        code_set = set(code)

    # Convert to list of lists
    return [list(all_words[idx]) for idx in code]


def _generate_words(q, n):
    """Generate all q^n words as tuples."""
    if n == 0:
        yield ()
        return
    for first in range(q):
        for rest in _generate_words(q, n - 1):
            yield (first,) + rest
