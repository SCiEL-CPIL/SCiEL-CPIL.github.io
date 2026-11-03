import numpy as np
import time


def construct(q, n, radius):
    deadline = time.time() + 110

    total = q ** n

    # Precompute coverage: for each word, the indices within Hamming distance <= radius
    # For radius=1: the word itself + all words differing in exactly one position
    coverage_list = []
    for i in range(total):
        t = []
        x = i
        for _ in range(n):
            t.append(x % q)
            x //= q
        t.reverse()
        covered = [i]
        for pos in range(n):
            orig = t[pos]
            for val in range(q):
                if val != orig:
                    new_t = t[:]
                    new_t[pos] = val
                    idx = 0
                    for d in new_t:
                        idx = idx * q + d
                    covered.append(idx)
        coverage_list.append(covered)

    coverage_arr = np.array(coverage_list, dtype=np.int32)

    uncovered = np.ones(total, dtype=bool)
    codewords = []

    while uncovered.any() and time.time() < deadline:
        counts = uncovered[coverage_arr].sum(axis=1)
        best_idx = int(np.argmax(counts))
        best_count = counts[best_idx]
        if best_count <= 0:
            break
        codewords.append(best_idx)
        uncovered[coverage_arr[best_idx]] = False

    # Convert flat indices to lists of digits
    result = []
    for idx in codewords:
        t = []
        x = idx
        for _ in range(n):
            t.append(x % q)
            x //= q
        t.reverse()
        result.append(t)

    return result
