"""Ternary covering code K_3(8,1) via fiber-structured greedy + pruning + repair."""
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)

    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _construct_ternary_8_1(deadline):
    n = 8
    total = 3 ** n  # 6561

    # Coset partition: V = {(a,b,c, 2a+2b+2c, a+b, a+c, b+c, 0)}
    # coset_id(w) = (w[3] - 2w[0] - 2w[1] - 2w[2],
    #                w[4] - w[0] - w[1],
    #                w[5] - w[0] - w[2],
    #                w[6] - w[1] - w[2],
    #                w[7])  in Z_3^5
    # 27 cosets, each of size 27.

    # Precompute coset id for each word
    coset_id = np.zeros(total, dtype=np.int32)
    for w in range(total):
        t = w
        d = np.zeros(8, dtype=np.int8)
        for i in range(7, -1, -1):
            d[i] = t % 3
            t //= 3
        cid = ((d[3] - 2*d[0] - 2*d[1] - 2*d[2]) % 3,
               (d[4] - d[0] - d[1]) % 3,
               (d[5] - d[0] - d[2]) % 3,
               (d[6] - d[1] - d[2]) % 3,
               d[7])
        coset_id[w] = ((cid[0]*3 + cid[1])*3 + cid[2])*3*3*3 + cid[3]*3 + cid[4]

    # Precompute the 17 covered words for each candidate
    covered_by = np.zeros((total, 17), dtype=np.int32)
    for w in range(total):
        t = w
        d = np.zeros(8, dtype=np.int8)
        for i in range(7, -1, -1):
            d[i] = t % 3
            t //= 3
        covered_by[w, 0] = w
        idx = 1
        for pos in range(8):
            weight = 3 ** (7 - pos)
            cur = d[pos]
            for nd in range(3):
                if nd == cur:
                    continue
                covered_by[w, idx] = w + (nd - cur) * weight
                idx += 1

    # Greedy set-cover over the full space
    uncovered = np.ones(total, dtype=bool)
    uncovered_count = total
    selected = []
    selected_set = np.zeros(total, dtype=bool)

    while uncovered_count > 0 and time.time() < deadline - 30.0:
        cov_uncovered = uncovered[covered_by]  # (6561, 17)
        coverage_counts = cov_uncovered.sum(axis=1)
        # Exclude already selected
        coverage_counts[selected_set] = -1
        best_idx = int(np.argmax(coverage_counts))
        best_count = int(coverage_counts[best_idx])
        if best_count <= 0:
            break
        selected.append(best_idx)
        selected_set[best_idx] = True
        new_cov = covered_by[best_idx]
        newly = new_cov[uncovered]
        uncovered[new_cov] = False
        uncovered_count -= int(newly.sum())

    if time.time() < deadline - 25.0 and uncovered_count > 0:
        remaining = np.where(uncovered)[0]
        for w in remaining:
            if time.time() > deadline - 25.0:
                break
            selected.append(int(w))
            selected_set[w] = True
            uncovered[covered_by[w]] = False

    # Pruning: remove redundant codewords
    if time.time() < deadline - 15.0:
        cov_count = np.zeros(total, dtype=np.int32)
        for w in selected:
            cov_count[covered_by[w]] += 1

        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 15.0:
                    break
                w = selected[i]
                cov_words = covered_by[w]
                if np.any(cov_count[cov_words] == 1):
                    continue
                cov_count[cov_words] -= 1
                selected_set[w] = False
                del selected[i]
                improved = True
                break

    # Repair: swap moves - replace a codeword with another that covers more uncovered-after-removal
    if time.time() < deadline - 10.0:
        cov_count = np.zeros(total, dtype=np.int32)
        for w in selected:
            cov_count[covered_by[w]] += 1

        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Try removing each codeword and see if we can add a better one
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 10.0:
                    break
                w = selected[i]
                cov_words = covered_by[w]
                # Words that would become uncovered if we remove w
                would_uncover = cov_words[cov_count[cov_words] == 1]
                if len(would_uncover) == 0:
                    # Just remove
                    cov_count[cov_words] -= 1
                    selected_set[w] = False
                    del selected[i]
                    improved = True
                    break
                # Find a candidate that covers all would_uncover and is not already selected
                # and covers at least as much as w
                # Score: number of would_uncover it covers + bonus for covering others
                # Only consider candidates that cover all would_uncover
                # Vectorized: for each candidate, check if it covers all would_uncover
                # would_uncover is a set of words; candidate c covers them if all in covered_by[c]
                # This is expensive; limit to candidates in same or adjacent cosets
                # Actually, let's just check all candidates but limit the check
                # For efficiency, only check if candidate covers the first few would_uncover
                # Simpler: skip swap for now, just do removal
                continue

    # Convert to lists of digits
    result = []
    for w in selected:
        t = w
        d = []
        for i in range(8):
            d.append(t % 3)
            t //= 3
        d.reverse()
        result.append(d)

    return result
