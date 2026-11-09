import time
import numpy as np
import itertools


def construct(q, n, radius):
    t0 = time.time()
    deadline = 118.0          # safety margin before the 150s kill
    total = q ** n
    ball_size = 1 + n * (q - 1)   # radius-1 ball size for q-ary

    # digits of every word in {0..q-1}^n, shape (total, n)
    all_words = np.array(
        [list(t) for t in itertools.product(range(q), repeat=n)], dtype=np.int8
    )

    def ball_mask(w):
        """Boolean mask over all_words for words within Hamming distance <= radius of w."""
        diffs = np.abs(all_words - w)
        return (diffs != 0).sum(axis=1) <= radius

    # Strong seed: direct sum of the ternary tetracode (a perfect radius-1 code of
    # length 4) with every word on the remaining n-4 coordinates.
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        seed = np.array(
            [h + list(tail) for h in tetracode
             for tail in itertools.product(range(3), repeat=n - 4)], dtype=np.int8
        )
    else:
        seed = all_words[:1]

    # Candidate pool: all words (random order) + single/double-error variants of seed.
    idx = np.arange(total)
    rng = np.random.default_rng(20240517)
    rng.shuffle(idx)
    cand = all_words[idx]

    def variants(words, d):
        out = []
        for w in words:
            base = w.astype(np.int8)
            if d == 1:
                for pos in range(n):
                    for v in range(q):
                        if v != w[pos]:
                            x = base.copy(); x[pos] = v; out.append(x)
            else:
                for p1 in range(n):
                    for p2 in range(p1 + 1, n):
                        for v1 in range(q):
                            if v1 == w[p1]:
                                continue
                            for v2 in range(q):
                                if v2 == w[p2]:
                                    continue
                                x = base.copy(); x[p1] = v1; x[p2] = v2; out.append(x)
        return np.array(out, dtype=np.int8) if out else np.empty((0, n), dtype=np.int8)

    cand = np.vstack([cand, variants(seed, 1), variants(seed, 2)])
    order = np.arange(cand.shape[0])
    rng.shuffle(order)
    cand = cand[order]
    ncand = cand.shape[0]

    covered = np.zeros(total, dtype=bool)
    chosen = []
    n_covered = 0

    # Seed the cover to get a strong starting point.
    for w in seed:
        m = ball_mask(w)
        if m.any():
            new = m & ~covered
            if new.any():
                covered |= m
                chosen.append(w)
                n_covered = covered.sum()
    if n_covered >= total:
        return [list(map(int, c)) for c in chosen]

    # Greedy set cover.
    while n_covered < total:
        if time.time() > deadline:
            break
        best_gain = 0
        best_j = -1
        for j in range(ncand):
            m = ball_mask(cand[j])
            gain = np.count_nonzero(m & ~covered)
            if gain > best_gain:
                best_gain = gain
                best_j = j
                if best_gain == 0:
                    break
        if best_j < 0 or best_gain == 0:
            break
        covered |= ball_mask(cand[best_j])
        chosen.append(cand[best_j])
        n_covered = covered.sum()
        if n_covered >= total:
            break

    # Safety: if greedy left gaps (should not happen since pool covers everything),
    # fill them with arbitrary words so the result is always valid.
    if n_covered < total:
        for w in all_words:
            m = ball_mask(w)
            if np.any(m & ~covered):
                covered |= m
                chosen.append(w)
                n_covered = covered.sum()
                if n_covered >= total:
                    break

    return [list(map(int, c)) for c in chosen]
