# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(q, n, radius):
    """Greedy covering-code construction with vectorized coverage and local improvement.

    Returns a list of codewords (each a list of n ints in 0..q-1) that covers every
    word of {0..q-1}^n within Hamming distance `radius`. Minimizes the count.
    """
    deadline = time.time() + 105.0   # safety margin well under the 150s kill
    total = q ** n

    # ---- fast path: perfect code (radius 1, q = p+1, n = p^2) ----
    # q-ary Hamming code of length (q^2-1)/(q-1) = q+1, size q^2, perfect.
    if radius == 1:
        p = q - 1
        if n == q + 1:
            code = [[(a + i * b) % q for i in range(n)] for a in range(q) for b in range(q)]
            return code

    # ---- small instances: exhaustive greedy (exact) ----
    if total <= 2000:
        all_words = [np.array(w, dtype=np.int32) for w in itertools.product(range(q), repeat=n)]
        cov = [0] * total
        chosen = []
        for _ in range(total):
            best, best_gain = -1, -1
            for i, w in enumerate(all_words):
                if cov[i] > 0:
                    continue
                gain = 0
                for j in range(total):
                    if cov[j] == 0 and np.count_nonzero(w != all_words[j]) <= radius:
                        gain += 1
                if gain > best_gain:
                    best_gain, best = gain, i
            if best < 0:
                break
            chosen.append(all_words[best].tolist())
            for j in range(total):
                if cov[j] == 0 and np.count_nonzero(all_words[best] != all_words[j]) <= radius:
                    cov[j] += 1
        return chosen

    # ---- general: vectorized greedy set cover ----
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)

    def ball_mask(center):
        # boolean mask of all words within Hamming distance `radius` of `center`
        if radius == 0:
            return all_words == center
        d = (all_words != center).sum(axis=1)
        return d <= radius

    # deterministic strong seed: q-ary Hamming code (if applicable) + random extras
    seed = []
    if radius == 1 and n == q + 1:
        for a in range(q):
            for b in range(q):
                seed.append([(a + i * b) % q for i in range(n)])
    rng = np.random.default_rng(0)
    while len(seed) < 80:
        seed.append(rng.integers(0, q, size=n).tolist())

    cov = np.zeros(total, dtype=np.int32)
    chosen = list(seed)
    for cw in chosen:
        cov[ball_mask(np.array(cw, dtype=np.int32))] += 1

    # greedy: add the word (uncovered or not) that covers the most uncovered words
    while cov.min() > 0 or True:
        if cov.min() > 0:
            break
        if time.time() > deadline - 3.0:
            break
        best, best_gain = -1, 0
        for i in range(total):
            if cov[i] > 0:
                continue
            g = int(ball_mask(all_words[i]).sum())
            if g > best_gain:
                best_gain, best = g, i
        if best < 0:
            break
        chosen.append(all_words[best].tolist())
        cov[ball_mask(all_words[best])] += 1

    # local improvement: drop redundant codewords, then try swaps (drop one, add one)
    def is_redundant(idx):
        # true if removing chosen[idx] leaves all words covered
        m = ball_mask(np.array(chosen[idx], dtype=np.int32))
        return bool((cov[m] > 1).all())

    improved = True
    while improved and time.time() < deadline - 6.0:
        improved = False
        # pass 1: drop redundant codewords
        for idx in range(len(chosen) - 1, -1, -1):
            if time.time() > deadline - 6.0:
                break
            if is_redundant(idx):
                cov[ball_mask(np.array(chosen[idx], dtype=np.int32))] -= 1
                chosen.pop(idx)
                improved = True
        # pass 2: try swaps — drop a codeword, then add the best single replacement
        for idx in range(len(chosen)):
            if time.time() > deadline - 6.0:
                break
            if not is_redundant(idx):
                continue
            # temporarily remove
            cov[ball_mask(np.array(chosen[idx], dtype=np.int32))] -= 1
            # find best single add to restore coverage
            best, best_gain = -1, 0
            for i in range(total):
                if cov[i] > 0:
                    continue
                g = int(ball_mask(all_words[i]).sum())
                if g > best_gain:
                    best_gain, best = g, i
            if best >= 0 and best_gain >= int(ball_mask(np.array(chosen[idx], dtype=np.int32)).sum()):
                # replacement covers at least as many; accept (reduces or keeps size)
                cov[ball_mask(all_words[best])] += 1
                chosen[idx] = all_words[best].tolist()
                improved = True
            else:
                # restore
                cov[ball_mask(np.array(chosen[idx], dtype=np.int32))] += 1

    # final safety: ensure full coverage
    if cov.min() == 0:
        while cov.min() == 0:
            best, best_gain = -1, 0
            for i in range(total):
                if cov[i] > 0:
                    continue
                g = int(ball_mask(all_words[i]).sum())
                if g > best_gain:
                    best_gain, best = g, i
            if best < 0:
                break
            chosen.append(all_words[best].tolist())
            cov[ball_mask(all_words[best])] += 1

    return [list(map(int, cw)) for cw in chosen]
# EVOLVE-BLOCK-END
