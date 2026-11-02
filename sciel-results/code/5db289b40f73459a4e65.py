"""Ternary covering code K_3(8,1): greedy set-cover + 1/2-exchange local search.

Every returned solution is verified to cover the whole space, so the result is
always valid.  A wall-clock deadline guard returns the best valid solution found.
"""
import time
import random
import numpy as np

_DEADLINE = 145.0  # safety margin under the 150s wall-clock kill
_EPS = 1e-12


def _all_words(q, n):
    # row i = base-q digits of i (most significant first)
    return np.array([list(divmod(i, q ** (n - k))[0] for k in range(n))
                     for i in range(q ** n)], dtype=np.int32)


def _expand(words, q, n):
    """Bitset of all words within Hamming distance 1 of any row of `words`."""
    m = words.shape[0]
    base = np.zeros(m * (n + 1), dtype=np.int32)
    base[:m] = words
    for j in range(n):
        block = base[m * (j + 1): m * (j + 2)]
        block[:] = words
        block[:, j] = (words[:, j] + 1) % q
    flat = base.reshape(-1).astype(np.int64)
    return flat


def _cover(words, q, n, all_words, size):
    """Return boolean array: all_words[i] covered by some word in `words`."""
    if words.size == 0:
        return np.zeros(size, dtype=bool)
    return np.isin(all_words.ravel(), _expand(words, q, n))


def construct(q, n, radius):
    t0 = time.time()
    rng = random.Random(20240601)
    size = q ** n
    all_words = _all_words(q, n)

    # ---- Greedy set-cover seed (max coverage, then max incremental) ----------
    # Precompute each candidate's coverage bitset (one per word).
    per = [np.isin(all_words.ravel(), _expand(np.array([w], dtype=np.int32), q, n))
           for w in all_words]
    uncovered = np.ones(size, dtype=bool)
    chosen = []
    for _ in range(size):
        if time.time() - t0 > _DEADLINE:
            break
        # pick the candidate with the most newly-covered words
        gains = np.array([int(np.count_nonzero(per[i] & uncovered)) for i in range(size)])
        i = int(np.argmax(gains))
        if gains[i] <= 0:
            break
        uncovered &= ~per[i]
        chosen.append(all_words[i])
        if not uncovered.any():
            break

    words = np.array(chosen, dtype=np.int32) if chosen else np.zeros((0, n), dtype=np.int32)
    cov = _cover(words, q, n, all_words, size)
    if not cov.all():
        # safety: complete coverage with a full exhaustive sweep (guaranteed valid)
        for w in all_words:
            if time.time() - t0 > _DEADLINE:
                break
            if not cov.all():
                words = np.vstack([words, w]) if words.size else w.reshape(1, n)
                cov = _cover(words, q, n, all_words, size)
    if not cov.all():
        # absolute fallback: return the whole space (always valid)
        return [list(map(int, w)) for w in all_words]

    best = words.copy()
    best_cov = cov.copy()

    def still_valid_after_delta(cov, rem_idx, add_words):
        # recompute coverage over the region touched by removed/added words
        rem = best[rem_idx]
        region = _expand(rem, q, n)
        if add_words.size:
            region = np.concatenate([region, _expand(add_words, q, n)])
        region = np.unique(region).astype(np.int64)
        # coverage of the region by (best minus removed) plus added
        base_words = np.delete(best, rem_idx, axis=0)
        base_cov = np.isin(region, _expand(base_words, q, n)) if base_words.size else np.zeros(len(region), dtype=bool)
        if add_words.size:
            base_cov |= np.isin(region, _expand(add_words, q, n))
        if not base_cov.all():
            return False
        # apply to global
        newcov = cov.copy()
        # words that were ONLY covered by removed words and not by add/base become uncovered
        # simpler: recompute full coverage for the region and merge
        newcov[region] = base_cov
        return newcov.all()

    # ---- 1-exchange local search ---------------------------------------------
    improved = True
    while improved:
        improved = False
        order = list(range(len(best)))
        rng.shuffle(order)
        for idx in order:
            if time.time() - t0 > _DEADLINE:
                return [list(map(int, w)) for w in best]
            rem = best[idx]
            base_words = np.delete(best, idx, axis=0)
            # region affected by removing `rem`
            region = np.unique(_expand(rem, q, n)).astype(np.int64)
            base_cov = np.isin(region, _expand(base_words, q, n)) if base_words.size else np.zeros(len(region), dtype=bool)
            # find a candidate word that covers all currently-uncovered-in-region
            need = ~base_cov
            if not need.any():
                # removing alone keeps coverage -> drop it
                best = base_words
                best_cov = _cover(best, q, n, all_words, size)
                improved = True
                continue
            # candidates = words whose coverage covers `need`
            # test incrementally: a word w is good if expand(w) covers region[need]
            need_words = region[need]
            # vectorize: for each candidate c, does expand(c) contain all need_words?
            cand = all_words
            ok = np.zeros(len(cand), dtype=bool)
            for nw in need_words:
                # expand(c) contains nw  <=>  nw within distance 1 of c
                d = (cand != nw).sum(axis=1)
                ok &= (d <= radius)
            idxs = np.where(ok)[0]
            if idxs.size == 0:
                continue
            pick = cand[int(idxs[rng.randrange(len(idxs))])]
            cand_words = np.vstack([base_words, pick])
            newcov = best_cov.copy()
            newcov[region] = np.isin(region, _expand(cand_words, q, n))
            if newcov.all():
                best = cand_words
                best_cov = newcov
                improved = True

    # ---- 2-exchange local search (escape local minima) -----------------------
    improved = True
    while improved:
        improved = False
        m = len(best)
        if m < 2:
            break
        for _ in range(200):
            if time.time() - t0 > _DEADLINE:
                return [list(map(int, w)) for w in best]
            i, j = rng.sample(range(m), 2)
            rem = best[[i, j]]
            base_words = np.delete(best, [i, j], axis=0)
            region = np.unique(_expand(rem, q, n)).astype(np.int64)
            base_cov = np.isin(region, _expand(base_words, q, n)) if base_words.size else np.zeros(len(region), dtype=bool)
            need = ~base_cov
            if not need.any():
                best = base_words
                best_cov = _cover(best, q, n, all_words, size)
                improved = True
                break
            need_words = region[need]
            # choose two candidate words together covering `need`
            cand = all_words
            # per-word coverage of need_words (bool matrix cand x need)
            covmat = np.zeros((len(cand), len(need_words)), dtype=bool)
            for t, nw in enumerate(need_words):
                d = (cand != nw).sum(axis=1)
                covmat[:, t] = (d <= radius)
            # find a pair (a,b) whose OR covers all need columns
            # restrict to words covering at least one need column
            active = np.where(covmat.any(axis=1))[0]
            if active.size < 2:
                continue
            found = False
            for a in active:
                if found:
                    break
                # need columns not covered by a
                rem_need = np.where(~covmat[a])[0]
                if rem_need.size == 0:
                    b = a  # single word suffices
                    pair = [a]
                else:
                    bmask = covmat[:, rem_need].all(axis=1)
                    bidx = np.where(bmask)[0]
                    if bidx.size == 0:
                        continue
                    b = int(bidx[rng.randrange(len(bidx))])
                    pair = [int(a), b]
                cand_words = np.vstack([base_words, cand[pair[0]], cand[pair[1]]]) if len(pair) == 2 \
                    else np.vstack([base_words, cand[pair[0]]])
                newcov = best_cov.copy()
                newcov[region] = np.isin(region, _expand(cand_words, q, n))
                if newcov.all():
                    best = cand_words
                    best_cov = newcov
                    improved = True
                    found = True
                    break

    return [list(map(int, w)) for w in best]
