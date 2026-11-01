"""Fresh design: greedy set-cover init + vectorized critical-word swap local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _all_words(q, n):
    # column-major enumeration: word index -> list of digits
    return np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8).T  # shape (n, q**n)


def _hamming_to_all(cw, allwords, q, n):
    # cw: (n, m) int8 ; returns (m, q**n) int16 distances
    # distance = number of positions where digit differs
    diff = (cw[:, :, None] != allwords[None, :, :]).sum(axis=0)  # (m, q**n)
    return diff


def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = 125.0  # safety margin under 150s kill
    N = q ** n

    # Precompute the full space as (n, N)
    allwords = _all_words(q, n)

    # ---------- 1) Greedy set-cover initialization ----------
    # covered: boolean (N,)
    covered = np.zeros(N, dtype=bool)
    code = []  # list of codeword tuples (length n)

    def covered_by_one(pos, val):
        # mask of words with digit val at position pos
        return allwords[pos] == val

    # Precompute per-position value masks for speed: masks[pos][val]
    masks = []
    for pos in range(n):
        mrow = np.array([allwords[pos] == v for v in range(q)], dtype=bool)  # (q, N)
        masks.append(mrow)

    while not covered.all():
        if time.time() > DEADLINE:
            break
        best_gain = -1
        best_word = None
        # candidate words: try all words that are not yet covered (they cover themselves)
        # To keep it fast, sample candidate set: all uncovered words + a fixed pool
        unc = np.where(~covered)[0]
        if len(unc) == 0:
            break
        # Build candidate list: uncovered words (each covers at least itself)
        cands = unc
        if len(cands) > 4000:
            # subsample for speed in greedy phase
            idx = np.random.choice(len(cands), size=4000, replace=False)
            cands = cands[idx]
        for w in cands:
            word = allwords[:, w]
            # ball of radius `radius` around word: words differing in <= radius positions
            # compute distance from word to all words
            dist = (word[:, None] != allwords).sum(axis=0)  # (N,)
            ball = dist <= radius
            gain = (~covered & ball).sum()
            if gain > best_gain:
                best_gain = gain
                best_word = word
        if best_word is None:
            break
        code.append(best_word.copy())
        dist = (best_word[:, None] != allwords).sum(axis=0)
        covered |= (dist <= radius)

    # Ensure we actually have a full cover (greedy may have stopped early on deadline)
    # If not fully covered, keep adding until covered or deadline
    while not covered.all() and time.time() < DEADLINE:
        unc = np.where(~covered)[0]
        for w in unc:
            word = allwords[:, w]
            dist = (word[:, None] != allwords).sum(axis=0)
            covered |= (dist <= radius)
            code.append(word.copy())
            if covered.all():
                break
        if covered.all():
            break

    # ---------- 2) Vectorized local search: swap + drop + add ----------
    # Represent code as (n, m) int8
    def to_matrix(lst):
        if len(lst) == 0:
            return np.zeros((n, 0), dtype=np.int8)
        return np.array(lst, dtype=np.int8).T  # (n, m)

    C = to_matrix(code)
    m = C.shape[1]

    # current coverage count per word: count of codewords within radius
    def coverage_counts(Cmat):
        # (N,) int
        if Cmat.shape[1] == 0:
            return np.zeros(N, dtype=np.int32)
        dist = (Cmat[:, :, None] != allwords[None, :, :]).sum(axis=0)  # (m, N)
        within = dist <= radius
        return within.sum(axis=0)

    cov = coverage_counts(C)
    # A word is "critical" if covered exactly once; the code is valid iff no word has cov==0.
    # For improvement we try to keep cov>=1 everywhere while reducing m.

    def is_valid(cov):
        return cov.min() >= 1

    best_m = m
    best_C = C.copy()

    # Precompute distance matrix helper
    def dist_matrix(Cmat):
        if Cmat.shape[1] == 0:
            return np.zeros((0, N), dtype=np.int16)
        return (Cmat[:, :, None] != allwords[None, :, :]).sum(axis=0)

    # ---- Local search loop ----
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        D = dist_matrix(C)  # (m, N)
        # Try SWAP: for each codeword i, find a replacement word w (not in code)
        # that keeps validity and ideally increases "slack" (words covered >=2).
        # A swap is valid if after removing i and adding w, no word drops to 0.
        # Removing i leaves word x uncovered iff cov[x]==1 and D[i,x]<=radius.
        # Adding w covers ball(w). So valid if: for all x with cov[x]==1 and D[i,x]<=radius,
        #   w must cover x (D_w[x]<=radius).
        # To reduce m we'd need add to be redundant, but swap keeps m same;
        # swap helps enable later drops.

        # Try DROP first: remove codeword i if cov after removal >=1 everywhere.
        # Removing i is safe iff no word has cov==1 and D[i,x]<=radius.
        for i in range(m):
            if time.time() > DEADLINE:
                break
            if cov.min() >= 2:
                # any single removal safe
                C = np.delete(C, i, axis=1)
                m -= 1
                cov = coverage_counts(C)
                improved = True
                break
            else:
                # find words with cov==1
                single = np.where(cov == 1)[0]
                # among those, which are covered by i?
                covered_by_i = D[i, single] <= radius
                if not covered_by_i.any():
                    # i is redundant
                    C = np.delete(C, i, axis=1)
                    m -= 1
                    cov = coverage_counts(C)
                    improved = True
                    break

        # After a drop, m changed; recompute best
        if m < best_m and is_valid(cov):
            best_m = m
            best_C = C.copy()

        if improved:
            continue

        # Try SWAP to create redundancy (enable future drops)
        D = dist_matrix(C)
        for i in range(m):
            if time.time() > DEADLINE:
                break
            single = np.where(cov == 1)[0]
            must_cover = single[D[i, single] <= radius]  # words that would become uncovered
            if len(must_cover) == 0:
                # i redundant (should have been dropped)
                C = np.delete(C, i, axis=1)
                m -= 1
                cov = coverage_counts(C)
                improved = True
                if m < best_m and is_valid(cov):
                    best_m = m
                    best_C = C.copy()
                break
            # Find candidate w that covers all must_cover and is not in code
            # candidate words: pick from must_cover themselves (they cover themselves)
            cand_set = set(must_cover.tolist())
            # also add neighbors to expand options
            found = False
            for w in cand_set:
                if time.time() > DEADLINE:
                    break
                word = allwords[:, w]
                # check w not equal to any existing codeword
                eq = (word[:, None] == C).all(axis=0)
                if eq.any():
                    continue
                # check w covers all must_cover
                distw = (word[:, None] != allwords[must_cover]).sum(axis=0)
                if distw.max() <= radius:
                    # valid swap
                    C[:, i] = word
                    cov = coverage_counts(C)
                    improved = True
                    if m < best_m and is_valid(cov):
                        best_m = m
                        best_C = C.copy()
                    found = True
                    break
            if found:
                break

    # Return best valid code found
    final_C = best_C if is_valid(coverage_counts(best_C)) else C
    # Fallback: if best not valid, use current C (should be valid)
    out = []
    for j in range(final_C.shape[1]):
        out.append(final_C[:, j].tolist())
    # Ensure distinct
    seen = set()
    dedup = []
    for w in out:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            dedup.append(w)
    return dedup
# EVOLVE-BLOCK-END
