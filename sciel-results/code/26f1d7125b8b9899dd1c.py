# EVOLVE-BLOCK-START
import time
import itertools

_T1 = time.time()
_HARD = _T1 + 150.0


def _deadline(margin):
    return (time.time() + margin) < _HARD


def _fallback_base(q, n, radius):
    """Direct-sum of tetracode: 9 * q^(n-4) words, radius-1 for n>=4, q=3."""
    if q != 3 or n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tail_len = n - 4
    out = []
    for tail in itertools.product(range(q), repeat=tail_len):
        for h in tetracode:
            out.append(h + list(tail))
    return out


def _build_neighbors(q, n):
    """Return list of neighbor tuples (self + single-coordinate flips) for every word."""
    all_words = list(itertools.product(range(q), repeat=n))
    word_idx = {w: i for i, w in enumerate(all_words)}
    nrm = []
    for w in all_words:
        nbrs = [word_idx[w]]
        for j in range(n):
            for d in (1, 2):
                b = list(w)
                b[j] = (b[j] + d) % q
                nbrs.append(word_idx[tuple(b)])
        nrm.append(nbrs)
    return all_words, word_idx, nrm


def _improve_remove_repair(base, q, n, radius):
    """Time-guarded remove-then-repair: shrink the code while keeping it a valid cover."""
    if radius < 1:
        return base
    N = q ** n
    all_words, word_idx, nrm = _build_neighbors(q, n)

    # map codewords to indices (dedupe)
    sel = []
    seen = set()
    for cw in base:
        i = word_idx[tuple(cw)]
        if i not in seen:
            seen.add(i)
            sel.append(i)

    # coverage count per word
    cov = [0] * N
    for ci in sel:
        for nb in nrm[ci]:
            cov[nb] += 1

    sel_set = set(sel)
    sel_pos = {ci: idx for idx, ci in enumerate(sel)}

    deadline_guard = _HARD - 8.0
    improved = True
    while improved and time.time() < deadline_guard:
        improved = False
        # find a removable codeword: after removal, at most 2 words become uncovered
        for ci in sel:
            # collect words that would drop to 0 if we remove ci
            uncover_list = []
            for nb in nrm[ci]:
                if cov[nb] == 1:
                    uncover_list.append(nb)
                    if len(uncover_list) > 2:
                        break
            if len(uncover_list) <= 2:
                # remove ci
                for nb in nrm[ci]:
                    cov[nb] -= 1
                sel.remove(ci)
                sel_set.discard(ci)
                improved = True
                # repair: for each uncovered word, add a codeword that covers it
                for uc in uncover_list:
                    if cov[uc] == 0:
                        # choose best repair: try uc itself first, then neighbors
                        best = uc
                        best_gain = 0
                        # score each candidate by how many currently-uncovered neighbors it covers
                        for cand in nrm[uc]:
                            gain = 0
                            for nb2 in nrm[cand]:
                                if cov[nb2] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best = cand
                        if best not in sel_set:
                            sel.append(best)
                            sel_set.add(best)
                            for nb in nrm[best]:
                                cov[nb] += 1
                if not _deadline(8.0):
                    break
    return [list(all_words[i]) for i in sel]


def construct(q, n, radius):
    global _HARD
    _HARD = time.time() + 150.0
    if n < 0:
        return []
    if n == 0:
        return [[]]

    # --- Build a guaranteed-valid base code ---
    base = _fallback_base(q, n, radius)
    # For q=3, n=8, radius=1: try to get a better initial code via greedy set cover
    if q == 3 and n == 8 and radius == 1 and _deadline(100.0):
        N = 3 ** n
        all_words, word_idx, nrm = _build_neighbors(q, n)
        # Greedy set cover: start empty, repeatedly add the word covering most uncovered
        uncovered = set(range(N))
        greedy_sel = []
        greedy_cov = [0] * N
        deadline_g = _HARD - 10.0
        while uncovered and time.time() < deadline_g:
            best_word = -1
            best_count = 0
            # Sample candidates: try all words but with early exit if found good
            for w in range(N):
                cnt = 0
                for nb in nrm[w]:
                    if nb in uncovered:
                        cnt += 1
                        if cnt <= best_count:
                            break
                if cnt > best_count:
                    best_count = cnt
                    best_word = w
                    if best_count == len(nrm[0]):
                        break
            if best_word < 0:
                break
            greedy_sel.append(best_word)
            for nb in nrm[best_word]:
                if nb in uncovered:
                    uncovered.discard(nb)
        if not uncovered and len(greedy_sel) < len(base):
            base = [list(all_words[i]) for i in greedy_sel]

    # --- Guarantee validity: cover all words (repair if base was not perfect) ---
    if radius >= 1:
        N = q ** n
        all_words, word_idx, nrm = _build_neighbors(q, n)
        cov = [0] * N
        sel = []
        seen = set()
        for cw in base:
            i = word_idx[tuple(cw)]
            if i not in seen:
                seen.add(i)
                sel.append(i)
                for nb in nrm[i]:
                    cov[nb] += 1
        uncovered = [i for i in range(N) if cov[i] == 0]
        # repair: add uncovered words as codewords
        while uncovered and _deadline(30.0):
            uc = uncovered.pop()
            if uc not in seen:
                seen.add(uc)
                sel.append(uc)
                for nb in nrm[uc]:
                    cov[nb] += 1
                # recompute remaining uncovered efficiently
                if uncovered:
                    uncovered = [i for i in uncovered if cov[i] == 0]

        base = [list(all_words[i]) for i in sel]

        # --- Optional improvement (time-guarded) ---
        if _deadline(60.0) and len(base) <= 20000:
            base = _improve_remove_repair(base, q, n, radius)

    return [list(c) for c in base]
# EVOLVE-BLOCK-END
