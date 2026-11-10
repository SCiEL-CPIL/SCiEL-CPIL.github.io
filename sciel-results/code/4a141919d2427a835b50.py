"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _neighbors_all(q, n, radius):
    """Return (neighbors, word_to_int) where neighbors is a list of lists of ints.

    neighbors[i] is the list of integer-encoded words within Hamming distance
    <= radius of the word encoded by i.  word_to_int maps a tuple of symbols
    to its integer encoding.
    """
    total = q ** n
    # Precompute per-position digit and powers.
    powers = [1] * n
    for pos in range(n - 2, -1, -1):
        powers[pos] = powers[pos + 1] * q
    # digit[word, pos]
    digits = np.empty((total, n), dtype=np.int32)
    for pos in range(n):
        p = powers[pos]
        digits[:, pos] = (np.arange(total) // p) % q
    # Precompute the "change digit at pos to v" delta.
    # For a given word w and pos, changing symbol s to v changes the code by (v - s) * powers[pos].
    # We build neighbor lists by iterating over all words and all radius-0/1 changes.
    neighbors = [[] for _ in range(total)]
    # radius 0: itself
    for w in range(total):
        neighbors[w].append(w)
    if radius >= 1:
        # For each position, for each pair of distinct symbols (s, v), the delta.
        # We can vectorize: for a fixed pos and fixed v, the words that have symbol s at pos
        # get a neighbor at w + (v - s)*p.
        for pos in range(n):
            p = powers[pos]
            d = digits[:, pos]  # shape (total,)
            for s in range(q):
                mask = (d == s)
                if not mask.any():
                    continue
                ws = np.where(mask)[0]
                for v in range(q):
                    if v == s:
                        continue
                    delta = (v - s) * p
                    targets = ws + delta
                    # append each target to neighbors of each ws
                    for i in range(ws.size):
                        neighbors[int(ws[i])].append(int(targets[i]))
    # word_to_int via a dict (fast enough for 2187).
    word_to_int = {}
    for w in range(total):
        syms = tuple(int(digits[w, pos]) for pos in range(n))
        word_to_int[syms] = w
    return neighbors, word_to_int


def construct(q, n, radius):
    start = time.time()
    DEADLINE = 110.0  # safety margin before the 120 s kill (150 s hard limit)

    # Fallback direct-sum construction (valid, ~2187 words).
    def _fallback():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total = q ** n
    try:
        neighbors, word_to_int = _neighbors_all(q, n, radius)
    except Exception:
        return _fallback()

    # Coverage bookkeeping.
    cov = np.zeros(total, dtype=np.int32)  # how many chosen codewords cover each word
    chosen = []
    chosen_set = set()

    def add(cw_int):
        chosen.append(cw_int)
        chosen_set.add(cw_int)
        cov[neighbors[cw_int]] += 1

    # --- Phase 1: greedy set cover ---
    # Start with the all-zeros word (arbitrary deterministic seed) to guarantee
    # a non-empty, valid base, then greedily add the word covering the most
    # currently-uncovered words.
    add(0)

    while time.time() < DEADLINE:
        if cov.min() >= 1:
            break
        # For each candidate codeword, count how many uncovered words it covers.
        # gain[cw] = number of words in neighbors[cw] with cov == 0.
        best_cw = None
        best_gain = 0
        # Iterate candidates; vectorize the gain computation per candidate is
        # expensive, so we accumulate using an array over all candidates at once.
        # Build a 2D boolean "uncovered" mask and count per candidate via the
        # neighbor lists (CSR-like).
        uncovered = (cov == 0)
        # Build CSR from neighbors.
        indptr = np.empty(total + 1, dtype=np.int64)
        indptr[0] = 0
        for i in range(total):
            indptr[i + 1] = indptr[i] + len(neighbors[i])
        indices = np.empty(indptr[-1], dtype=np.int64)
        for i in range(total):
            st = indptr[i]
            indices[st:st + len(neighbors[i])] = neighbors[i]
        # gain[i] = sum(uncovered[indices[indptr[i]:indptr[i+1]]])
        # Compute via np.add.reduceat on a masked index.
        masked = uncovered[indices]  # boolean length = total neighbors
        # reduceat over the segments defined by indptr.
        gain = np.add.reduceat(masked.astype(np.int64), indptr[:-1])
        # Only consider codewords not already chosen (adding a chosen one gains 0 anyway
        # since it's already counted, but its neighbors are already covered by itself;
        # still, re-adding is invalid, so exclude).
        for cw in chosen_set:
            gain[cw] = 0
        best_gain = int(gain.max())
        if best_gain <= 0:
            break
        best_cw = int(np.argmax(gain))
        add(best_cw)

    # --- Phase 2: local swap (1-in, 1-out) to reduce size ---
    # Repeatedly try to remove a codeword whose removal leaves some words
    # uncovered, then re-add the single codeword that covers the most of those
    # newly-uncovered words; accept if it reduces (or keeps) the total size while
    # restoring full coverage.  Net size change is 0 (remove 1, add 1) unless the
    # removed codeword is redundant (remove 1, add 0 -> size -1).
    def _try_reductions():
        nonlocal cov, chosen, chosen_set
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for idx in range(len(chosen)):
                if time.time() > DEADLINE:
                    return
                cw = chosen[idx]
                # Words that would become uncovered if we remove cw:
                # those with cov == 1 and covered by cw.
                # Compute new_cov = cov - 1 on neighbors[cw]; uncovered = new_cov == 0.
                nb = neighbors[cw]
                new_cov = cov.copy()
                new_cov[nb] -= 1
                unc = (new_cov == 0)
                if not unc.any():
                    # cw is redundant: remove it, size -1.
                    cov[nb] -= 1
                    chosen.pop(idx)
                    chosen_set.remove(cw)
                    improved = True
                    break
                # Need to re-add a codeword covering the most of unc.
                # gain2[c] = number of unc words in neighbors[c].
                # Build masked count.
                # Reuse CSR indices.
                masked = unc[indices]
                gain2 = np.add.reduceat(masked.astype(np.int64), indptr[:-1])
                # Exclude already-chosen codewords (adding them is invalid).
                for c in chosen_set:
                    gain2[c] = 0
                # The best replacement must cover ALL of unc to be a valid 1-in-1-out.
                max_unc = int(unc.sum())
                best_c = int(np.argmax(gain2))
                if gain2[best_c] >= max_unc:
                    # Accept: remove cw, add best_c. Size unchanged, but may enable
                    # further reductions.  Only accept if it doesn't increase size;
                    # here size is the same, so accept to allow progress.
                    # To avoid useless churn, only accept if best_c != cw (guaranteed,
                    # cw excluded) -- always true.  Accept.
                    cov = new_cov
                    cov[neighbors[best_c]] += 1
                    chosen[idx] = best_c
                    chosen_set.discard(cw)
                    chosen_set.add(best_c)
                    improved = True
                    break

    _try_reductions()

    # --- Phase 3: second greedy pass over still-uncovered (should be none) ---
    if cov.min() < 1:
        while time.time() < DEADLINE:
            if cov.min() >= 1:
                break
            uncovered = (cov == 0)
            masked = uncovered[indices]
            gain = np.add.reduceat(masked.astype(np.int64), indptr[:-1])
            for cw in chosen_set:
                gain[cw] = 0
            if gain.max() <= 0:
                break
            best_cw = int(np.argmax(gain))
            add(best_cw)

    # Convert integer codewords back to lists of symbols.
    powers = [1] * n
    for pos in range(n - 2, -1, -1):
        powers[pos] = powers[pos + 1] * q
    result = []
    for cw in chosen:
        syms = []
        x = cw
        for pos in range(n - 1, -1, -1):
            p = powers[pos]
            syms.append((x // p) % q)
        syms.reverse()
        result.append(syms)

    # Final validity guard: ensure every word is covered.
    final_cov = np.zeros(total, dtype=np.int32)
    for cw in chosen:
        final_cov[neighbors[cw]] += 1
    if final_cov.min() < 1:
        # Should not happen; fall back to the guaranteed-valid construction.
        return _fallback()

    return result
# EVOLVE-BLOCK-END
