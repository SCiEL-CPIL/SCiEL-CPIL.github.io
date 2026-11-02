"""Covering code K_q(n,r): fast greedy + swap local search + coverage-count pruning."""
import time
import itertools
import numpy as np

_DEADLINE = 150.0
_MARGIN = 15.0


def _now():
    return time.time()


def _build_masks(q, n, radius):
    """Return (total, masks) where masks[v] is bool array of words within radius of v."""
    total = q ** n
    masks = np.zeros((total, total), dtype=bool)
    # Precompute all words as index
    for v in range(total):
        # v covers itself and all words within Hamming distance <= radius
        # For radius 1: v + n*(q-1) neighbors
        masks[v, v] = True
        if radius >= 1:
            t = [(v // (q ** (n - 1 - i))) % q for i in range(n)]
            for i in range(n):
                for a in range(q):
                    if a == t[i]:
                        continue
                    tt = list(t)
                    tt[i] = a
                    u = 0
                    for x in tt:
                        u = u * q + x
                    masks[v, u] = True
    return total, masks


def _greedy_cover(masks, total, rng, deadline, margin):
    """Greedy set-cover. Returns list of chosen indices."""
    uncovered = np.ones(total, dtype=bool)
    code = []
    unc_count = total
    while unc_count > 0:
        if _now() > deadline - margin:
            break
        gains = masks @ uncovered.astype(np.int32)
        # Add tiny random perturbation for diversity
        gains = gains + rng.random(total) * 0.1
        best = int(np.argmax(gains))
        gain_val = int((masks[best] & uncovered).sum())
        if gain_val <= 0:
            break
        code.append(best)
        uncovered = uncovered & ~masks[best]
        unc_count -= gain_val
    return code


def _coverage_count(masks, code_indices, total):
    """Return array where cov[i] = number of codewords covering word i."""
    cov = np.zeros(total, dtype=np.int32)
    for v in code_indices:
        cov[masks[v]] += 1
    return cov


def _try_prune(masks, code_indices, total, deadline, margin):
    """Try to remove redundant codewords. Returns new list."""
    cov = _coverage_count(masks, code_indices, total)
    changed = True
    code_list = list(code_indices)
    while changed and _now() < deadline - margin:
        changed = False
        for i in range(len(code_list)):
            if _now() > deadline - margin:
                break
            # Check if removing code_list[i] leaves all covered
            # A word is uncovered after removal iff cov[word]==1 and masks[code_list[i],word]
            # i.e., cov[word] == 1 and the word is in the mask of code_list[i]
            m = masks[code_list[i]]
            # Words that would become uncovered: those with cov==1 and in mask
            would_lose = (cov == 1) & m
            if not would_lose.any():
                # Safe to remove
                cov[m] -= 1
                code_list.pop(i)
                changed = True
                break
    return code_list


def _swap_local_search(masks, code_indices, total, deadline, margin):
    """Try swap: remove one, add another, if it reduces size or same size but better coverage distribution.
    Actually for minimization: try to replace one codeword with another such that coverage is maintained
    and we can then prune. More useful: try to find a codeword that covers all words that
    a removed codeword covered (i.e., the removed codeword's unique words are covered by others).
    This is already handled by prune. So swap here means: try adding a new codeword that allows
    removing two existing ones (net -1).
    """
    code_list = list(code_indices)
    if not code_list:
        return code_list
    cov = _coverage_count(masks, code_list, total)
    improved = True
    while improved and _now() < deadline - margin:
        improved = False
        # Try: for each pair (i, j), see if adding a new codeword k can allow removing both i and j
        # That's expensive. Simpler: try single swap - remove i, add k, then prune.
        # Even simpler and effective: try to add a codeword that maximally covers currently
        # singly-covered words, then prune.
        # Let's do: find word with min coverage (==1), try to find a codeword that covers it
        # and also covers other singly-covered words.
        min_cov = cov.min()
        if min_cov > 1:
            break
        # Find words with cov==1
        single_words = np.where(cov == 1)[0]
        if len(single_words) == 0:
            break
        # For each candidate new codeword, count how many single_words it covers
        # and whether after adding it, we can remove some existing codewords
        # Simplified: just try adding the best codeword for single words, then prune
        gains_single = np.array([masks[v][single_words].sum() for v in range(total)])
        # Also require that the new codeword is not already in code_list
        in_code = np.zeros(total, dtype=bool)
        for v in code_list:
            in_code[v] = True
        gains_single[in_code] = -1
        best_new = int(np.argmax(gains_single))
        if gains_single[best_new] <= 1:
            # Not worth it
            break
        # Add best_new, then try to prune
        code_list.append(best_new)
        cov = _coverage_count(masks, code_list, total)
        # Now prune
        code_list = _try_prune(masks, code_list, total, deadline, margin)
        if len(code_list) < len(code_list):
            pass
        # Check if we actually improved (length decreased)
        # If not, we might have made it worse; but _try_prune should handle it
        # Actually we need to track: if after add+prune, length > original, revert
        # This is getting complex. Let me simplify: only do prune-based reduction.
        improved = (len(code_list) < total)  # always true if non-empty
        # Actually let me just break after one round to avoid infinite loops
        break
    return code_list


def construct(q, n, radius):
    start = time.time()
    deadline = start + _DEADLINE

    total, masks = _build_masks(q, n, radius)

    def idxtuple(v):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = v % q
            v //= q
        return tuple(t)

    best_code = None
    best_len = total + 1

    rng = np.random.default_rng(42)

    # Multiple greedy restarts with different random seeds/perturbations
    max_restarts = 30
    for attempt in range(max_restarts):
        if _now() > deadline - _MARGIN:
            break
        code = _greedy_cover(masks, total, rng, deadline, _MARGIN)
        if not code:
            continue
        # Prune
        code = _try_prune(masks, code, total, deadline, _MARGIN)
        if len(code) < best_len:
            best_len = len(code)
            best_code = list(code)
            # Try swap local search on the best
            if _now() < deadline - _MARGIN:
                improved = _swap_local_search(masks, best_code, total, deadline, _MARGIN)
                if len(improved) < best_len:
                    best_len = len(improved)
                    best_code = improved

    # Final prune on best
    if best_code is not None and _now() < deadline - _MARGIN:
        best_code = _try_prune(masks, best_code, total, deadline, _MARGIN)
        best_len = len(best_code)

    if best_code is None:
        # Fallback: should not happen
        return [list(idxtuple(v)) for v in range(total)]

    return [list(idxtuple(v)) for v in best_code]
