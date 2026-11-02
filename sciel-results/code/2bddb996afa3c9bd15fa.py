"""Covering code K_q(n,r): fast greedy set-cover with numpy bitsets + redundancy pruning."""
import time
import itertools
import numpy as np

_DEADLINE = 150.0
_MARGIN = 12.0


def _now():
    return time.time()


def _is_covered_fast(code, q, n, r):
    """Check coverage using precomputed neighbor masks (passed via closure not possible);
    standalone fallback using set logic."""
    covered = set()
    for cw in code:
        covered.add(tuple(cw))
        for i in range(n):
            for a in range(q):
                if a != cw[i]:
                    w = list(cw)
                    w[i] = a
                    covered.add(tuple(w))
    for w in itertools.product(range(q), repeat=n):
        if w not in covered:
            return False
    return True


def _build_neighbor_masks(q, n):
    """Return (total, masks) where masks[v] is a numpy bool array of length total,
    True for words covered by candidate v (v itself + distance-1 neighbors)."""
    total = q ** n
    masks = np.zeros((total, total), dtype=bool)
    for v in range(total):
        t = [(v // (q ** (n - 1 - i))) % q for i in range(n)]
        masks[v, v] = True
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


def _greedy(total, masks, q, n, rng, deadline):
    """Greedy set-cover using bitset masks. Returns list of codewords."""
    uncovered = np.ones(total, dtype=bool)
    code = []
    while uncovered.any():
        gains = masks[uncovered_indices if False else slice(None)] @ uncovered  # placeholder
        # compute gain for each candidate
        gains = np.array([masks[v].sum() & 0 for v in range(total)])  # placeholder
        break
    return code


def construct(q, n, radius):
    start = time.time()
    deadline = start + 150.0

    # Build neighbor masks once.
    total, masks = _build_neighbor_masks(q, n)

    # word index <-> tuple helpers
    def idxtuple(v):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = v % q
            v //= q
        return tuple(t)

    def greedy():
        uncovered = np.ones(total, dtype=bool)
        code = []
        while uncovered.any():
            # gain for each candidate = number of currently uncovered words it covers
            gains = masks @ uncovered.astype(np.int32)
            best = int(np.argmax(gains))
            if gains[best] <= 0:
                break
            code.append(best)
            uncovered = uncovered & ~masks[best]
            if _now() > deadline - _MARGIN:
                break
        return code

    # Baseline: direct sum for ternary r=1
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        baseline = [h + list(tail) for h in tetracode
                    for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        baseline = [list(idxtuple(v)) for v in greedy()]

    best = baseline
    best_len = len(best)

    # Improve with greedy from random tie-breaks: add small random perturbation to gains.
    rng = np.random.default_rng(0)
    attempts = 0
    while _now() < deadline - _MARGIN and attempts < 50:
        def greedy_pert():
            uncovered = np.ones(total, dtype=bool)
            code = []
            while uncovered.any():
                gains = masks @ uncovered.astype(np.int32)
                gains = gains + rng.random(total) * 0.5
                best = int(np.argmax(gains))
                if (masks[best] & uncovered).sum() <= 0:
                    break
                code.append(best)
                uncovered = uncovered & ~masks[best]
                if _now() > deadline - _MARGIN:
                    break
            return code
        code = greedy_pert()
        if len(code) < best_len:
            best = [list(idxtuple(v)) for v in code]
            best_len = len(code)
        attempts += 1

    # Redundancy pruning: remove codewords whose removal keeps coverage.
    if _now() < deadline - _MARGIN:
        covered_all = np.zeros(total, dtype=bool)
        for v in best:
            idx = 0
            for x in v:
                idx = idx * q + x
            covered_all |= masks[idx]
        changed = True
        while changed and _now() < deadline - _MARGIN:
            changed = False
            for i in range(len(best)):
                idx = 0
                for x in best[i]:
                    idx = idx * q + x
                # recompute coverage without i
                tmp = np.zeros(total, dtype=bool)
                for j in range(len(best)):
                    if j == i:
                        continue
                    jj = 0
                    for x in best[j]:
                        jj = jj * q + x
                    tmp |= masks[jj]
                if tmp.all():
                    best = best[:i] + best[i + 1:]
                    changed = True
                    break

    return best
