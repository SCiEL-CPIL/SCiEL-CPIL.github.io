import time
import numpy as np

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE = time.time() + 140.0


def _to_base3(a):
    a = np.asarray(a, dtype=np.int64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    n = a.shape[1]
    powers = 3 ** np.arange(n - 1, -1, -1, dtype=np.int64)
    return (a * powers).sum(axis=1).astype(np.int64)


def _from_base3(x, n, q=3):
    x = np.asarray(x, dtype=np.int64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    out = np.zeros((x.shape[0], n), dtype=np.int64)
    for i in range(n - 1, -1, -1):
        out[:, i] = x % q
        x = x // q
    return out


def construct(q, n, radius):
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    if q < 2:
        q = 2
    if radius < 1:
        radius = 1
    if n < 1:
        n = 1

    # ---- Precompute per-position Hamming masks (vectorized distance helper) ----
    masks = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            masks[i, j] = 1 if i == j else 0

    # ---- Warm start: direct sum of the ternary tetracode (9 words, len 4) ----
    tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tet = np.array(tet, dtype=np.int64)
    if n >= 4:
        tails = _from_base3(np.arange(3 ** (n - 4), dtype=np.int64), n - 4)
        code = np.concatenate([np.repeat(tet, tails.shape[0], axis=0),
                               np.tile(tails, (tet.shape[0], 1))], axis=1)
    else:
        code = np.array([[0] * n], dtype=np.int64)
    code = np.unique(code, axis=0)

    # ---- Vectorized greedy: repeatedly add the word covering the most uncovered ----
    while time.time() < _DEADLINE - 25:
        if len(code) == 0:
            code = np.array([[0] * n], dtype=np.int64)
        # Current coverage (any codeword within radius).
        covered = np.zeros(3 ** n, dtype=bool)
        for i in range(len(code)):
            d = np.abs(_from_base3(np.arange(3 ** n, dtype=np.int64), n) - code[i]).sum(axis=1)
            covered |= (d <= radius)
        if covered.all():
            break
        # Find uncovered words.
        uncov = np.where(~covered)[0]
        best_word = None
        best_gain = -1
        # Score candidate words; prefer uncovered words, then random tie-break.
        cand = uncov.copy()
        if len(cand) > 200000:
            cand = cand[np.random.choice(len(cand), 200000, replace=False)]
        else:
            np.random.shuffle(cand)
        for idx in cand:
            w = _from_base3(np.array([idx], dtype=np.int64), n)[0]
            d = np.abs(_from_base3(np.arange(3 ** n, dtype=np.int64), n) - w).sum(axis=1)
            ball = d <= radius
            gain = int((ball & ~covered).sum())
            if gain > best_gain:
                best_gain = gain
                best_word = w
                if gain >= 0.5 * len(uncov):
                    break
        if best_word is None:
            break
        code = np.vstack([code, best_word.reshape(1, -1)])

    # ---- Swap-based local search: 2-for-1 / 1-for-2 to escape local optima ----
    def covered_by(code):
        c = np.zeros(3 ** n, dtype=bool)
        for i in range(len(code)):
            d = np.abs(_from_base3(np.arange(3 ** n, dtype=np.int64), n) - code[i]).sum(axis=1)
            c |= (d <= radius)
        return c

    def is_valid(code):
        return covered_by(code).all()

    base_cov = covered_by(code)
    base_count = len(code)
    rng = np.random.default_rng(0)

    def try_remove_and_add(code, cov, i):
        # Remove codeword i, then add the best word to maximize coverage.
        others = np.delete(code, i, axis=0)
        others_cov = np.zeros(3 ** n, dtype=bool)
        for j in range(len(others)):
            d = np.abs(_from_base3(np.arange(3 ** n, dtype=np.int64), n) - others[j]).sum(axis=1)
            others_cov |= (d <= radius)
        if others_cov.all():
            return others, others_cov  # removal alone keeps validity
        uncov = np.where(~others_cov)[0]
        cand = uncov.copy()
        if len(cand) > 100000:
            cand = cand[rng.choice(len(cand), 100000, replace=False)]
        else:
            cand = cand[rng.permutation(len(cand))]
        best_w, best_g = None, -1
        for idx in cand:
            w = _from_base3(np.array([idx], dtype=np.int64), n)[0]
            d = np.abs(_from_base3(np.arange(3 ** n, dtype=np.int64), n) - w).sum(axis=1)
            ball = d <= radius
            g = int((ball & ~others_cov).sum())
            if g > best_g:
                best_g, best_w = g, w
                if best_g >= 0.5 * len(uncov):
                    break
        if best_w is None:
            return None, None
        new_code = np.vstack([others, best_w.reshape(1, -1)])
        new_cov = covered_by(new_code)
        if new_cov.all():
            return new_code, new_cov
        return None, None

    # 1-for-2: try to replace one codeword with two to reduce count (net -1).
    # We attempt: remove i, add two words, if still valid and count reduced by 1.
    improved = True
    while improved and time.time() < _DEADLINE - 15:
        improved = False
        for i in range(len(code)):
            if time.time() > _DEADLINE - 15:
                break
            new_code, new_cov = try_remove_and_add(code, base_cov, i)
            if new_code is not None and len(new_code) < len(code):
                code, base_cov = new_code, new_cov
                improved = True
                break

    # ---- Final exact polish: drop any redundant codeword (tighter check) ----
    for _ in range(3):
        if time.time() > _DEADLINE - 5:
            break
        changed = False
        for i in range(len(code)):
            others = np.delete(code, i, axis=0)
            if is_valid(others):
                code = others
                changed = True
                break
        if not changed:
            break

    return [list(map(int, row)) for row in code]
