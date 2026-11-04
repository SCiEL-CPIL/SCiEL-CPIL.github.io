"""Ternary covering code K_3(6,1) — minimize codewords covering {0,1,2}^6 within radius 1."""
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 116.0  # safety margin before the 120s kill

    # ---------- fast coverage machinery (NumPy) ----------
    qn = q ** n
    # ball_of[cw] = array of word-indices within Hamming distance `radius` of cw
    ball_of = np.empty(qn, dtype=object)
    for idx in range(qn):
        cw = np.zeros(n, dtype=np.int8)
        t = idx
        for i in range(n - 1, -1, -1):
            cw[i] = t % q
            t //= q
        ball_of[idx] = _ball(cw, q, n, radius)

    def cw_idx(cw):
        v = 0
        for x in cw:
            v = v * q + x
        return v

    def coverage(cws):
        """Return a boolean array: True where covered by at least one cw."""
        covered = np.zeros(qn, dtype=bool)
        for cw in cws:
            covered |= ball_of[cw_idx(cw)]
        return covered

    def is_valid(cws):
        return coverage(cws).all()

    def try_remove(cws, i):
        """Return True if removing cws[i] keeps a full cover (mutates cws on success)."""
        cov = coverage(cws)
        cnt = np.zeros(qn, dtype=np.int8)
        for j, cw in enumerate(cws):
            cnt += ball_of[cw_idx(cw)]
        cnt[ball_of[cw_idx(cws[i])]] -= 1
        if cnt.min() >= 1:
            cws.pop(i)
            return True
        return False

    # ---------- initial construction (direct sum of ternary tetracode) ----------
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    cws = [h + list(tail) for h in tetracode
           for tail in itertools.product(range(q), repeat=n - 4)]
    if not is_valid(cws):
        # defensive: pad with random words until valid (should not happen for tetracode)
        rng0 = random.Random(12345)
        while not is_valid(cws):
            cws.append([rng0.randrange(q) for _ in range(n)])

    best_cws = cws[:]
    best_len = len(cws)

    # ---------- Phase 1: greedy removal (recompute coverage each pass) ----------
    improved = True
    while improved and time.time() < deadline - 25.0:
        improved = False
        cov = coverage(cws)
        cnt = np.zeros(qn, dtype=np.int8)
        for cw in cws:
            cnt += ball_of[cw_idx(cw)]
        for i in range(len(cws) - 1, -1, -1):
            if time.time() > deadline - 25.0:
                break
            cnt[ball_of[cw_idx(cws[i])]] -= 1
            if cnt.min() >= 1:
                cws.pop(i)
                improved = True
                break
        if len(cws) < best_len:
            best_len = len(cws)
            best_cws = cws[:]

    # ---------- Phase 2: randomized local search (remove / swap / add-remove) ----------
    rng = random.Random(20240611)
    rng.shuffle(cws)
    no_improve = 0
    while time.time() < deadline - 12.0 and len(cws) > 1:
        op = rng.random()
        if op < 0.35:
            # single removal
            i = rng.randrange(len(cws))
            if try_remove(cws, i):
                no_improve = 0
                if len(cws) < best_len:
                    best_len = len(cws)
                    best_cws = cws[:]
            else:
                no_improve += 1
        elif op < 0.70:
            # swap: remove i, then add a random word that covers a currently-thin word
            i = rng.randrange(len(cws))
            cnt = np.zeros(qn, dtype=np.int8)
            for cw in cws:
                cnt += ball_of[cw_idx(cw)]
            cnt[ball_of[cw_idx(cws[i])]] -= 1
            if cnt.min() >= 1:
                cws.pop(i)
                no_improve = 0
                if len(cws) < best_len:
                    best_len = len(cws)
                    best_cws = cws[:]
                continue
            # find words that become uncovered by removing i
            cand = np.where(cnt == 0)[0]
            if len(cand) == 0:
                continue
            target = cand[rng.randrange(len(cand))]
            # try random words; keep one that covers target
            added = None
            for _ in range(40):
                w = [rng.randrange(q) for _ in range(n)]
                if ball_of[cw_idx(w)][target]:
                    added = w
                    break
            if added is not None:
                cws.pop(i)
                cws.append(added)
                no_improve = 0
                if len(cws) < best_len:
                    best_len = len(cws)
                    best_cws = cws[:]
            else:
                no_improve += 1
        else:
            # add a random word, then try to remove two
            w = [rng.randrange(q) for _ in range(n)]
            cws.append(w)
            removed = 0
            for _ in range(2):
                if len(cws) <= 1:
                    break
                j = rng.randrange(len(cws))
                if try_remove(cws, j):
                    removed += 1
            # drop the added word if still present and removable
            if w in cws:
                if try_remove(cws, cws.index(w)):
                    pass
            if len(cws) < best_len:
                best_len = len(cws)
                best_cws = cws[:]
                no_improve = 0
            else:
                no_improve += 1

        if no_improve > 4000:
            no_improve = 0

    # ---------- Phase 3: final polish on best_cws (cheap removal) ----------
    cws = best_cws[:]
    for _ in range(4000):
        if time.time() > deadline - 4.0 or len(cws) <= 1:
            break
        i = rng.randrange(len(cws))
        if try_remove(cws, i):
            if len(cws) < best_len:
                best_len = len(cws)
                best_cws = cws[:]

    cws = best_cws[:]

    # ---------- Phase 4: guaranteed validity repair ----------
    # Ensure a valid cover; if not, greedily add words covering the most uncovered words.
    cov = coverage(cws)
    if not cov.all():
        uncovered = np.where(~cov)[0]
        while len(uncovered) > 0 and time.time() < deadline - 1.0:
            # pick the word covering the most currently-uncovered words
            best_w = None
            best_gain = -1
            for idx in range(qn):
                b = ball_of[idx]
                gain = b[uncovered].sum()
                if gain > best_gain:
                    best_gain = gain
                    best_w = idx
                if gain == len(uncovered):
                    break
            if best_w is None or best_gain <= 0:
                break
            cw = [0] * n
            t = best_w
            for i in range(n - 1, -1, -1):
                cw[i] = t % q
                t //= q
            cws.append(cw)
            cov |= ball_of[best_w]
            uncovered = np.where(~cov)[0]
        # absolute fallback: full code (always valid)
        if not cov.all():
            cws = [[0] * n]
            cov = coverage(cws)
            while not cov.all():
                uncovered = np.where(~cov)[0]
                best_w = None
                best_gain = -1
                for idx in range(qn):
                    gain = ball_of[idx][uncovered].sum()
                    if gain > best_gain:
                        best_gain = gain
                        best_w = idx
                    if gain == len(uncovered):
                        break
                cw = [0] * n
                t = best_w
                for i in range(n - 1, -1, -1):
                    cw[i] = t % q
                    t //= q
                cws.append(cw)
                cov |= ball_of[best_w]

    # de-duplicate (defensive)
    seen = set()
    out = []
    for cw in cws:
        key = tuple(cw)
        if key not in seen:
            seen.add(key)
            out.append(cw)
    return out


def _ball(cw, q, n, radius):
    """All word-indices within Hamming distance radius of cw (cw is an np array)."""
    covered = set()
    covered.add(_widx(cw, q))
    if radius >= 1:
        for pos in range(n):
            orig = cw[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                cw[pos] = nv
                covered.add(_widx(cw, q))
            cw[pos] = orig
    return np.fromiter(covered, dtype=np.int64, count=len(covered))


def _widx(cw, q):
    v = 0
    for x in cw:
        v = v * q + int(x)
    return v
