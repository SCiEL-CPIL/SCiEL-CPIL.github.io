import time
import numpy as np

N = 256

# ---- Precomputation ----
# all_aps: array of shape (M,4) of 0-indexed positions of every 4-AP
# For each position p: affected_aps[p] = indices into all_aps where p appears
# other_idx[p] = the other three positions (for delta computation)

def _precompute():
    rows = []
    for d in range(1, (N - 3) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            rows.append((i0, i0 + d, i0 + 2 * d, i0 + 3 * d))
    all_aps = np.array(rows, dtype=np.int32)
    affected_aps = [None] * N
    other_idx = [None] * N
    for p in range(N):
        mask = np.any(all_aps == p, axis=1)
        idx = np.where(mask)[0]
        affected_aps[p] = idx
        # other three positions for each affected AP
        if len(idx) > 0:
            sub = all_aps[idx]  # (k,4)
            # for each row, the three columns that are not p
            # build a (k,3) array of the other indices
            other = np.empty((len(idx), 3), dtype=np.int32)
            for j in range(4):
                col = sub[:, j]
                notp = col != p
                other[:, j] = np.where(notp, col, -1)
            # remove the -1 entries per row -> actually exactly one -1 per row
            # reshape: for each row, the three non-p entries
            # since exactly one column equals p, the other three are valid
            # We'll build other_idx[p] as (k,3)
            # Find which column is p for each row
            pcol = np.where(sub == p)  # (k,) row indices and (k,) col indices
            # For each row, the three other columns
            cols = np.arange(4)
            # other_idx[p] shape (k,3): for row r, the three columns != pcol[r]
            other2 = np.empty((len(idx), 3), dtype=np.int32)
            for r in range(len(idx)):
                c = pcol[1][r]
                others = [c2 for c2 in range(4) if c2 != c]
                other2[r] = sub[r, others]
            other_idx[p] = other2
        else:
            other_idx[p] = np.empty((0, 3), dtype=np.int32)
    return all_aps, affected_aps, other_idx

_ALL_APS, _AFFECTED, _OTHER = _precompute()
_M = len(_ALL_APS)

def count_mono(c):
    """c: numpy array of 0/1, length 256. Returns int count of mono 4-APs."""
    if _M == 0:
        return 0
    vals = c[_ALL_APS]  # (M,4)
    return int(np.sum(np.all(vals == vals[:, :1], axis=1)))

def flip_delta(c, p):
    """Return change in mono count if we flip position p. c is current (unflipped)."""
    idx = _AFFECTED[p]
    if len(idx) == 0:
        return 0
    oi = _OTHER[p]  # (k,3)
    cur = c[oi]  # (k,3)
    oldp = c[p]
    newp = 1 - oldp
    # was mono: oldp == all three others
    was_mono = np.all(cur == oldp, axis=1)
    # now mono: newp == all three others
    is_mono = np.all(cur == newp, axis=1)
    delta = int(np.sum(is_mono) - np.sum(was_mono))
    return delta

def pair_delta(c, p, q):
    """Return change in mono count if we flip both p and q."""
    if p == q:
        return 0
    setp = set(_AFFECTED[p])
    setq = set(_AFFECTED[q])
    common = list(setp.intersection(setq))
    onlyp = [i for i in _AFFECTED[p] if i not in setq]
    onlyq = [i for i in _AFFECTED[q] if i not in setp]
    delta = 0
    c_ = c
    # onlyp: only p flips
    for i in onlyp:
        ap = _ALL_APS[i]
        was = np.all(c_[ap] == c_[ap[0]], axis=0) if False else (len(set(c_[ap])) == 1)
        # after flipping p
        c_[p] = 1 - c_[p]
        now = (len(set(c_[ap])) == 1)
        c_[p] = 1 - c_[p]
        if now and not was:
            delta += 1
        elif was and not now:
            delta -= 1
    # onlyq: only q flips
    for i in onlyq:
        ap = _ALL_APS[i]
        was = (len(set(c_[ap])) == 1)
        c_[q] = 1 - c_[q]
        now = (len(set(c_[ap])) == 1)
        c_[q] = 1 - c_[q]
        if now and not was:
            delta += 1
        elif was and not now:
            delta -= 1
    # common: both flip
    for i in common:
        ap = _ALL_APS[i]
        was = (len(set(c_[ap])) == 1)
        c_[p] = 1 - c_[p]
        c_[q] = 1 - c_[q]
        now = (len(set(c_[ap])) == 1)
        c_[p] = 1 - c_[p]
        c_[q] = 1 - c_[q]
        if now and not was:
            delta += 1
        elif was and not now:
            delta -= 1
    return delta

def _thue_morse(n):
    seq = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        seq[i] = seq[i >> 1] ^ (i & 1)
    return seq

def _random_color():
    return np.random.randint(0, 2, size=N, dtype=np.int8)

def _starts():
    starts = []
    starts.append(_thue_morse(N))
    starts.append(1 - _thue_morse(N))
    # shifted thue-morse
    tm = _thue_morse(N + 8)
    for s in [1, 2, 3, 4, 5, 6]:
        starts.append(tm[s:s+N])
    # v2 color
    v2 = np.zeros(N, dtype=np.int8)
    for i in range(1, N):
        x = i + 1
        p = 0
        while x % 2 == 0:
            x //= 2
            p += 1
        v2[i] = p % 2
    starts.append(v2)
    starts.append(1 - v2)
    # block patterns
    for b in [2, 3, 4, 5, 7]:
        blk = np.zeros(N, dtype=np.int8)
        for i in range(N):
            blk[i] = (i // b) % 2
        starts.append(blk)
    # random starts
    for _ in range(6):
        starts.append(_random_color())
    # XOR thue-morse with bit patterns
    base = _thue_morse(N)
    for shift in [1, 2, 3, 4, 5]:
        pat = np.zeros(N, dtype=np.int8)
        idx = np.arange(N)
        pat = ((idx >> shift) & 1).astype(np.int8)
        starts.append((base ^ pat))
    return starts

def _greedy(c, deadline):
    c = c.copy()
    cur = count_mono(c)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in range(N):
            if time.time() > deadline:
                return c, cur
            d = flip_delta(c, p)
            if d < 0:
                c[p] = 1 - c[p]
                cur += d
                improved = True
    return c, cur

def _anneal(c, deadline):
    c = c.copy()
    cur = count_mono(c)
    t = time.time()
    total = deadline - t
    t0 = t
    T_start = 1.5
    T_end = 0.05
    # pair flip probability
    pair_prob = 0.12
    while time.time() < deadline:
        # temperature schedule (linear in time)
        frac = min(1.0, (time.time() - t0) / max(1e-9, total))
        T = T_start + (T_end - T_start) * frac
        if np.random.random() < pair_prob:
            p = np.random.randint(0, N)
            q = np.random.randint(0, N)
            if p == q:
                continue
            d = pair_delta(c, p, q)
            newc = cur + d
            if newc < cur or (T > 0 and np.random.random() < np.exp(-d / T)):
                c[p] = 1 - c[p]
                c[q] = 1 - c[q]
                cur = newc
        else:
            p = np.random.randint(0, N)
            d = flip_delta(c, p)
            newc = cur + d
            if newc < cur or (T > 0 and np.random.random() < np.exp(-d / T)):
                c[p] = 1 - c[p]
                cur = newc
    return c, cur

def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill

    rng = np.random.default_rng(12345)
    np.random.seed(12345)

    best = None
    best_count = float('inf')

    for s in _starts():
        if time.time() > deadline:
            break
        s = np.asarray(s, dtype=np.int8)
        if len(s) != N:
            s = s[:N]
        c, cnt = _greedy(s, deadline)
        if cnt < best_count:
            best_count = cnt
            best = c.copy()
        # anneal from the greedy result
        c2, cnt2 = _anneal(c, deadline)
        if cnt2 < best_count:
            best_count = cnt2
            best = c2.copy()

    # final greedy polish on best
    if best is not None and time.time() < deadline:
        c3, cnt3 = _greedy(best, deadline)
        if cnt3 < best_count:
            best_count = cnt3
            best = c3.copy()

    if best is None:
        best = _thue_morse(N)

    # Return list of ints 0/1
    return [int(x) for x in best]
