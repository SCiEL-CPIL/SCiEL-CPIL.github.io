import time
import random
import numpy as np

N = 256


def _build_ap_table(n=N):
    """Return a list of (idx, ps) for every AP (step d, residue r) in [0, n).
    idx = positions of the AP, ps = cumsum (length len(idx))."""
    table = []
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                continue
            idx = np.arange(r, n, d)
            if idx.size == 0:
                continue
            table.append((idx, None))
    return table


def _init_ps(table, signs):
    """Fill ps in each AP entry and return the list of ps arrays."""
    sa = np.asarray(signs, dtype=np.int32)
    ps_list = []
    for i, (idx, _) in enumerate(table):
        ps = np.empty(idx.size, dtype=np.int32)
        np.cumsum(sa[idx], out=ps)
        table[i] = (idx, ps)
        ps_list.append(ps)
    return ps_list


def _score_signs(table, signs):
    """Return (global_max_abs_ap_sum, ps_list)."""
    sa = np.asarray(signs, dtype=np.int32)
    ps_list = []
    gmax = 0
    for i, (idx, _) in enumerate(table):
        ps = np.empty(idx.size, dtype=np.int32)
        np.cumsum(sa[idx], out=ps)
        ps_list.append(ps)
        table[i] = (idx, ps)
        m = int(np.max(ps))
        mn = int(np.min(ps))
        v = m - mn
        if v > gmax:
            gmax = v
    return gmax, ps_list


def _init_stats(ps_list):
    """Return (maxs, mins, amaxs, amins, gmax) for the AP table."""
    maxs = np.empty(len(ps_list), dtype=np.int32)
    mins = np.empty(len(ps_list), dtype=np.int32)
    amaxs = np.empty(len(ps_list), dtype=np.int32)
    amins = np.empty(len(ps_list), dtype=np.int32)
    gmax = 0
    for i, ps in enumerate(ps_list):
        m = int(np.max(ps)); mn = int(np.min(ps))
        amax = int(np.argmax(ps)); amin = int(np.argmin(ps))
        maxs[i] = m; mins[i] = mn; amaxs[i] = amax; amins[i] = amin
        v = m - mn
        if v > gmax:
            gmax = v
    return maxs, mins, amaxs, amins, gmax


def _ap_of_pos(n=N):
    """For each position p, the list of (ap_index, k) where k is the index in the AP."""
    table = _build_ap_table(n)
    m = len(table)
    ap_of_pos = [[] for _ in range(n)]
    for i, (idx, _) in enumerate(table):
        for k in range(idx.size):
            ap_of_pos[int(idx[k])].append((i, k))
    return table, ap_of_pos


class _APTable:
    """Holds the AP table and per-AP max/min stats for fast flip accounting."""

    def __init__(self, table, ap_of_pos):
        self.table = table
        self.ap_of_pos = ap_of_pos
        self.ps_list = None
        self.maxs = None
        self.mins = None
        self.amaxs = None
        self.amins = None
        self.gmax = 0
        self.count = 0  # number of APs with range == gmax

    def load(self, signs):
        sa = np.asarray(signs, dtype=np.int32)
        self.ps_list = []
        for i, (idx, _) in enumerate(self.table):
            ps = np.empty(idx.size, dtype=np.int32)
            np.cumsum(sa[idx], out=ps)
            self.table[i] = (idx, ps)
            self.ps_list.append(ps)
        self.maxs, self.mins, self.amaxs, self.amins, self.gmax = _init_stats(self.ps_list)
        self.count = int(np.sum(self.maxs - self.mins == self.gmax))
        return self.gmax

    def _recompute_ap(self, i):
        ps = self.ps_list[i]
        self.maxs[i] = int(np.max(ps))
        self.mins[i] = int(np.min(ps))
        self.amaxs[i] = int(np.argmax(ps))
        self.amins[i] = int(np.argmin(ps))

    def _apply_ap(self, i, k, delta):
        ps = self.ps_list[i]
        ps[k:] += delta
        amax = int(self.amaxs[i]); amin = int(self.amins[i])
        if amax >= k:
            self._recompute_ap(i)
        else:
            self.maxs[i] += delta
        if amin >= k:
            self._recompute_ap(i)
        else:
            self.mins[i] += delta

    def apply_flip(self, p, old_sign):
        delta = -2 * old_sign
        for i, k in self.ap_of_pos[p]:
            self._apply_ap(i, k, delta)
        return self._recompute_gmax()

    def _recompute_gmax(self):
        g = int(np.max(self.maxs - self.mins))
        self.gmax = g
        self.count = int(np.sum(self.maxs - self.mins == g))
        return g

    def get_gmax(self):
        return int(self.gmax)


def _thue_morse(n=N):
    return [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]


def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    n = N
    rng = random.Random(12345)

    table, ap_of_pos = _ap_of_pos(n)
    ap = _APTable(table, ap_of_pos)

    # ---- initializations ----
    inits = [
        _thue_morse(n),
        [1 if i % 2 == 0 else -1 for i in range(n)],
        [1 if (i // 2) % 2 == 0 else -1 for i in range(n)],
        [1 if (i // 3) % 2 == 0 else -1 for i in range(n)],
    ]
    for _ in range(3):
        inits.append([rng.choice((-1, 1)) for _ in range(n)])

    best_signs = None
    best_score = float('inf')
    for init in inits:
        s, _ = _score_signs(table, init)
        if s < best_score:
            best_score = s
            best_signs = init[:]
        if time.time() > deadline:
            break

    current = best_signs[:]
    cur = ap.load(current)

    # ---- local search (single + double flips) with simulated-annealing restarts ----
    no_improve = 0
    max_no_improve = 50
    restart_interval = 1200
    it = 0
    cand = list(range(n))

    def recompute_candidates():
        nonlocal cand
        if ap.gmax <= 0:
            cand = list(range(n))
            return
        thr = ap.gmax - 1
        cnt = np.zeros(n, dtype=np.int32)
        for i, (idx, ps) in enumerate(ap.table):
            if ap.maxs[i] - ap.mins[i] >= thr:
                for p in idx:
                    cnt[p] += 1
        cand = np.argsort(-cnt)[:40].tolist()
        if not cand:
            cand = list(range(n))

    while time.time() < deadline:
        it += 1

        if no_improve > max_no_improve or it % restart_interval == 0:
            if rng.random() < 0.45:
                current = best_signs[:]
                for _ in range(rng.randint(3, 12)):
                    idx = rng.randint(0, n - 1)
                    current[idx] = -current[idx]
            else:
                current = _thue_morse(n)
                for _ in range(rng.randint(2, 8)):
                    idx = rng.randint(0, n - 1)
                    current[idx] = -current[idx]
            cur = ap.load(current)
            if cur < best_score:
                best_score = cur
                best_signs = current[:]
            no_improve = 0
            recompute_candidates()
            continue

        if it % 40 == 0:
            recompute_candidates()

        if rng.random() < 0.85:
            p = cand[rng.randint(0, len(cand) - 1)] if cand else rng.randint(0, n - 1)
            old = current[p]
            ns = ap.apply_flip(p, old)
            current[p] = -old
            if ns < cur:
                cur = ns
                no_improve = 0
                if cur < best_score:
                    best_score = cur
                    best_signs = current[:]
            else:
                # undo
                ap.apply_flip(p, -old)  # flip back: old_sign for this flip is -old
                current[p] = old
                no_improve += 1
        else:
            if cand and len(cand) >= 2:
                i1 = cand[rng.randint(0, len(cand) - 1)]
                i2 = cand[rng.randint(0, len(cand) - 1)]
                if i1 == i2:
                    i2 = (i2 + 1) % n
            else:
                i1 = rng.randint(0, n - 1)
                i2 = rng.randint(0, n - 1)
                if i1 == i2:
                    i2 = (i2 + 1) % n
            o1 = current[i1]; o2 = current[i2]
            ap.apply_flip(i1, o1); current[i1] = -o1
            ap.apply_flip(i2, o2); current[i2] = -o2
            ns = ap.get_gmax()
            if ns < cur:
                cur = ns
                no_improve = 0
                if cur < best_score:
                    best_score = cur
                    best_signs = current[:]
            else:
                ap.apply_flip(i2, -o2); current[i2] = o2
                ap.apply_flip(i1, -o1); current[i1] = o1
                no_improve += 1

    # ---- final high-precision polish pass (exact objective) ----
    if time.time() < deadline:
        best_signs = best_signs[:]
        best_score = _score_signs(table, best_signs)[0]
        cur = best_score
        ap.load(best_signs)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for p in range(n):
                if time.time() > deadline:
                    break
                old = best_signs[p]
                ns = ap.apply_flip(p, old)
                best_signs[p] = -old
                if ns < cur:
                    cur = ns
                    best_score = ns
                    improved = True
                else:
                    ap.apply_flip(p, -old)
                    best_signs[p] = old

    return best_signs
