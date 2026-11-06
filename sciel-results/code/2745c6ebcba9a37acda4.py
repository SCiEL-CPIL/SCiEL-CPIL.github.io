"""Sidon set (incl. 0) covering differences 1..30; minimize the largest element."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

_D = 30          # must cover differences 1..D
_START = time.time()
_DEADLINE = _START + 110.0   # safety margin under the 120s wall-clock limit


def _time_left():
    return _DEADLINE - time.time()


def _diffs(arr):
    """All positive differences of a sorted list."""
    out = []
    n = len(arr)
    for i in range(n):
        xi = arr[i]
        for j in range(i + 1, n):
            out.append(arr[j] - xi)
    return out


def _is_sidon(arr):
    """True iff all positive differences are distinct (and arr is sorted)."""
    n = len(arr)
    seen = set()
    for i in range(n):
        xi = arr[i]
        for j in range(i + 1, n):
            d = arr[j] - xi
            if d in seen:
                return False
            seen.add(d)
    return True


def _covers(arr):
    """Set of positive differences present in arr."""
    n = len(arr)
    s = set()
    for i in range(n):
        xi = arr[i]
        for j in range(i + 1, n):
            s.add(arr[j] - xi)
    return s


def _valid(arr):
    if not arr or arr[0] != 0:
        return False
    if len(arr) > 32:
        return False
    if arr[-1] > 10**6:
        return False
    if not _is_sidon(arr):
        return False
    cov = _covers(arr)
    for d in range(1, _D + 1):
        if d not in cov:
            return False
    return True


# ---- Exact cover / DLX (Algorithm X) to find a minimum-size Sidon set ----
class DLX:
    __slots__ = ("w", "h", "colsize", "maxsize", "n", "sol", "cnt")

    def __init__(self, rows, ncols, maxsize=None):
        self.n = ncols
        self.maxsize = maxsize
        self.h = [None] * (ncols + 1)
        self.w = [None] * (ncols + 1)
        self.h[0] = 0
        self.w[0] = 0
        for c in range(1, ncols + 1):
            self.h[c] = c
            self.w[c] = c
        self.colsize = [0] * (ncols + 1)
        for r in range(1, ncols + 1):
            self.w[r - 1] = r
            self.h[r] = r - 1
        self.sol = []
        self.cnt = 0
        for row in rows:
            self.add_row(row)

    def add_row(self, cols):
        r = len(self.h)
        self.h.append(r)
        self.w.append(r)
        self.colsize.append(0)
        first = None
        for c in cols:
            if first is None:
                first = r
            self.h[r] = self.h[c]
            self.w[self.h[c]] = r
            self.h[c] = r
            self.w[r] = self.w[c]
            self.colsize[c] += 1
        self.h[r] = first
        self.w[first] = r

    def _remove(self, c):
        self.w[self.h[c]] = self.w[c]
        self.h[self.w[c]] = self.h[c]
        r = self.w[c]
        while r != c:
            u = self.h[r]
            self.w[u] = self.w[r]
            self.h[self.w[r]] = u
            r = u

    def _restore(self, c):
        r = self.h[c]
        while r != c:
            u = self.w[r]
            self.h[u] = r
            self.w[r] = u
            r = u
        self.w[self.h[c]] = c
        self.h[self.w[c]] = c

    def _choose(self):
        best = 1
        bsz = 10**9
        c = self.w[0]
        while c != 0:
            if self.colsize[c] < bsz:
                bsz = self.colsize[c]
                best = c
                if bsz == 0:
                    break
            c = self.w[c]
        return best

    def _search(self, k):
        if _time_left() <= 0:
            return
        if self.maxsize is not None and k > self.maxsize:
            return
        if self.w[0] == 0:
            self.cnt += 1
            return
        c = self._choose()
        self._remove(c)
        r = self.w[c]
        while r != c:
            self.sol.append(r)
            j = self.w[r]
            while j != r:
                self._remove(self.h[j])
                j = self.w[j]
            self._search(k + 1)
            j = self.h[r]
            while j != r:
                self._restore(self.h[j])
                j = self.h[j]
            self.sol.pop()
            r = self.w[r]
        self._restore(c)


def _build_rows(maxsize):
    """Rows: each (x, x+d) with d in 1..D and x+d <= maxsize.
    Columns: 1..D are the differences; D+1..D+maxsize are the vertices."""
    rows = []
    for d in range(1, _D + 1):
        for x in range(0, maxsize - d + 1):
            rows.append((d, _D + x + 1, _D + x + d + 1))
    return rows


def _solve_size(maxsize, size, first=True, deadline=None):
    """Find a Sidon set of exactly `size` (incl. 0) covering 1..D within maxsize."""
    if deadline is not None and time.time() > deadline:
        return None
    ncols = _D + maxsize + 1
    dlx = DLX(_build_rows(maxsize), ncols, maxsize=size)
    if first:
        dlx._search(0)
    else:
        # force 0 as a vertex (vertex 0 => column D+1)
        dlx.sol.append(0)
        dlx._remove(_D + 1)
        j = dlx.w[0]
        while j != 0:
            dlx._remove(dlx.h[j])
            j = dlx.w[j]
        dlx._search(1)
        # restore
        j = dlx.h[0]
        while j != 0:
            dlx._restore(dlx.h[j])
            j = dlx.w[j]
        dlx._restore(_D + 1)
        dlx.sol.pop()
    if dlx.cnt == 0:
        return None
    cols = dlx.sol[0]
    used = set()
    for r in cols:
        j = dlx.w[r]
        while j != r:
            used.add(dlx.h[j])
            j = dlx.w[j]
    verts = sorted((c - _D - 1) for c in used if c > _D)
    arr = [0] + [v for v in verts if v != 0]
    if arr[0] != 0:
        arr = [0] + arr
    arr = sorted(set(arr))
    return arr


def _solve_any(maxsize, deadline):
    """Minimum-size Sidon set covering 1..D within maxsize (0 included)."""
    for size in range(_D, 33):
        arr = _solve_size(maxsize, size, first=True, deadline=deadline)
        if arr is not None and _valid(arr):
            return arr
    return None


def _polish(arr, deadline):
    """Local search: try to drop or relocate a single element to shrink the max."""
    arr = list(arr)
    improved = True
    while improved and _time_left() > 0.3:
        improved = False
        n = len(arr)
        # 1) try to drop an element (only if still covering 1..D and Sidon)
        for i in range(n):
            if arr[i] == 0:
                continue
            cand = arr[:i] + arr[i + 1:]
            if _is_sidon(cand) and all(d in _covers(cand) for d in range(1, _D + 1)):
                arr = cand
                improved = True
                break
        if improved:
            continue
        # 2) try to relocate a non-zero element to a smaller value
        for i in range(n):
            if arr[i] == 0:
                continue
            base = arr[:i] + arr[i + 1:]
            base_set = set(base)
            base_cov = _covers(base)
            need = [d for d in range(1, _D + 1) if d not in base_cov]
            if not need:
                continue
            cur = arr[i]
            for nv in range(1, cur):
                if nv in base_set:
                    continue
                cand = sorted(base + [nv])
                if _is_sidon(cand):
                    ccov = _covers(cand)
                    if all(d in ccov for d in range(1, _D + 1)):
                        arr = cand
                        improved = True
                        break
            if improved:
                break
    return arr


def _greedy_fallback():
    """Fast constructive heuristic to always have a feasible answer."""
    a = [0]
    for d in range(1, _D + 1):
        if d in _covers(sorted(a)):
            continue
        placed = False
        cands = []
        for v in a:
            for s in (d, -d):
                x = v + s
                if x >= 0 and x not in a:
                    cands.append(x)
        for x in sorted(set(cands)):
            trial = sorted(a + [x])
            if _is_sidon(trial):
                a = trial
                placed = True
                break
        if not placed:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if _is_sidon(trial):
                    a = trial
                    break
                y += 1
    return a


def construct():
    random.seed(20240517)
    best = None

    # Phase 1: binary search the diameter for a minimum-size solution.
    lo, hi = 40, 400
    found = None
    while lo <= hi and _time_left() > 5.0:
        mid = (lo + hi) // 2
        arr = _solve_any(mid, _DEADLINE)
        if arr is not None:
            found = arr
            hi = mid - 1
        else:
            lo = mid + 1
    if found is not None:
        best = _polish(found, _DEADLINE)
    else:
        # Fallback: try a fixed moderate diameter, then greedy.
        for M in (120, 90, 70, 55):
            if _time_left() <= 2.0:
                break
            arr = _solve_any(M, _DEADLINE)
            if arr is not None:
                best = _polish(arr, _DEADLINE)
                break
        if best is None:
            best = _greedy_fallback()

    # Phase 2: polish the best solution.
    if best is not None and _time_left() > 1.0:
        best = _polish(best, _DEADLINE)

    # Final safety: ensure validity; fall back to greedy if broken.
    if best is None or not _valid(best):
        best = _greedy_fallback()
    return best

# EVOLVE-BLOCK-END
