"""No-isosceles set: maximize grid points in [0,N)^2 with no isosceles triangle.

Equivalent constraint: for every selected point, the squared distances from it to
all other selected points are pairwise distinct. (An isosceles triple always has an
apex with two equal edges, and vice-versa.) This is a per-vertex packing constraint,
much looser in 2D than the 1D line used by the seed.
"""
import time
import random

N = 64
DEADLINE = time.time() + 320  # safety margin before the 330s kill

_X = [0] * (N * N)
_Y = [0] * (N * N)
for _x in range(N):
    for _y in range(N):
        _i = _x * N + _y
        _X[_i] = _x
        _Y[_i] = _y
_NPTS = N * N


def _try_add(c, sel, dsets, in_set):
    """Try to add point c. Mutates sel/dsets/in_set on success. Returns (ok, ds)."""
    Xc = _X[c]
    Yc = _Y[c]
    seen = set()
    ds = []
    for i, p in enumerate(sel):
        dx = Xc - _X[p]
        dy = Yc - _Y[p]
        d = dx * dx + dy * dy
        if d in seen or d in dsets[i]:
            return False, None
        seen.add(d)
        ds.append(d)
    sel.append(c)
    dsets.append(seen)
    in_set.add(c)
    for i in range(len(sel) - 1):
        dsets[i].add(ds[i])
    return True, ds


def _complete(sel, dsets, in_set):
    """Repeatedly add any feasible point until none remain (maximal w.r.t. membership)."""
    while time.time() < DEADLINE:
        added = False
        for c in range(_NPTS):
            if c in in_set:
                continue
            ok, _ = _try_add(c, sel, dsets, in_set)
            if ok:
                added = True
                break
        if not added:
            break


def _remove_at(s_sel, s_dsets, s_in, rm):
    idx = s_sel[rm]
    for i in range(len(s_sel)):
        if i != rm:
            d = (_X[idx] - _X[s_sel[i]]) ** 2 + (_Y[idx] - _Y[s_sel[i]]) ** 2
            s_dsets[i].discard(d)
    s_sel.pop(rm)
    s_dsets.pop(rm)
    s_in.discard(idx)


def construct():
    rng = random.Random(20240607)
    best_sel = [0]
    best_dsets = [set()]
    best_in = {0}
    best_len = 1

    while time.time() < DEADLINE:
        # --- randomized greedy restart ---
        order = list(range(_NPTS))
        rng.shuffle(order)
        sel, dsets, in_set = [], [], set()
        for c in order:
            _try_add(c, sel, dsets, in_set)
        _complete(sel, dsets, in_set)
        if len(sel) > best_len:
            best_len = len(sel)
            best_sel, best_dsets, best_in = sel, dsets, in_set

        # --- local search: kick out a few points, re-complete, accept if larger ---
        if time.time() < DEADLINE:
            for _ in range(3):
                if time.time() >= DEADLINE:
                    break
                k = rng.randint(1, 3)
                s_sel = list(best_sel)
                s_dsets = [set(s) for s in best_dsets]
                s_in = set(best_in)
                for _r in range(k):
                    if not s_sel:
                        break
                    _remove_at(s_sel, s_dsets, s_in, rng.randrange(len(s_sel)))
                _complete(s_sel, s_dsets, s_in)
                if len(s_sel) > best_len:
                    best_len = len(s_sel)
                    best_sel, best_dsets, best_in = s_sel, s_dsets, s_in

    return [(_X[i], _Y[i]) for i in best_sel]
