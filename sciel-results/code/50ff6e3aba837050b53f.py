"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 320
    rng = random.Random(20240601)

    xs = np.arange(N, dtype=np.int64)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    M = pts.shape[0]

    class State:
        __slots__ = ("sel", "ds", "nbr")
        def __init__(self):
            self.sel = set()
            self.ds = {}   # idx -> set of distances
            self.nbr = {}  # idx -> dict {dist: neighbor}

        def size(self):
            return len(self.sel)

        def _dist(self, a, b):
            return int(((pts[a] - pts[b]) ** 2).sum())

        def conflicts_for(self, cand):
            if not self.sel:
                return []
            sel = list(self.sel)
            sel_pts = pts[sel]
            p = pts[cand]
            d2 = ((sel_pts - p) ** 2).sum(axis=1)
            d2l = d2.tolist()
            if len(set(d2l)) != len(d2l):
                return None
            conf = []
            for i, q in enumerate(sel):
                if d2l[i] in self.ds[q]:
                    conf.append((i, q))
            return conf

        def add(self, cand):
            self.sel.add(cand)
            self.ds[cand] = set()
            self.nbr[cand] = {}
            if self.sel != {cand}:
                sel = list(self.sel)
                sel.remove(cand)
                sel_pts = pts[sel]
                p = pts[cand]
                d2 = ((sel_pts - p) ** 2).sum(axis=1)
                for i, q in enumerate(sel):
                    d = int(d2[i])
                    self.ds[cand].add(d)
                    self.nbr[cand][d] = q
                    self.ds[q].add(d)
                    self.nbr[q][d] = cand

        def remove(self, cand):
            self.sel.discard(cand)
            if cand in self.nbr:
                for d, q in self.nbr[cand].items():
                    self.ds[q].discard(d)
                    self.nbr[q].pop(d, None)
                del self.ds[cand]
                del self.nbr[cand]

    def greedy(order, st):
        for cand in order:
            if cand in st.sel:
                continue
            conf = st.conflicts_for(cand)
            if conf is not None and not conf:
                st.add(cand)

    def local_search(st, budget):
        best = st.size()
        best_sel = set(st.sel)
        cands = [i for i in range(M) if i not in st.sel]
        rng.shuffle(cands)
        ci = 0
        while time.time() < budget:
            if ci >= len(cands):
                cands = [i for i in range(M) if i not in st.sel]
                if not cands:
                    break
                rng.shuffle(cands)
                ci = 0
            cand = cands[ci]; ci += 1
            if cand in st.sel:
                continue
            conf = st.conflicts_for(cand)
            if conf is None:
                continue
            if not conf:
                st.add(cand)
                if st.size() > best:
                    best = st.size()
                    best_sel = set(st.sel)
                if cand in cands:
                    cands.remove(cand)
            else:
                for (i, q) in conf:
                    if time.time() >= budget:
                        break
                    saved_sel = set(st.sel)
                    saved_ds = {k: set(v) for k, v in st.ds.items()}
                    saved_nbr = {k: dict(v) for k, v in st.nbr.items()}
                    st.remove(q)
                    conf2 = st.conflicts_for(cand)
                    if conf2 is not None and not conf2:
                        st.add(cand)
                        if st.size() > best:
                            best = st.size()
                            best_sel = set(st.sel)
                    else:
                        st.sel = saved_sel
                        st.ds = saved_ds
                        st.nbr = saved_nbr
                    break
        if st.size() < best:
            st.sel = set(best_sel)
            st.ds = {i: set() for i in st.sel}
            st.nbr = {i: {} for i in st.sel}
            sel = list(st.sel)
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    a, b = sel[i], sel[j]
                    d = st._dist(a, b)
                    st.ds[a].add(d); st.nbr[a][d] = b
                    st.ds[b].add(d); st.nbr[b][d] = a
        return st

    all_points = list(range(M))
    global_best_sel = set()

    n_restarts = 80
    for r in range(n_restarts):
        if time.time() > deadline - 120:
            break
        st = State()
        order = all_points[:]
        rng.shuffle(order)
        greedy(order, st)
        st = local_search(st, min(deadline - 90, time.time() + 10))
        if st.size() > len(global_best_sel):
            global_best_sel = set(st.sel)

    if time.time() < deadline - 15:
        stf = State()
        stf.sel = set(global_best_sel)
        stf.ds = {i: set() for i in stf.sel}
        stf.nbr = {i: {} for i in stf.sel}
        sel = list(stf.sel)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                a, b = sel[i], sel[j]
                d = stf._dist(a, b)
                stf.ds[a].add(d); stf.nbr[a][d] = b
                stf.ds[b].add(d); stf.nbr[b][d] = a
        stf = local_search(stf, deadline - 5)
        if stf.size() > len(global_best_sel):
            global_best_sel = set(stf.sel)

    return [tuple(int(v) for v in pts[i]) for i in global_best_sel]
# EVOLVE-BLOCK-END
