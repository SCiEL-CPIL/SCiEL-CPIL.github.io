"""Rewritten program: maximize edges of a 20-vertex graph, 20 colours, every 4-cycle rainbow."""
import random
import time

N = 20
K = 20
DEADLINE = 140.0  # safety margin under 150s wall-clock

# Precompute, for every vertex pair (u,v), the list of common neighbours.
# Adding edge (u,v) with colour c is valid iff no vertex w in common(u,v)
# already has an incident edge of colour c (i.e. c not in col_at[w]).
_COMMON = [[[] for _ in range(N)] for _ in range(N)]
for u in range(N):
    for v in range(u + 1, N):
        _COMMON[u][v] = [w for w in range(N) if w != u and w != v
                         and (w in _COMMON[u][v] or True) and
                         (w not in (u, v))]
# build properly
_COMMON = [[[] for _ in range(N)] for _ in range(N)]
for u in range(N):
    for v in range(u + 1, N):
        lst = []
        for w in range(N):
            if w == u or w == v:
                continue
            lst.append(w)
        _COMMON[u][v] = lst
        _COMMON[v][u] = lst


class State:
    __slots__ = ('edge', 'color', 'col_at')

    def __init__(self):
        self.edge = {}
        self.color = {}
        self.col_at = [set() for _ in range(N)]

    def add(self, u, v, c):
        self.edge[(u, v)] = c
        self.color[(u, v)] = c
        self.col_at[u].add(c)
        self.col_at[v].add(c)

    def remove(self, u, v):
        c = self.edge.pop((u, v))
        self.color.pop((u, v))
        self.col_at[u].discard(c)
        self.col_at[v].discard(c)
        return c

    def recolor(self, u, v, c):
        old = self.edge[(u, v)]
        if old == c:
            return True
        self.col_at[u].discard(old)
        self.col_at[v].discard(old)
        self.col_at[u].add(c)
        self.col_at[v].add(c)
        self.edge[(u, v)] = c
        self.color[(u, v)] = c
        return True

    def valid_add(self, u, v, c):
        for w in _COMMON[u][v]:
            if c in self.col_at[w]:
                return False
        return True

    def count(self):
        return len(self.edge)


def greedy_build(rng):
    st = State()
    pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng.shuffle(pairs)
    for u, v in pairs:
        # choose a colour not present at u or v, and valid
        used = st.col_at[u] | st.col_at[v]
        candidates = [c for c in range(K) if c not in used and st.valid_add(u, v, c)]
        if not candidates:
            continue
        # pick the colour with lowest total usage to keep spread
        c = min(candidates, key=lambda cc: sum(1 for (a, b) in st.edge if st.edge[(a, b)] == cc))
        st.add(u, v, c)
    return st


def try_add_edges(st, rng, budget):
    improved = True
    while improved and time.time() < budget:
        improved = False
        pairs = [(u, v) for u in range(N) for v in range(u + 1, N) if (u, v) not in st.edge]
        rng.shuffle(pairs)
        for u, v in pairs:
            if time.time() > budget:
                break
            used = st.col_at[u] | st.col_at[v]
            candidates = [c for c in range(K) if c not in used and st.valid_add(u, v, c)]
            if not candidates:
                continue
            c = min(candidates, key=lambda cc: sum(1 for (a, b) in st.edge if st.edge[(a, b)] == cc))
            st.add(u, v, c)
            improved = True
    return st


def recolor_pass(st, rng, budget):
    improved = True
    while improved and time.time() < budget:
        improved = False
        pairs = [(u, v) for u in range(N) for v in range(u + 1, N) if (u, v) not in st.edge]
        rng.shuffle(pairs)
        for u, v in pairs:
            if time.time() > budget:
                break
            used_uv = st.col_at[u] | st.col_at[v]
            # colours that fail only due to a common-vertex constraint
            bad = {}
            for c in range(K):
                if c in used_uv:
                    continue
                for w in _COMMON[u][v]:
                    if c in st.col_at[w]:
                        bad.setdefault(c, set()).add(w)
            # try to recolor one blocking edge per candidate colour
            for c, blockers in bad.items():
                if not blockers:
                    continue
                # attempt: find a blocker edge (w,x) of colour c and recolor it
                ok = False
                for w in blockers:
                    # find an edge (w, x) with colour c
                    for x in range(N):
                        if x == w:
                            continue
                        e = (min(w, x), max(w, x))
                        if st.edge.get(e) == c:
                            # recolor (w,x) to a new colour d
                            used_wx = st.col_at[w] | st.col_at[x]
                            d_candidates = [d for d in range(K) if d not in used_wx and st.valid_add(w, x, d)]
                            if d_candidates:
                                d = min(d_candidates, key=lambda dd: sum(1 for (a, b) in st.edge if st.edge[(a, b)] == dd))
                                st.recolor(w, x, d)
                                # now (u,v) may be addable with c
                                if st.valid_add(u, v, c):
                                    st.add(u, v, c)
                                    improved = True
                                    ok = True
                                    break
                    if ok:
                        break
                if ok:
                    break
    return st


def remove_rebuild(st, rng, budget):
    # remove a random subset of edges, then re-greedy the freed pairs
    if not st.edge:
        return st
    n_remove = max(3, len(st.edge) // 8)
    keys = list(st.edge.keys())
    rng.shuffle(keys)
    removed = keys[:n_remove]
    saved = []
    for e in removed:
        c = st.remove(e[0], e[1])
        saved.append((e[0], e[1], c))
    # re-add greedily
    pairs = [(u, v) for (u, v, c) in saved]
    rng.shuffle(pairs)
    for u, v in pairs:
        used = st.col_at[u] | st.col_at[v]
        candidates = [c for c in range(K) if c not in used and st.valid_add(u, v, c)]
        if candidates:
            c = min(candidates, key=lambda cc: sum(1 for (a, b) in st.edge if st.edge[(a, b)] == cc))
            st.add(u, v, c)
    return st


def polish(st, budget):
    # final add pass
    try_add_edges(st, random.Random(0), budget)
    return st


def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = None
    best_count = -1
    for seed in range(0, 2000000):
        if time.time() > deadline - 8.0:
            break
        rng = random.Random(seed)
        st = greedy_build(rng)
        try_add_edges(st, rng, deadline - 6.0)
        recolor_pass(st, rng, deadline - 6.0)
        # occasional remove-rebuild
        if seed % 3 == 0:
            st = remove_rebuild(st, rng, deadline - 6.0)
            try_add_edges(st, rng, deadline - 6.0)
        cnt = st.count()
        if cnt > best_count:
            best_count = cnt
            best = st
            if best_count >= 200:
                break
        if time.time() > deadline - 8.0:
            break
    if best is None:
        # fallback: a star
        return [[0, v, 0] for v in range(1, N)]
    # final polish
    polish(best, deadline - 2.0)
    edges = [[u, v, c] for (u, v), c in best.edge.items()]
    return edges
