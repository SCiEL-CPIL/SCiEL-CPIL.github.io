"""Rewritten solver: hypergraph independent set via conflict-count local search."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd

    t0 = time.time()
    DEADLINE = t0 + 80.0          # safety margin before the 90s kill
    N = 128
    ALL = list(range(1, N + 1))

    # ---- gcd table ----
    g = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        gi = g[i]
        for j in range(i, N + 1):
            v = gcd(i, j)
            gi[j] = v
            g[j][i] = v

    # ---- bad triples: (a<b<c) with all three pairwise gcds equal ----
    bad_with = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        ga = g[a]
        for b in range(a + 1, N + 1):
            gab = ga[b]
            gb = g[b]
            for c in range(b + 1, N + 1):
                if ga[c] == gab and gb[c] == gab:
                    bad_with[a].append((b, c))
                    bad_with[b].append((a, c))
                    bad_with[c].append((a, b))

    degree = [len(bad_with[i]) for i in range(N + 1)]
    bw = bad_with

    class State:
        __slots__ = ("inS", "block", "size")
        def __init__(self):
            self.inS = [False] * (N + 1)
            self.block = [0] * (N + 1)
            self.size = 0

    def fill(st, order, max_passes=40):
        ins = st.inS; bl = st.block; sz = [st.size]
        for _ in range(max_passes):
            changed = False
            for x in order:
                if not ins[x] and bl[x] == 0:
                    ins[x] = True
                    sz[0] += 1
                    for a, b in bw[x]:
                        bl[a] += 1
                        bl[b] += 1
                    changed = True
            st.size = sz[0]
            if not changed:
                break
            if time.time() > DEADLINE:
                break

    def try_remove_set(st, removals, order):
        ins = st.inS; bl = st.block
        snap_ins = ins.copy(); snap_bl = bl.copy(); old = st.size
        for r in removals:
            ins[r] = False
            st.size -= 1
            for a, b in bw[r]:
                bl[a] -= 1
                bl[b] -= 1
        fill(st, order)
        if st.size > old:
            return True
        ins[:] = snap_ins; bl[:] = snap_bl; st.size = old
        return False

    def improve(st):
        ins = st.inS
        while time.time() < DEADLINE:
            progressed = False
            cur = [i for i in ALL if ins[i]]
            if not cur:
                break
            random.shuffle(cur)
            for r in cur:
                if time.time() > DEADLINE:
                    break
                order = ALL[:]; random.shuffle(order)
                if try_remove_set(st, [r], order):
                    progressed = True
                    break
            if not progressed:
                # 2-removal refill to escape harder local optima
                for _ in range(25):
                    if time.time() > DEADLINE:
                        break
                    cur = [i for i in ALL if ins[i]]
                    if len(cur) < 2:
                        break
                    r1, r2 = random.sample(cur, 2)
                    order = ALL[:]; random.shuffle(order)
                    if try_remove_set(st, [r1, r2], order):
                        progressed = True
                        break
            if not progressed:
                break

    best_inS = None
    best_size = 0

    def record(st):
        nonlocal best_inS, best_size
        if st.size > best_size:
            best_size = st.size
            best_inS = st.inS.copy()

    restart = 0
    while time.time() < DEADLINE:
        st = State()
        strat = restart % 3
        order = ALL[:]
        if strat == 0:
            order.sort(key=lambda x: degree[x])          # safest first
        elif strat == 1:
            random.shuffle(order)                          # random
        else:
            random.shuffle(order)
            # seed with all degree-0 elements (always addable)
            for x in ALL:
                if degree[x] == 0:
                    st.inS[x] = True; st.size += 1
                    for a, b in bw[x]:
                        st.block[a] += 1; st.block[b] += 1
        fill(st, order)
        record(st)
        improve(st)
        record(st)
        restart += 1

    if best_inS is None or best_size == 0:
        return [1]
    return [i for i in ALL if best_inS[i]]
# EVOLVE-BLOCK-END
