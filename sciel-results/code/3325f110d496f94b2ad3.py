import random
import time

N = 18
FULL = (1 << N) - 1


def _is_k4_free(adj):
    """Bitmask K4-freeness: for each edge (u,v), the common-neighbor set must
    be an independent set (no edge inside it), else a K4 exists."""
    for u in range(N):
        au = adj[u]
        m = au
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if v <= u:
                continue
            common = au & adj[v] & ~((1 << u) | (1 << v))
            cm = common
            while cm:
                cb = cm & -cm
                a = cb.bit_length() - 1
                cm ^= cb
                if adj[a] & common:
                    return False
    return True


def _max_triangle_free(adj, deadline=None):
    """Largest vertex set inducing a triangle-free subgraph, via bit-parallel
    branch and bound. Returns (value, found_exact)."""
    n = N
    best = [0]
    start = time.time()

    def rec(cand, sel):
        if best[0] >= n:
            return
        if not cand:
            if sel > best[0]:
                best[0] = sel
            return
        if sel + cand.bit_count() <= best[0]:
            return
        if deadline and (time.time() - start) > deadline:
            return

        # Try to include a vertex that is not adjacent to any selected vertex.
        ok = cand
        sm = sel
        while sm:
            sb = sm & -sm
            s = sb.bit_length() - 1
            sm ^= sb
            ok &= ~adj[s]
        if ok:
            lb = ok & -ok
            v = lb.bit_length() - 1
            rec(cand & ~lb, sel | lb)

        # Exclude the lowest candidate vertex.
        lb = cand & -cand
        rec(cand & ~lb, sel)

    rec(FULL, 0)
    exact = (time.time() - start) < (deadline or 1e9)
    return best[0], exact


def _edges_to_adj(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


def _adj_to_edges(adj):
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges


def _turan_adj():
    """Turan T(18,3): three parts of 6, all cross edges present (K4-free)."""
    adj = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            pi = 0 if i < 6 else (1 if i < 12 else 2)
            pj = 0 if j < 6 else (1 if j < 12 else 2)
            if pi != pj:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj


def _part(i):
    return 0 if i < 6 else (1 if i < 12 else 2)


def construct():
    start = time.time()
    deadline = start + 112  # safety margin under the 150s kill

    def now():
        return time.time()

    # ---- Seed: Turan T(18,3) (K4-free), score 12 ----
    adj = _turan_adj()
    best_l, _ = _max_triangle_free(adj, 30)
    best_adj = adj[:]
    temperature = 1.0
    cooling = 0.99
    min_temp = 0.02

    def accept(l):
        nonlocal best_l, best_adj, temperature
        if l < best_l:
            best_l = l
            best_adj = adj[:]
            temperature = max(temperature * cooling, min_temp)
            return True
        delta = l - best_l
        if delta == 0:
            return random.random() < 0.5
        if random.random() < min(1.0, temperature / delta):
            temperature = max(temperature * cooling, min_temp)
            return True
        return False

    def evaluate():
        l, _ = _max_triangle_free(adj, deadline - now() - 1.0)
        return l

    def inter_edges():
        out = []
        for i in range(N):
            ai = adj[i]
            for j in range(i + 1, N):
                if (ai >> j) & 1 and _part(i) != _part(j):
                    out.append((i, j))
        return out

    def intra_absent():
        out = []
        for i in range(N):
            ai = adj[i]
            for j in range(i + 1, N):
                if not ((ai >> j) & 1) and _part(i) == _part(j):
                    out.append((i, j))
        return out

    def all_edges():
        out = []
        for i in range(N):
            ai = adj[i]
            for j in range(i + 1, N):
                if (ai >> j) & 1:
                    out.append((i, j))
        return out

    def all_non_edges():
        out = []
        for i in range(N):
            ai = adj[i]
            for j in range(i + 1, N):
                if not ((ai >> j) & 1):
                    out.append((i, j))
        return out

    def flip(u, v, add):
        if add:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

    while now() < deadline - 4:
        mt = random.randint(0, 4)

        if mt == 0:
            # structured swap: drop a cross edge, add an intra edge (keep density)
            ie = inter_edges()
            ia = intra_absent()
            if not ie or not ia:
                continue
            u, v = random.choice(ie)
            x, y = random.choice(ia)
            flip(u, v, False)
            flip(x, y, True)
            if _is_k4_free(adj):
                if not accept(evaluate()):
                    flip(u, v, True)
                    flip(x, y, False)
            else:
                flip(u, v, True)
                flip(x, y, False)

        elif mt == 1:
            # single edge removal
            ae = all_edges()
            if not ae:
                continue
            u, v = random.choice(ae)
            flip(u, v, False)
            if not accept(evaluate()):
                flip(u, v, True)

        elif mt == 2:
            # single edge addition, K4-guarded (raises density)
            ane = all_non_edges()
            if not ane:
                continue
            u, v = random.choice(ane)
            flip(u, v, True)
            if _is_k4_free(adj):
                if not accept(evaluate()):
                    flip(u, v, False)
            else:
                flip(u, v, False)

        elif mt == 3:
            # double swap
            ie = inter_edges()
            ia = intra_absent()
            if len(ie) < 2 or len(ia) < 2:
                continue
            (u1, v1), (u2, v2) = random.sample(ie, 2)
            (x1, y1), (x2, y2) = random.sample(ia, 2)
            if len({(u1, v1), (u2, v2), (x1, y1), (x2, y2)}) < 4:
                continue
            flip(u1, v1, False); flip(u2, v2, False)
            flip(x1, y1, True); flip(x2, y2, True)
            if _is_k4_free(adj):
                if not accept(evaluate()):
                    flip(u1, v1, True); flip(u2, v2, True)
                    flip(x1, y1, False); flip(x2, y2, False)
            else:
                flip(u1, v1, True); flip(u2, v2, True)
                flip(x1, y1, False); flip(x2, y2, False)

        else:
            # random restart from a perturbed Turan graph, K4-repaired
            adj = _turan_adj()
            # randomise by removing some cross edges
            for (u, v) in inter_edges():
                if random.random() < 0.15:
                    flip(u, v, False)
            # add a few random intra edges
            for _ in range(8):
                ia = intra_absent()
                if not ia:
                    break
                x, y = random.choice(ia)
                flip(x, y, True)
                if not _is_k4_free(adj):
                    flip(x, y, False)
            if not _is_k4_free(adj):
                continue
            l, _ = _max_triangle_free(adj, deadline - now() - 1.0)
            if l < best_l:
                best_l = l
                best_adj = adj[:]
                temperature = 1.0

    return _adj_to_edges(best_adj)
