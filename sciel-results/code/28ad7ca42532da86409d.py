import numpy as np
import time
import random

def count_edges(adj):
    return np.sum(adj != -1) // 2

def construct():
    n = 20
    start = time.time()
    deadline = start + 146.0

    # ---------- Precompute all 4-cycles as index arrays ----------
    # A 4-cycle is (u,v,w,x) with edges (u,v),(v,w),(w,x),(x,u).
    # All distinct 4-cycles in K_n: choose 4 vertices, 3 pairings.
    idx_u, idx_v, idx_w, idx_x = [], [], [], []
    for a in range(n):
        for b in range(n):
            if b == a:
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                for d in range(n):
                    if d == a or d == b or d == c:
                        continue
                    # canonical: smallest vertex = a, and a->b is the
                    # first edge from a in the cycle (b < d) to dedupe
                    if b < d:
                        idx_u.append(a); idx_v.append(b)
                        idx_w.append(c); idx_x.append(d)
    cu = np.asarray(idx_u); cv = np.asarray(idx_v)
    cw = np.asarray(idx_w); cx = np.asarray(idx_x)
    M = len(cu)

    # Adjacency index arrays for fast edge lookup
    eu = np.array([i for i in range(n) for j in range(i+1, n)])
    ev = np.array([j for i in range(n) for j in range(i+1, n)])
    E = len(eu)
    pair_to_idx = {}
    for k in range(E):
        pair_to_idx[(int(eu[k]), int(ev[k]))] = k
    # reverse lookup for arbitrary (a,b)
    def pidx(a, b):
        if a > b:
            a, b = b, a
        return pair_to_idx[(a, b)]

    def get_conflicts(col, present):
        """Vectorized: return set of cycle-indices that are present & non-rainbow."""
        c1 = col[cu, cv]
        c2 = col[cv, cw]
        c3 = col[cw, cx]
        c4 = col[cx, cu]
        present_mask = (c1 >= 0) & (c2 >= 0) & (c3 >= 0) & (c4 >= 0)
        if not present_mask.any():
            return np.array([], dtype=int)
        c1 = np.where(present_mask, c1, 0)
        c2 = np.where(present_mask, c2, 0)
        c3 = np.where(present_mask, c3, 0)
        c4 = np.where(present_mask, c4, 0)
        bad = (c1 == c2) | (c1 == c3) | (c1 == c4) | (c2 == c3) | (c2 == c4) | (c3 == c4)
        bad &= present_mask
        return np.where(bad)[0]

    def edge_cycle_counts(col, present):
        """For each edge, count present non-rainbow cycles it belongs to."""
        conf = get_conflicts(col, present)
        if len(conf) == 0:
            return np.zeros(E, dtype=int)
        counts = np.zeros(E, dtype=int)
        for ci in conf:
            counts[pidx(cu[ci], cv[ci])] += 1
            counts[pidx(cv[ci], cw[ci])] += 1
            counts[pidx(cw[ci], cx[ci])] += 1
            counts[pidx(cx[ci], cu[ci])] += 1
        return counts

    def repair(col, present, rng):
        """Remove edges (highest conflict count) until no non-rainbow cycle remains."""
        steps = 0
        while True:
            if time.time() > deadline:
                return col, present, False
            conf = get_conflicts(col, present)
            if len(conf) == 0:
                return col, present, True
            counts = edge_cycle_counts(col, present)
            cand = np.where(present)[0]
            cc = counts[cand]
            if cc.max() <= 0:
                # conflicts exist but no edge counted? fallback
                cand = np.where((col[eu, ev] >= 0))[0]
                if len(cand) == 0:
                    return col, present, True
            # pick among top-conflict edges, with a little randomness
            top = cc.max()
            top_edges = cand[cc == top]
            k = int(rng.choice(top_edges))
            col[eu[k], ev[k]] = -1
            col[ev[k], eu[k]] = -1
            present[k] = False
            steps += 1
            if steps > 2000:
                return col, present, False
        return col, present, False

    def grow(col, present, rng):
        """Try to add missing edges back with a valid color."""
        improved = True
        guard = 0
        while improved and guard < 20:
            if time.time() > deadline:
                break
            improved = False
            guard += 1
            missing = np.where(~present)[0]
            if len(missing) == 0:
                break
            order = missing.copy()
            rng.shuffle(order.tolist())
            for k in order:
                if time.time() > deadline:
                    break
                a = int(eu[k]); b = int(ev[k])
                # find a color that keeps all 4-cycles through (a,b) rainbow
                # forbidden colors come from the other 3 edges of each present 4-cycle
                forbidden = set()
                # cycles where (a,b) is the (u,v) edge: need (b,c),(c,d),(d,a) present
                for c in range(n):
                    if c == a or c == b: continue
                    if col[b, c] < 0: continue
                    for d in range(n):
                        if d in (a, b, c): continue
                        if col[c, d] < 0 or col[d, a] < 0: continue
                        forbidden.add(col[b, c]); forbidden.add(col[c, d]); forbidden.add(col[d, a])
                # (a,b) as (v,w): need (u,a),(u? ) ... symmetric; handle via (b,a) ordering
                # Since (a,b) with a<b, treat as edge; the 4-cycle can place it in any of 4 slots.
                # We cover all by iterating which two other vertices form the path.
                # Path endpoints a,b; the other two vertices p,q with edges a-p, p-q, q-b
                for p in range(n):
                    if p in (a, b): continue
                    if col[a, p] < 0: continue
                    for q in range(n):
                        if q in (a, b, p): continue
                        if col[p, q] < 0 or col[q, b] < 0: continue
                        forbidden.add(col[a, p]); forbidden.add(col[p, q]); forbidden.add(col[q, b])
                avail = [c for c in range(n) if c not in forbidden]
                if avail:
                    c = rng.choice(avail)
                    col[a, b] = c; col[b, a] = c
                    present[k] = True
                    improved = True

    def recolor_search(col, present, rng, iters):
        """Local search: try recoloring an edge to reduce conflicts, then grow."""
        for _ in range(iters):
            if time.time() > deadline:
                break
            conf = get_conflicts(col, present)
            if len(conf) == 0:
                break
            # recolor a random edge involved in a conflict
            counts = edge_cycle_counts(col, present)
            cand = np.where(present)[0]
            cc = counts[cand]
            if cc.max() <= 0:
                break
            top = cc.max()
            top_edges = cand[cc == top]
            k = int(rng.choice(top_edges))
            a = int(eu[k]); b = int(ev[k])
            old = col[a, b]
            colors = list(range(n)); rng.shuffle(colors)
            best_new = None; best_conf = len(conf)
            for nc in colors:
                if nc == old: continue
                col[a, b] = nc; col[b, a] = nc
                nc_conf = len(get_conflicts(col, present))
                if nc_conf < best_conf:
                    best_conf = nc_conf; best_new = nc
                col[a, b] = old; col[b, a] = old
            if best_new is not None:
                col[a, b] = best_new; col[b, a] = best_new
            else:
                # no recolor helps; break to avoid spinning
                break

    best_count = 0
    best_col = None

    base_seed = 12345
    restart = 0
    while True:
        if time.time() > deadline - 8.0:
            break
        rng = random.Random(base_seed + restart)
        # start from complete graph with random colors
        col = np.full((n, n), -1, dtype=int)
        present = np.ones(E, dtype=bool)
        for k in range(E):
            c = rng.randint(0, n - 1)
            col[eu[k], ev[k]] = c
            col[ev[k], eu[k]] = c
        # repair
        col, present, ok = repair(col, present, rng)
        if not ok:
            restart += 1
            continue
        # light recolor to free up capacity, then grow
        recolor_search(col, present, rng, 30)
        grow(col, present, rng)
        # repair again if grow introduced issues (it shouldn't, but be safe)
        col, present, ok2 = repair(col, present, rng)
        cnt = count_edges(col)
        if cnt > best_count:
            best_count = cnt
            best_col = col.copy()
        restart += 1

    # Final polish on best: aggressive grow
    if best_col is not None and time.time() < deadline - 3.0:
        col = best_col.copy()
        present = np.ones(E, dtype=bool)
        for k in range(E):
            if col[eu[k], ev[k]] < 0:
                present[k] = False
        rng = random.Random(999999)
        grow(col, present, rng)
        col, present, _ = repair(col, present, rng)
        cnt = count_edges(col)
        if cnt > best_count:
            best_count = cnt
            best_col = col

    # Build result
    res = []
    if best_col is None:
        # fallback: spanning tree (always valid)
        for v in range(1, n):
            res.append([0, v, v % n])
        return res
    for k in range(E):
        if best_col[eu[k], ev[k]] >= 0:
            res.append([int(eu[k]), int(ev[k]), int(best_col[eu[k], ev[k]])])
    return res
