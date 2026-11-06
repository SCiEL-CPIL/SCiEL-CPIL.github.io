import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 138.0  # Safety margin vs 150s hard kill

    def elapsed():
        return time.time() - start_time

    def build_paley(p):
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        adj = np.zeros((p, p), dtype=bool)
        for i in range(p):
            for j in range(i + 1, p):
                if ((j - i) % p) in squares:
                    adj[i, j] = True
                else:
                    adj[j, i] = True
        return p, adj

    def bitsets(n, adj):
        """outb[u] = bitmask of vertices u points to; incb[v] = bitmask of vertices pointing to v."""
        outb = np.zeros(n, dtype=np.uint64)
        incb = np.zeros(n, dtype=np.uint64)
        for u in range(n):
            m = 0
            row = adj[u]
            for v in range(n):
                if row[v]:
                    m |= (1 << v)
            outb[u] = m
        for v in range(n):
            m = 0
            col = adj[:, v]
            for u in range(n):
                if col[u]:
                    m |= (1 << u)
            incb[v] = m
        return outb, incb

    def check_and_critical(n, incb):
        """
        Returns (ok, critical). ok True iff every 4-set has a common dominator.
        critical[v] True iff v is the unique dominator of some 4-set.
        """
        critical = np.zeros(n, dtype=bool)
        ok = True
        for a in range(n - 3):
            ia = incb[a]
            for b in range(a + 1, n - 2):
                iab = ia & incb[b]
                for c in range(b + 1, n - 1):
                    iabc = iab & incb[c]
                    for d in range(c + 1, n):
                        dom = iabc & incb[d]
                        if dom == 0:
                            ok = False
                            return ok, critical
                        if dom & (dom - 1) == 0:
                            critical[dom.bit_length() - 1] = True
        return ok, critical

    def orientation_from_adj(adj):
        n = adj.shape[0]
        o = np.zeros(n * (n - 1) // 2, dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                o[idx] = 0 if adj[i, j] else 1
                idx += 1
        return o.tolist()

    # Start from Paley(67)
    p_init, full_adj = build_paley(67)
    if elapsed() > DEADLINE:
        return {'n': p_init, 'orientation': orientation_from_adj(full_adj)}

    # ---- Deterministic greedy removal ----
    active = list(range(p_init))
    best_n = p_init
    best_adj = full_adj.copy()

    while len(active) > 5 and elapsed() < DEADLINE:
        m = len(active)
        sub = full_adj[np.ix_(active, active)]
        _, incb = bitsets(m, sub)
        ok, critical = check_and_critical(m, incb)
        if not ok:
            break
        noncrit = [i for i in range(m) if not critical[i]]
        if not noncrit:
            break
        # Remove the single non-critical vertex with smallest out-degree (least "load")
        outdeg = sub.sum(axis=1)
        noncrit.sort(key=lambda i: outdeg[i])
        gi = active.pop(noncrit[0])
        if len(active) < best_n:
            best_n = len(active)
            best_adj = full_adj[np.ix_(active, active)].copy()

    if elapsed() > DEADLINE:
        return {'n': best_n, 'orientation': orientation_from_adj(best_adj)}

    # ---- Randomized multi-start removal to escape local minima ----
    for seed in range(100000):
        if elapsed() > DEADLINE:
            break
        rng = np.random.default_rng(seed)
        act = list(range(p_init))
        while len(act) > 5 and elapsed() < DEADLINE:
            m = len(act)
            sub = full_adj[np.ix_(act, act)]
            _, incb = bitsets(m, sub)
            ok, critical = check_and_critical(m, incb)
            if not ok:
                break
            noncrit = [i for i in range(m) if not critical[i]]
            if not noncrit:
                break
            # remove up to 3 random non-critical vertices
            k = min(3, len(noncrit))
            picks = rng.choice(noncrit, size=k, replace=False)
            for pi in picks:
                act.pop(int(pi))
            if len(act) < best_n:
                best_n = len(act)
                best_adj = full_adj[np.ix_(act, act)].copy()

    return {'n': best_n, 'orientation': orientation_from_adj(best_adj)}
