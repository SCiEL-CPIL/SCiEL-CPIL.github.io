import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    # ---------- Fast vectorized validity check (O(2^n)) ----------
    def is_valid(adj, n):
        """Return True iff no subset of size >= 6 induces a regular subgraph."""
        if n < 6:
            return True
        total = 1 << n
        # deg[v][m] = number of neighbors of v inside mask m
        deg = np.empty((n, total), dtype=np.int16)
        for v in range(n):
            d = np.zeros(total, dtype=np.int16)
            for b in range(n):
                if (adj[v] >> b) & 1:
                    d[1 << b:] += 1
            deg[v] = d
        # popcount for all masks (vectorized via byte table)
        pop = np.empty(total, dtype=np.int16)
        for i in range(total):
            pop[i] = i.bit_count()
        for size in range(6, n + 1):
            masks = np.nonzero(pop == size)[0]
            if masks.size == 0:
                continue
            sub = deg[:, masks]            # (n, len(masks))
            d0 = sub[0]
            eq = (sub == d0[None, :])      # broadcast
            if eq.all(axis=0).any():
                return False
        return True

    def to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # ---------- Local search to repair a near-valid graph ----------
    def local_search(adj, n, budget):
        cur = adj[:]
        cur_bad = None
        for _ in range(400):
            if time.time() > budget:
                break
            if cur_bad is None:
                if is_valid(cur, n):
                    return cur
                cur_bad = 1
            # try random single-edge flips, accept if it becomes valid
            for _ in range(30):
                if time.time() > budget:
                    break
                u = random.randrange(n)
                v = random.randrange(n)
                if u == v:
                    continue
                bit = 1 << v
                if cur[u] & bit:
                    cur[u] &= ~bit
                    cur[v] &= ~(1 << u)
                else:
                    cur[u] |= bit
                    cur[v] |= (1 << u)
                if is_valid(cur, n):
                    return cur
                # revert
                if cur[u] & bit:
                    cur[u] &= ~bit
                    cur[v] &= ~(1 << u)
                else:
                    cur[u] |= bit
                    cur[v] |= (1 << u)
        return cur

    # ---------- Deterministic algebraic constructions ----------
    def paley(p):
        qr = set()
        for x in range(1, p):
            if pow(x, (p - 1) // 2, p) == 1:
                qr.add(x)
        adj = [0] * p
        for i in range(p):
            for d in qr:
                j = (i + d) % p
                adj[i] |= 1 << j
                adj[j] |= 1 << i
        return adj

    def induced(adj, keep):
        n = len(keep)
        mp = {old: new for new, old in enumerate(keep)}
        out = [0] * n
        for old_i, ni in mp.items():
            for old_j, nj in mp.items():
                if old_i != old_j and (adj[old_i] >> old_j) & 1:
                    out[ni] |= 1 << nj
        return out

    # ---- Primary: Paley(23) with 2 vertices removed -> 21 vertices ----
    p = 23
    base = paley(p)
    for rm in itertools.combinations(range(p), 2):
        if time.time() > deadline - 30:
            break
        keep = [v for v in range(p) if v not in rm]
        cand = induced(base, keep)
        if is_valid(cand, 21):
            return {'n': 21, 'edges': to_edges(cand, 21)}
        # attempt repair
        if time.time() < deadline - 25:
            rep = local_search(cand, 21, deadline - 25)
            if is_valid(rep, 21):
                return {'n': 21, 'edges': to_edges(rep, 21)}

    # ---- Secondary: Paley(29) with 8 removed -> 21 vertices ----
    p2 = 29
    base2 = paley(p2)
    for rm in itertools.combinations(range(p2), 8):
        if time.time() > deadline - 20:
            break
        keep = [v for v in range(p2) if v not in rm]
        cand = induced(base2, keep)
        if is_valid(cand, 21):
            return {'n': 21, 'edges': to_edges(cand, 21)}
        if time.time() < deadline - 18:
            rep = local_search(cand, 21, deadline - 18)
            if is_valid(rep, 21):
                return {'n': 21, 'edges': to_edges(rep, 21)}

    # ---- Fallback: random search, largest n that is feasible ----
    rng = np.random.default_rng(12345)
    for n in range(21, 4, -1):
        # estimate budget per candidate
        per = 0.15 if n <= 18 else 0.08
        while time.time() < deadline - 8:
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        adj[i] |= 1 << j
                        adj[j] |= 1 << i
            if is_valid(adj, n):
                return {'n': n, 'edges': to_edges(adj, n)}
            if n <= 18 and time.time() < deadline - 12:
                rep = local_search(adj, n, deadline - 12)
                if is_valid(rep, n):
                    return {'n': n, 'edges': to_edges(rep, n)}

    # ---- Last resort: tiny valid graph ----
    return {'n': 5, 'edges': []}
