"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd
import numpy as np

def construct():
    start = time.time()
    DEADLINE = 58.0  # safety margin under 90s kill

    def elapsed():
        return time.time() - start

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    w = [1.0 / v for v in numbers]

    # ---- Build conflict graph (bitset masks) ----
    # conflict[i] is a Python int bitset of indices j that conflict with i (lcm<=256)
    conflict = [0] * n
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            g = gcd(a, b)
            if a * b // g <= N:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    # ---- Greedy + local search (warm start / polish) ----
    def is_compatible(idx, cur):
        m = conflict[idx]
        for c in cur:
            if (m >> c) & 1:
                return False
        return True

    def greedy():
        cur = set()
        for idx in sorted(range(n), key=lambda i: -w[i]):
            if is_compatible(idx, cur):
                cur.add(idx)
        return cur

    def score_of(cur):
        return sum(w[i] for i in cur)

    best = greedy()
    best_score = score_of(best)

    def local_search(cur):
        nonlocal best, best_score
        cur = set(cur)
        sc = score_of(cur)
        improved = True
        while improved and elapsed() < DEADLINE - 3.0:
            improved = False
            # add
            for idx in range(n):
                if idx in cur:
                    continue
                if is_compatible(idx, cur):
                    cur.add(idx); sc += w[idx]; improved = True
                    if sc > best_score:
                        best_score = sc; best = set(cur)
                    break
            if improved:
                continue
            # 1-for-1 swap
            for rem in list(cur):
                for add in range(n):
                    if add in cur:
                        continue
                    m = conflict[add]
                    ok = True
                    for c in cur:
                        if c == rem:
                            continue
                        if (m >> c) & 1:
                            ok = False; break
                    if ok and (w[add] - w[rem]) > 1e-12:
                        cur.discard(rem); cur.add(add); sc += w[add] - w[rem]
                        improved = True
                        if sc > best_score:
                            best_score = sc; best = set(cur)
                        break
                if improved:
                    break
        return cur, sc

    best, best_score = local_search(best)

    # ---- Branch and Bound for Weighted MIS ----
    # Order nodes by weight descending for tight LP bounds
    order = sorted(range(n), key=lambda i: -w[i])
    # map old index -> position in order; we'll work in "order" space
    pos = [0] * n
    for p, i in enumerate(order):
        pos[i] = p
    wo = [w[i] for i in order]
    # reindex conflict masks into order-space
    co = [0] * n
    for p in range(n):
        old = order[p]
        m = conflict[old]
        nm = 0
        for q in range(n):
            if (m >> order[q]) & 1:
                nm |= (1 << q)
        co[p] = nm

    # Bitset helpers in order-space
    def bitcount(x):
        return bin(x).count("1")

    # LP relaxation bound: max sum wo[i] x_i s.t. x_i + x_j <= 1 (edges), 0<=x<=1.
    # We solve the dual: min sum y_e s.t. for each vertex i: sum_{e incident to i} y_e >= wo[i], y>=0.
    # Use a simple coordinate-descent / subgradient-free iterative solver (fast, local).
    def lp_bound(cand_mask):
        # vertices present
        verts = []
        m = cand_mask
        while m:
            lb = m & (-m)
            p = lb.bit_length() - 1
            verts.append(p)
            m ^= lb
        k = len(verts)
        if k == 0:
            return 0.0
        if k == 1:
            return wo[verts[0]]
        # Build edge list among verts
        edges = []
        vset = set(verts)
        for a in verts:
            m2 = co[a] & cand_mask
            while m2:
                lb = m2 & (-m2)
                b = lb.bit_length() - 1
                if b > a:
                    edges.append((a, b))
                m2 ^= lb
        E = len(edges)
        if E == 0:
            return sum(wo[v] for v in verts)
        # dual variables
        y = np.zeros(E)
        # vertex -> incident edge indices
        inc = {v: [] for v in verts}
        for ei, (a, b) in enumerate(edges):
            inc[a].append(ei)
            inc[b].append(ei)
        # iterative: set y_e = max(0, (wo[a]-sum_other_a)/1 ...) use simple scaling
        # Use a projected subgradient on dual feasibility, minimize sum y
        # Simpler: multiplicative update for feasibility then shrink.
        for _ in range(60):
            # compute slack per vertex
            s = np.zeros(k)
            idxmap = {v: ii for ii, v in enumerate(verts)}
            for ii, v in enumerate(verts):
                tot = 0.0
                for ei in inc[v]:
                    tot += y[ei]
                s[ii] = tot
            # for each edge, needed = max(0, max(wo[a]-s[a], wo[b]-s[b]))
            for ei, (a, b) in enumerate(edges):
                ia = idxmap[a]; ib = idxmap[b]
                need = max(0.0, wo[a] - s[ia], wo[b] - s[ib])
                if need > 0:
                    y[ei] = max(y[ei], need)
            # shrink toward feasibility-minimizing by scaling down edges that are "over"
            # recompute
            for ii, v in enumerate(verts):
                tot = 0.0
                for ei in inc[v]:
                    tot += y[ei]
                s[ii] = tot
            # edges where both endpoints over-satisfied -> reduce
            for ei, (a, b) in enumerate(edges):
                ia = idxmap[a]; ib = idxmap[b]
                over = min(s[ia] - wo[a], s[ib] - wo[b])
                if over > 0:
                    y[ei] = max(0.0, y[ei] - over)
        return float(np.sum(y))

    # Branch and bound
    full_mask = (1 << n) - 1
    # We'll do DFS with fractional bound, branching on a vertex.
    # To keep it fast, use a stack of (cand_mask, forced_sum) but we use bound directly.
    # Prune: if lp_bound(cand) + 0 <= best_score -> skip.
    # Branch: pick vertex v with max weight in cand. Branch include(v) and exclude(v).

    # Use a priority: explore include-first to find good solutions quickly.
    stack = [(full_mask, 0)]
    nodes_explored = 0
    MAX_NODES = 400000

    while stack and elapsed() < DEADLINE - 2.0 and nodes_explored < MAX_NODES:
        cand, _ = stack.pop()
        nodes_explored += 1
        if cand == 0:
            continue
        # quick upper bound: sum of weights in cand (loose) then LP
        # loose bound first (cheap)
        loose = 0.0
        m = cand
        while m:
            lb = m & (-m)
            p = lb.bit_length() - 1
            loose += wo[p]
            m ^= lb
        if loose <= best_score + 1e-12:
            continue
        bound = lp_bound(cand)
        if bound <= best_score + 1e-9:
            continue
        # choose branch vertex: max weight in cand
        m = cand
        v = -1
        bestw = -1.0
        while m:
            lb = m & (-m)
            p = lb.bit_length() - 1
            if wo[p] > bestw:
                bestw = wo[p]; v = p
            m ^= lb
        if v < 0:
            continue
        # include v: new cand = cand & ~co[v] & ~(1<<v); gain wo[v]
        # exclude v: new cand = cand & ~(1<<v)
        # Push exclude first so include is explored next (LIFO)
        stack.append((cand & ~(1 << v), 0))
        stack.append((cand & ~(1 << v) & ~co[v], wo[v]))

    # Refine with local search from best
    best, best_score = local_search(best)

    result = [numbers[i] for i in best]
    if not result:
        result = [256]

    # Final validation
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.lcm(result[i], result[j]) <= N:
                # invalid: fall back to greedy
                g = [numbers[i] for i in greedy()]
                return g if g else [256]
    return result
# EVOLVE-BLOCK-END
