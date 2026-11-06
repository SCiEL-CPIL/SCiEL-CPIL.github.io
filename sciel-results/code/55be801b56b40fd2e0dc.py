"""Rewritten program: polarity-graph base + vectorized C4 + SA local search."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 138.0  # safety margin before 150s kill

    n = 31
    target = 91

    # ---------- C4 counting (vectorized) ----------
    def count_c4(A):
        """Number of (not necessarily induced) 4-cycles = sum_{u<v} C(A[u,v],2) / 2."""
        S = (A @ A).astype(np.int64)
        iu = np.triu_indices(n, k=1)
        vals = S[iu]
        c = vals * (vals - 1) // 2
        return int(c.sum() // 2)

    # ---------- polarity graph of PG(2,5) ----------
    def pg2_points():
        pts = []
        for x in range(5):
            for y in range(5):
                pts.append((1, x, y))
        for z in range(5):
            pts.append((0, 1, z))
        pts.append((0, 0, 1))
        return pts

    def bilinear(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (x1 * z2 + x2 * z1 - 2 * y1 * y2) % 5

    def polarity_edges():
        pts = pg2_points()
        E = set()
        for i in range(n):
            for j in range(i + 1, n):
                if bilinear(pts[i], pts[j]) == 0:
                    E.add((i, j))
        return E

    # ---------- greedy C4-free base ----------
    def greedy_base():
        A = np.zeros((n, n), dtype=np.int64)
        E = set()
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for (u, v) in pairs:
            if len(E) >= 90:
                break
            if A[u, v] == 0:
                A[u, v] = 1
                A[v, u] = 1
                E.add((u, v))
                A[u] += A[v]
                A[v] += A[u]
        if len(E) < 90:
            for (u, v) in pairs:
                if len(E) >= 90:
                    break
                if (u, v) not in E and A[u, v] == 0:
                    A[u, v] = 1
                    A[v, u] = 1
                    E.add((u, v))
                    A[u] += A[v]
                    A[v] += A[u]
        return E

    # ---------- build base ----------
    base_E = polarity_edges()
    base_A = np.zeros((n, n), dtype=np.int64)
    for (u, v) in base_E:
        base_A[u, v] = 1
        base_A[v, u] = 1
    base_A = (base_A @ base_A).astype(np.int64)
    base_c4 = count_c4(base_A)

    if len(base_E) != 90 or base_c4 != 0:
        base_E = greedy_base()
        base_A = np.zeros((n, n), dtype=np.int64)
        for (u, v) in base_E:
            base_A[u, v] = 1
            base_A[v, u] = 1
        base_A = (base_A @ base_A).astype(np.int64)
        base_c4 = count_c4(base_A)

    # ---------- add the best single edge ----------
    def best_extra(E, A):
        Eset = set()
        for (a, b) in E:
            Eset.add((a, b))
            Eset.add((b, a))
        best = None
        best_c = None
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in Eset:
                    c = int(A[i, j])
                    if best_c is None or c < best_c:
                        best_c = c
                        best = (i, j)
        return best, best_c

    extra, extra_c = best_extra(base_E, base_A)
    if extra is None:
        extra = (0, 1)
        extra_c = int(base_A[0, 1])

    start_E = list(base_E) + [extra]
    start_A = base_A.copy()
    u, v = extra
    start_A[u] += base_A[v]
    start_A[v] += base_A[u]
    start_c4 = base_c4 + extra_c

    # ---------- delta helpers ----------
    def apply_add(A, Eset, u, v):
        A[u] += A[v]
        A[v] += A[u]
        A[u, v] = 1
        A[v, u] = 1
        Eset.add((u, v))
        Eset.add((v, u))

    def apply_remove(A, Eset, u, v):
        A[u, v] = 0
        A[v, u] = 0
        A[u] -= A[v]
        A[v] -= A[u]
        Eset.discard((u, v))
        Eset.discard((v, u))

    def eval_swap(A, Eset, rm, add):
        u1, v1 = rm
        u2, v2 = add
        return -int(A[u1, v1]) + int(A[u2, v2] + A[u1, u2] * A[v1, u2] + A[u1, v2] * A[v1, v2])

    # ---------- local search (SA + greedy polish) ----------
    def local_search(A0, E0, budget):
        A = A0.copy()
        E = list(E0)
        Eset = set()
        for (a, b) in E:
            Eset.add((a, b))
            Eset.add((b, a))
        cur = count_c4(A)
        best = cur
        best_A = A.copy()
        best_E = list(E)
        best_Eset = set(Eset)
        rng = random.Random(abs(hash((tuple(sorted(E)), cur))) % (2**31))

        def do_swap(rm, add):
            u1, v1 = rm
            u2, v2 = add
            apply_remove(A, Eset, u1, v1)
            apply_add(A, Eset, u2, v2)
            E.remove(rm)
            E.append(add)

        temp = 8.0
        cool = 0.985
        iters = 0
        while time.time() < budget:
            iters += 1
            idx = rng.randrange(len(E))
            rm = E[idx]
            cand = None
            for _ in range(40):
                a = rng.randrange(n)
                b = rng.randrange(n)
                if a == b:
                    continue
                if (a, b) in Eset:
                    continue
                cand = (a, b)
                break
            if cand is None:
                continue
            d = eval_swap(A, Eset, rm, cand)
            if d < 0 or (temp > 0.05 and rng.random() < min(1.0, np.exp(-d / max(temp, 1e-9)))):
                do_swap(rm, cand)
                cur += d
                if cur < best:
                    best = cur
                    best_A = A.copy()
                    best_E = list(E)
                    best_Eset = set(Eset)
            temp *= cool
            if iters % 2000 == 0 and temp < 0.5:
                temp = 0.5

        # final greedy polish on the best state
        A = best_A.copy()
        E = list(best_E)
        Eset = set(best_Eset)
        cur = best
        improved = True
        while improved and time.time() < budget:
            improved = False
            for i in range(len(E)):
                if time.time() > budget:
                    break
                rm = E[i]
                badd = None
                bd = None
                for a in range(n):
                    for b in range(a + 1, n):
                        if (a, b) in Eset:
                            continue
                        d = eval_swap(A, Eset, rm, (a, b))
                        if bd is None or d < bd:
                            bd = d
                            badd = (a, b)
                if badd is not None and bd < 0:
                    do_swap(rm, badd)
                    cur += bd
                    improved = True
                    if cur < best:
                        best = cur
                        best_A = A.copy()
                        best_E = list(E)
                        best_Eset = set(Eset)
                    break
        return best_E, best

    # ---------- multi-start ----------
    global_best_E = start_E
    global_best = start_c4

    e, c = local_search(start_A, start_E, deadline - 8.0)
    if c < global_best:
        global_best = c
        global_best_E = e

    rng2 = random.Random(12345)
    nstarts = 0
    while time.time() < deadline - 12.0 and nstarts < 12:
        nstarts += 1
        E = list(base_E)
        A = base_A.copy()
        Eset = set()
        for (a, b) in E:
            Eset.add((a, b))
            Eset.add((b, a))
        k = rng2.choice([1, 2, 3])
        for _ in range(k):
            if len(E) == 0:
                break
            idx = rng2.randrange(len(E))
            rm = E.pop(idx)
            u1, v1 = rm
            apply_remove(A, Eset, u1, v1)
        for _ in range(k):
            cand = None
            for _ in range(60):
                a = rng2.randrange(n)
                b = rng2.randrange(n)
                if a == b or (a, b) in Eset:
                    continue
                cand = (a, b)
                break
            if cand is None:
                continue
            u2, v2 = cand
            apply_add(A, Eset, u2, v2)
            E.append(cand)
        while len(E) < 91:
            cand = None
            for _ in range(60):
                a = rng2.randrange(n)
                b = rng2.randrange(n)
                if a == b or (a, b) in Eset:
                    continue
                cand = (a, b)
                break
            if cand is None:
                break
            u2, v2 = cand
            apply_add(A, Eset, u2, v2)
            E.append(cand)
        while len(E) > 91:
            rm = E.pop()
            u1, v1 = rm
            apply_remove(A, Eset, u1, v1)
        if len(E) != 91:
            continue
        e, c = local_search(A, E, deadline - 10.0)
        if c < global_best:
            global_best = c
            global_best_E = e

    # ---------- final verification and return ----------
    out = []
    seen = set()
    for (a, b) in global_best_E:
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        out.append([key[0], key[1]])

    if len(out) != target:
        out = []
        seen = set()
        for (a, b) in start_E:
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            out.append([key[0], key[1]])

    return out

# EVOLVE-BLOCK-END
