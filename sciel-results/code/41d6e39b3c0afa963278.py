"""Difference triangle sets (n_rows, k) via VNS over difference-sets."""
import time
import random
import math
from collections import deque


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    L = k + 1
    D = k * (k + 1) // 2

    # ---------- helpers ----------
    def scope_of(rows):
        return max(max(r) for r in rows)

    def row_diff_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def diff_set_of_row(row):
        return row_diff_set(row)

    def reconstruct(diff_set, max_depth=600000):
        """Reconstruct a row from its difference set. Returns list or None."""
        if len(diff_set) != D:
            return None
        maxd = max(diff_set)
        nodes = [0]
        idx = {0: 0}
        q = deque([0])
        steps = 0
        while len(nodes) < L:
            steps += 1
            if steps > max_depth:
                return None
            new_nodes = []
            for v in q:
                for d in diff_set:
                    if d > maxd:
                        continue
                    nv = v + d
                    if nv > maxd:
                        continue
                    if nv in idx:
                        continue
                    # all differences from nv to existing nodes must be in diff_set
                    ok = True
                    for w in nodes:
                        if (nv - w) not in diff_set:
                            ok = False
                            break
                    if not ok:
                        continue
                    idx[nv] = len(nodes)
                    nodes.append(nv)
                    new_nodes.append(nv)
            if not new_nodes:
                return None
            q = deque(new_nodes)
        nodes.sort()
        # verify
        if row_diff_set(nodes) != diff_set:
            return None
        return nodes

    # ---------- construction ----------
    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def pressure_greedy(window=60):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                cands = list(range(x + 1, x + window + 1))
                random.shuffle(cands)
                found = False
                best_cand = None
                best_maxd = None
                for cand in cands:
                    new = {cand - a for a in row}
                    if new & used:
                        continue
                    md = max(new)
                    if best_maxd is None or md < best_maxd:
                        best_maxd = md
                        best_cand = cand
                if best_cand is None:
                    cand = x + 1
                    while True:
                        new = {cand - a for a in row}
                        if not (new & used):
                            best_cand = cand
                            break
                        cand += 1
                row.append(best_cand)
                used |= {best_cand - a for a in row[:-1]}
                x = best_cand
                found = True
            rows.append(row)
        return rows

    # ---------- validity oracle ----------
    def valid(rows):
        used = set()
        for r in rows:
            s = row_diff_set(r)
            if len(s) != D:
                return False
            if s & used:
                return False
            used |= s
        return True

    # ---------- VNS over difference-sets ----------
    def vns(start_rows, time_limit):
        # work on diff-sets
        cur_sets = [row_diff_set(r) for r in start_rows]
        cur_rows = [reconstruct(s) for s in cur_sets]
        if any(r is None for r in cur_rows):
            cur_rows = [r[:] for r in start_rows]
            cur_sets = [row_diff_set(r) for r in cur_rows]
        cur_scope = scope_of(cur_rows)
        best_sets = [set(s) for s in cur_sets]
        best_rows = [r[:] for r in cur_rows]
        best_scope = cur_scope

        # used bitset
        used = set()
        for s in cur_sets:
            used |= s

        def rebuild_row(i):
            r = reconstruct(cur_sets[i])
            if r is None:
                return False
            cur_rows[i] = r
            return True

        def try_swap(i, a, b):
            nonlocal used, cur_scope, best_sets, best_rows, best_scope
            s = cur_sets[i]
            if a in s:
                s.discard(a)
                used.discard(a)
            if b in s:
                s.discard(b)
                used.discard(b)
            s.add(b)
            used.add(b)
            if not rebuild_row(i):
                # revert
                s.discard(b)
                used.discard(b)
                s.add(a)
                used.add(a)
                rebuild_row(i)
                return False
            ns = max(cur_rows[i])
            nscope = max(best_scope, ns)
            if nscope < cur_scope or random.random() < math.exp(-(nscope - cur_scope) / 1.5):
                cur_scope = nscope
                if nscope < best_scope:
                    best_scope = nscope
                    best_sets = [set(x) for x in cur_sets]
                    best_rows = [r[:] for r in cur_rows]
                return True
            # revert
            s.discard(b)
            used.discard(b)
            s.add(a)
            used.add(a)
            rebuild_row(i)
            return False

        it = 0
        while time.time() < time_limit:
            it += 1
            ri = random.randrange(n_rows)
            s = cur_sets[ri]
            elems = list(s)
            if len(elems) < 2:
                continue
            a = random.choice(elems)
            b = random.randint(1, best_scope + 30)
            if b == a:
                continue
            try_swap(ri, a, b)

        return best_rows, best_scope

    def vns_pair(start_rows, time_limit):
        """VNS with pairwise neighborhood switches between two rows."""
        cur_sets = [row_diff_set(r) for r in start_rows]
        cur_rows = [reconstruct(s) for s in cur_sets]
        if any(r is None for r in cur_rows):
            cur_rows = [r[:] for r in start_rows]
            cur_sets = [row_diff_set(r) for r in cur_rows]
        cur_scope = scope_of(cur_rows)
        best_sets = [set(s) for s in cur_sets]
        best_rows = [r[:] for r in cur_rows]
        best_scope = cur_scope

        used = set()
        for s in cur_sets:
            used |= s

        def rebuild(i):
            r = reconstruct(cur_sets[i])
            if r is None:
                return False
            cur_rows[i] = r
            return True

        it = 0
        while time.time() < time_limit and n_rows >= 2:
            it += 1
            i, j = random.sample(range(n_rows), 2)
            si = cur_sets[i]
            sj = cur_sets[j]
            ei = list(si)
            ej = list(sj)
            if not ei or not ej:
                continue
            a = random.choice(ei)
            b = random.choice(ej)
            if a == b:
                continue
            # remove a from si, b from sj; add b to si, a to sj
            si.discard(a); used.discard(a)
            sj.discard(b); used.discard(b)
            si.add(b); used.add(b)
            sj.add(a); used.add(a)
            ok_i = rebuild(i)
            ok_j = rebuild(j)
            if ok_i and ok_j:
                nscope = max(best_scope, max(cur_rows[i]), max(cur_rows[j]))
                if nscope < cur_scope or random.random() < math.exp(-(nscope - cur_scope) / 2.0):
                    cur_scope = nscope
                    if nscope < best_scope:
                        best_scope = nscope
                        best_sets = [set(x) for x in cur_sets]
                        best_rows = [r[:] for r in cur_rows]
                    continue
            # revert
            si.discard(b); used.discard(b)
            sj.discard(a); used.discard(a)
            si.add(a); used.add(a)
            sj.add(b); used.add(b)
            rebuild(i)
            rebuild(j)

        return best_rows, best_scope

    # ---------- main ----------
    best_rows = pure_greedy()
    best_scope = scope_of(best_rows)

    # Phase 1: diverse starts
    t1 = deadline - 55.0
    attempts = 0
    while time.time() < t1 and attempts < 200:
        attempts += 1
        w = 20 + random.randint(0, 120)
        rows = pressure_greedy(window=w)
        sc = scope_of(rows)
        if sc < best_scope:
            best_rows = rows
            best_scope = sc
            if best_scope <= 112:
                break

    # Phase 2: VNS on best + alternatives
    pool = [best_rows]
    t2 = deadline - 25.0
    for _ in range(8):
        if time.time() >= t2:
            break
        rows = pressure_greedy(window=20 + random.randint(0, 120))
        sc = scope_of(rows)
        if sc <= best_scope + 40:
            pool.append(rows)

    t3 = deadline - 15.0
    for start in pool:
        if time.time() >= t3:
            break
        improved, sc = vns(start, t3)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc
            if best_scope <= 112:
                break

    # Phase 3: VNS pair on best
    if time.time() < t3 - 5.0 and n_rows >= 2:
        improved, sc = vns_pair(best_rows, t3 - 5.0)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc

    # Phase 4: final greedy polish on best scope
    if time.time() < deadline - 3.0:
        rows = pure_greedy()
        sc = scope_of(rows)
        if sc < best_scope:
            best_rows = rows
            best_scope = sc

    return best_rows
