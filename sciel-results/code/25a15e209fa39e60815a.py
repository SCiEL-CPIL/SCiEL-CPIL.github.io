"""Difference triangle set (n_rows, k): minimize the largest entry (scope).
Bitmask backtracking + iterative deepening + re-optimization/perturbation."""
import time
import random


def _row_mask(row):
    """Bitmask of all positive pairwise differences of a row (bit d set iff d used)."""
    m = 0
    L = len(row)
    for i in range(L):
        ai = row[i]
        for j in range(i + 1, L):
            m |= 1 << (row[j] - ai)
    return m


def _valid(rows, k):
    seen = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 110.0          # safety margin well under the 120/150 s limit
    diffs_per_row = k * (k + 1) // 2
    lower = n_rows * diffs_per_row  # counting bound on the scope
    row_floor = max(k, diffs_per_row)  # min possible max of a single (k+1)-mark ruler

    # ---- greedy baseline (always valid) ----
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                m = 0
                ok = True
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        ok = False
                        break
                    m |= 1 << d
                if ok:
                    row.append(x)
                    used |= m
            rows.append(row)
        return rows

    best = greedy()
    M = max(max(r) for r in best)

    # ---- row generator: yields Golomb rulers with max <= Mbound,
    #      differences disjoint from `used`, forward-checked, node-capped ----
    def gen_rows(used, Mbound, deadline, rng, node_cap):
        row = [0]
        rmask = [0]
        nodes = [0]

        def dfs():
            if time.time() > deadline:
                return
            if len(row) == k + 1:
                yield row[:]
                return
            last = row[-1]
            need = k + 1 - len(row)
            if last + need > Mbound:
                return
            hi = Mbound - (need - 1)
            cands = list(range(last + 1, hi + 1))
            if rng is not None:
                rng.shuffle(cands)
            for x in cands:
                nodes[0] += 1
                if nodes[0] > node_cap:
                    return
                m = 0
                ok = True
                for a in row:
                    d = x - a
                    if ((used >> d) & 1) or ((rmask[0] >> d) & 1):
                        ok = False
                        break
                    m |= 1 << d
                if not ok:
                    continue
                row.append(x)
                old = rmask[0]
                rmask[0] = old | m
                yield from dfs()
                rmask[0] = old
                row.pop()

        return dfs()

    # ---- full backtracking solve under a global bound ----
    def attempt(Mbound, deadline, rng, node_cap):
        used = [0]
        rows = []

        def bt(i):
            if time.time() > deadline:
                return False
            if i == n_rows:
                return True
            for row in gen_rows(used[0], Mbound, deadline, rng, node_cap):
                rm = _row_mask(row)
                rows.append(row)
                old = used[0]
                used[0] = old | rm
                if bt(i + 1):
                    return True
                used[0] = old
                rows.pop()
            return False

        ok = bt(0)
        return rows[:] if ok else None

    # ---- Phase B: iterative deepening full solve, downward (time-limited) ----
    tb_dead = t0 + 40.0
    for Mtry in range(M - 1, lower - 1, -1):
        if time.time() > tb_dead:
            break
        found = None
        for seed in range(3):
            if time.time() > tb_dead:
                break
            rng = None if seed == 0 else random.Random(1000 + seed)
            sol = attempt(Mtry, tb_dead, rng, 400000)
            if sol is not None:
                found = sol
                break
        if found is not None:
            best = found
            M = Mtry
        else:
            break  # lower bounds are harder; spend remaining time re-optimizing

    # ---- Phase C: re-optimization + perturbation ratchet ----
    tc_dead = DEADLINE - 2.0
    while time.time() < tc_dead:
        maxv = max(max(r) for r in best)
        # target a row holding the global max
        idx = next(i for i in range(n_rows) if max(best[i]) == maxv)
        others = 0
        for j in range(n_rows):
            if j != idx:
                others |= _row_mask(best[j])
        cur = max(best[idx])
        newrow = None
        for bnd in range(row_floor, cur):
            if time.time() > tc_dead:
                break
            for row in gen_rows(others, bnd, tc_dead,
                                random.Random(idx * 100003 + bnd + 7), 20000):
                newrow = row
                break
            if newrow is not None:
                break
        if newrow is not None:
            cand = best[:]
            cand[idx] = newrow
            best = cand
            continue

        # perturbation: re-solve a random row (keeps validity) to diversify
        j = random.randrange(n_rows)
        others = 0
        for t2 in range(n_rows):
            if t2 != j:
                others |= _row_mask(best[t2])
        curj = max(best[j])
        bnd = min(curj, maxv)
        for row in gen_rows(others, bnd, tc_dead,
                            random.Random((time.time_ns() + j) & 0xFFFFFFFF), 20000):
            cand = best[:]
            cand[j] = row
            if max(max(r) for r in cand) <= maxv:
                best = cand
            break

    # ---- final safety: guarantee validity ----
    if not _valid(best, k):
        best = greedy()
    return best
