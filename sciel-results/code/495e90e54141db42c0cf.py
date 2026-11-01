import time
import random

def construct(n_rows, k):
    # n_rows=4, k=7 -> 8 numbers per row (starting at 0), all differences distinct
    start_time = time.time()
    # Safety margin before the 150 s wall-clock kill.
    deadline = start_time + 142.0

    def time_left():
        return deadline - time.time()

    MAXV = 160  # bitset window; differences we track are < MAXV

    def scope(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        seen = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    # ---- fast bitset bookkeeping --------------------------------------------
    def build_state(rows):
        masks = [0] * n_rows
        used = 0
        for r in range(n_rows):
            row = rows[r]
            m = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    m |= 1 << (row[j] - row[i])
            masks[r] = m
            used |= m
        return masks, used

    def row_mask(row):
        m = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                m |= 1 << (row[j] - row[i])
        return m

    def greedy_r(used, rng):
        """Build one row of k+1 increasing ints from 0, avoiding `used` diffs."""
        row = [0]
        rd = 0
        for _ in range(k):
            base = row[-1] + 1
            cands = []
            for d in range(1, 60):
                x = base + d - 1
                new = 0
                ok = True
                for a in row:
                    diff = x - a
                    bit = 1 << diff
                    if (used | rd | new) & bit:
                        ok = False
                        break
                    new |= bit
                if ok:
                    cands.append((x, new))
            if not cands:
                return None
            # Randomize among the smallest valid candidates to diversify restarts.
            t = min(6, len(cands))
            x, new = cands[rng.randrange(t)]
            row.append(x)
            rd |= new
            used |= new
        return row

    def local_search(rows, max_iters, rng):
        """Single-element local search minimizing the global max entry."""
        rows = [list(r) for r in rows]
        masks, used = build_state(rows)
        for _ in range(max_iters):
            if time_left() < 3:
                break
            r = max(range(n_rows), key=lambda i: max(rows[i]))
            row = rows[r]
            L = len(row)
            curmax = row[-1]
            m_other = used ^ masks[r]
            improved = False

            # 1) shift a single element to a smaller unused value
            for pos in range(L - 1, 0, -1):
                old = row[pos]
                if old >= curmax:
                    continue  # last element is curmax; shifting it changes scope
                lo = row[pos - 1] + 1
                hi = row[pos + 1] - 1 if pos + 1 < L else old - 1
                newm = masks[r] & ~(1 << 0)
                for a in range(pos):
                    newm &= ~(1 << (old - row[a]))
                for b in range(pos + 1, L):
                    newm &= ~(1 << (row[b] - old))
                for nv in range(lo, hi + 1):
                    cand = 0
                    ok = True
                    for a in range(pos):
                        diff = nv - row[a]
                        bit = 1 << diff
                        if (m_other | newm | cand) & bit:
                            ok = False
                            break
                        cand |= bit
                    if ok:
                        for b in range(pos + 1, L):
                            diff = row[b] - nv
                            bit = 1 << diff
                            if (m_other | newm | cand) & bit:
                                ok = False
                                break
                            cand |= bit
                    if ok:
                        row[pos] = nv
                        masks[r] = newm | cand
                        used = (used ^ masks[r]) | (newm | cand)
                        improved = True
                        break
                if improved:
                    break

            # 2) swap two elements (same row)
            if not improved:
                done = False
                for pos in range(L - 1, 0, -1):
                    if done:
                        break
                    old = row[pos]
                    if old >= curmax:
                        continue
                    for pos2 in range(L - 1, pos, -1):
                        old2 = row[pos2]
                        if old2 >= curmax:
                            continue
                        if old2 <= row[pos - 1] or old >= row[pos2 + 1] if pos2 + 1 < L else False:
                            pass
                        # feasibility: new positions keep increasing order
                        if pos2 + 1 < L and old >= row[pos2 + 1]:
                            continue
                        if old2 <= row[pos - 1]:
                            continue
                        if old2 >= (row[pos2 + 1] if pos2 + 1 < L else old2 + 1):
                            continue
                        newm = masks[r] & ~(1 << 0)
                        for a in range(L):
                            if a == pos or a == pos2:
                                continue
                            newm &= ~(1 << (old - row[a]))
                            newm &= ~(1 << (row[a] - old))
                            newm &= ~(1 << (old2 - row[a]))
                            newm &= ~(1 << (row[a] - old2))
                        cand = 0
                        ok = True
                        for a in range(pos):
                            diff = old2 - row[a]
                            bit = 1 << diff
                            if (m_other | newm | cand) & bit:
                                ok = False
                                break
                            cand |= bit
                        if ok:
                            for b in range(pos + 1, L):
                                if b == pos2:
                                    continue
                                diff = row[b] - old2
                                bit = 1 << diff
                                if (m_other | newm | cand) & bit:
                                    ok = False
                                    break
                                cand |= bit
                        if ok:
                            for a in range(pos2):
                                if a == pos:
                                    continue
                                diff = old - row[a]
                                bit = 1 << diff
                                if (m_other | newm | cand) & bit:
                                    ok = False
                                    break
                                cand |= bit
                        if ok:
                            for b in range(pos2 + 1, L):
                                diff = row[b] - old
                                bit = 1 << diff
                                if (m_other | newm | cand) & bit:
                                    ok = False
                                    break
                                cand |= bit
                        if ok:
                            # cross term between the two swapped positions
                            cross = abs(old - old2)
                            bit = 1 << cross
                            if (m_other | newm | cand) & bit:
                                continue
                            cand |= bit
                            row[pos], row[pos2] = old2, old
                            masks[r] = newm | cand
                            used = (used ^ masks[r]) | (newm | cand)
                            improved = True
                            done = True
                            break
            if not improved:
                break
        return rows

    # ---- greedy baseline (deterministic) ------------------------------------
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = greedy_r(used, random.Random(0))
            if row is None:
                return None
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---- multi-restart: diverse greedy + local search -----------------------
    restarts = 0
    while time_left() > 20:
        rng = random.Random()
        used = 0
        rows = []
        ok = True
        for _ in range(n_rows):
            row = greedy_r(used, rng)
            if row is None:
                ok = False
                break
            rows.append(row)
            used |= row_mask(row)
        if ok:
            polished = local_search(rows, 4000, rng)
            s = scope(polished)
            if s < best_scope:
                best_rows = polished
                best_scope = s
        restarts += 1
        # Occasionally re-polish the current best to squeeze more.
        if restarts % 3 == 0 and time_left() > 15:
            repol = local_search(best_rows, 6000, random.Random())
            s = scope(repol)
            if s < best_scope:
                best_rows = repol
                best_scope = s

    # ---- final high-precision polish (tighter budget, more iterations) ------
    if time_left() > 4:
        final = local_search(best_rows, 20000, random.Random())
        s = scope(final)
        if s < best_scope:
            best_rows = final
            best_scope = s

    # safety: return a guaranteed-valid set
    if not is_valid(best_rows):
        fb = greedy()
        if fb is not None:
            return fb
    return best_rows
