import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before the 150s kill

    L = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1  # counting lower bound on the scope

    # ---------- fast conflict-counting core ----------
    def build_counts(rows, maxv):
        cnt = [0] * (maxv + 1)
        for row in rows:
            for i in range(L):
                ri = row[i]
                for j in range(i + 1, L):
                    cnt[row[j] - ri] += 1
        return cnt

    def num_conflicts(cnt):
        s = 0
        for v in cnt:
            if v >= 2:
                s += v - 1
        return s

    def scope(rows):
        m = 0
        for row in rows:
            if row[-1] > m:
                m = row[-1]
        return m

    def compaction_pass(rows, cnt, maxv, time_limit, backward):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r in range(n_rows):
                row = rows[r]
                positions = range(L - 1, 0, -1) if backward else range(1, L)
                for pos in positions:
                    old = row[pos]
                    lower = row[pos - 1] + 1
                    if lower >= old:
                        continue
                    upper = row[pos + 1] if pos < L - 1 else old
                    best = old
                    for nv in range(lower, min(upper, old)):
                        # remove old diffs
                        for i in range(L):
                            if i == pos:
                                continue
                            d = old - row[i] if i < pos else row[i] - old
                            cnt[d] -= 1
                        # add new diffs
                        ok = True
                        for i in range(L):
                            if i == pos:
                                continue
                            d = nv - row[i] if i < pos else row[i] - nv
                            if d >= maxv:
                                ok = False
                                break
                            if cnt[d] > 0:
                                ok = False
                                break
                            cnt[d] += 1
                        if ok:
                            row[pos] = nv
                            best = nv
                            improved = True
                        else:
                            # undo new, restore old
                            for i in range(L):
                                if i == pos:
                                    continue
                                d = nv - row[i] if i < pos else row[i] - nv
                                cnt[d] -= 1
                            for i in range(L):
                                if i == pos:
                                    continue
                                d = old - row[i] if i < pos else row[i] - old
                                cnt[d] += 1
                            break
                    # if best < old the row was already updated; else unchanged
        return improved

    def swap_pass(rows, cnt, maxv, time_limit):
        """Try to reduce total conflicts / scope by swapping a suffix of one row
        with a suffix of another (same length, both increasing, no new conflicts)."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r1 in range(n_rows):
                for r2 in range(r1 + 1, n_rows):
                    row1 = rows[r1]
                    row2 = rows[r2]
                    for s in range(L - 1, 0, -1):
                        # candidate: row1[0:s] + row2[s:], row2[0:s] + row1[s:]
                        tail1 = row1[s:]
                        tail2 = row2[s:]
                        # both tails must be increasing and compatible with prefix
                        if tail1[0] <= row1[s - 1]:
                            continue
                        if tail2[0] <= row2[s - 1]:
                            continue
                        # gather all diffs that will change
                        # old diffs involving positions >= s in row1 and row2
                        old_diffs = []
                        new_diffs = []
                        for i in range(L):
                            for j in range(i + 1, L):
                                if i >= s or j >= s:
                                    old_diffs.append((row1[j], row1[i]))
                        for i in range(L):
                            for j in range(i + 1, L):
                                if i >= s or j >= s:
                                    old_diffs.append((row2[j], row2[i]))
                        for (hi, lo) in old_diffs:
                            cnt[hi - lo] -= 1
                        # build new rows
                        new_row1 = row1[:s] + tail2
                        new_row2 = row2[:s] + tail1
                        ok = True
                        for i in range(L):
                            for j in range(i + 1, L):
                                if i >= s or j >= s:
                                    d = new_row1[j] - new_row1[i]
                                    if d >= maxv or cnt[d] > 0:
                                        ok = False
                                        break
                                    cnt[d] += 1
                            if not ok:
                                break
                        if ok:
                            for i in range(L):
                                for j in range(i + 1, L):
                                    if i >= s or j >= s:
                                        d = new_row2[j] - new_row2[i]
                                        if d >= maxv or cnt[d] > 0:
                                            ok = False
                                            break
                                        cnt[d] += 1
                        if ok:
                            rows[r1] = new_row1
                            rows[r2] = new_row2
                            improved = True
                        else:
                            # undo new, restore old
                            for i in range(L):
                                for j in range(i + 1, L):
                                    if i >= s or j >= s:
                                        d = new_row1[j] - new_row1[i]
                                        cnt[d] -= 1
                            for i in range(L):
                                for j in range(i + 1, L):
                                    if i >= s or j >= s:
                                        d = new_row2[j] - new_row2[i]
                                        cnt[d] -= 1
                            for (hi, lo) in old_diffs:
                                cnt[hi - lo] += 1
                            break
                    if improved:
                        break
                if improved:
                    break
        return improved

    def polish(rows, maxv, time_limit):
        cnt = build_counts(rows, maxv)
        # alternate compaction (forward+backward) and swaps until no improvement
        while time.time() < time_limit:
            p1 = compaction_pass(rows, cnt, maxv, time_limit, backward=True)
            p2 = compaction_pass(rows, cnt, maxv, time_limit, backward=False)
            p3 = swap_pass(rows, cnt, maxv, time_limit)
            if not (p1 or p2 or p3):
                break
        return rows

    # ---------- seeding ----------
    def greedy(seed, mode, gap_base, gap_var, sparse):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < L:
                last = row[-1]
                found = False
                if sparse:
                    # deliberately large gaps to leave room for compaction
                    search_end = last + gap_base + rng.randint(0, gap_var) + 25
                else:
                    search_end = last + gap_base + rng.randint(0, gap_var)
                hi = min(search_end, maxv)
                candidates = list(range(last + 1, hi))
                if rng.random() < 0.5:
                    rng.shuffle(candidates)
                else:
                    rng.shuffle(candidates)  # still shuffle for diversity
                for x in candidates:
                    nd = [x - a for a in row]
                    if any(d in used for d in nd):
                        continue
                    row.append(x)
                    used.update(nd)
                    found = True
                    break
                if not found:
                    for x in range(last + 1, maxv + 1):
                        nd = [x - a for a in row]
                        if any(d in used for d in nd):
                            continue
                        row.append(x)
                        used.update(nd)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def random_restart(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < L:
                last = row[-1]
                found = False
                jump = rng.randint(1, 60)
                x = last + jump
                for candidate in range(max(last + 1, x - 12), min(x + 14, maxv + 1)):
                    nd = [candidate - a for a in row]
                    if any(d in used for d in nd):
                        continue
                    row.append(candidate)
                    used.update(nd)
                    found = True
                    break
                if not found:
                    for candidate in range(last + 1, maxv + 1):
                        nd = [candidate - a for a in row]
                        if any(d in used for d in nd):
                            continue
                        row.append(candidate)
                        used.update(nd)
                        found = True
                        break
                    if not found:
                        return None
            rows.append(row)
        return rows

    # ---------- main search ----------
    maxv = 140  # hard cap on element values (well above the best known 134)
    best_rows = None
    best_scope = float('inf')

    def consider(rows):
        nonlocal best_rows, best_scope
        if rows is None:
            return
        sc = scope(rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = [list(r) for r in rows]

    # Phase 1: diverse greedy seeds, quick polish
    modes = [('basic', 26, 16, False), ('compact', 18, 10, False),
             ('spread', 40, 30, False), ('sparse', 30, 20, True)]
    for i in range(1200):
        if time.time() > deadline - 25:
            break
        mode, gb, gv, sp = modes[i % len(modes)]
        rows = greedy(i * 31 + 7, mode, gb, gv, sp)
        if rows is None:
            continue
        sc = scope(rows)
        if sc < best_scope:
            # only polish promising seeds (cheap early polish)
            polished = polish([list(r) for r in rows], maxv, min(time.time() + 0.6, deadline - 25))
            consider(polished)
        if best_scope == lb:
            break

    # Phase 2: random restarts + full polish
    for i in range(600):
        if time.time() > deadline - 20:
            break
        rows = random_restart(100000 + i * 97)
        if rows is None:
            continue
        sc = scope(rows)
        if sc < best_scope + 30:  # only polish if plausibly competitive
            polished = polish([list(r) for r in rows], maxv, min(time.time() + 0.8, deadline - 20))
            consider(polished)
        if best_scope == lb:
            break

    # Phase 3: intensive polish of the best found (full budget)
    if best_rows is not None and time.time() < deadline - 3:
        polished = polish([list(r) for r in best_rows], maxv, deadline - 2)
        consider(polished)

    # Phase 4: final quick polish
    if best_rows is not None and time.time() < deadline - 1:
        polished = polish([list(r) for r in best_rows], maxv, deadline - 0.5)
        consider(polished)

    return best_rows


# Quick self-test
if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
