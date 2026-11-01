import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i - 1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    # ---------- Greedy incumbent (gap-based, fast) ----------
    def greedy_init(seed=0, rand_prob=0.12, window=40):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            gaps = []
            prefix = 0
            for gi in range(k):
                # required differences from this new entry to the gi existing entries
                # entry value = prefix + g ; differences = (prefix+g) - prefix_{j}
                candidates = []
                x = 1
                # entries so far are prefix sums; keep them
                entries = [0]
                s = 0
                for g in gaps:
                    s += g
                    entries.append(s)
                # entries has gi+1 elements (prefix is last)
                found = False
                scan = 0
                while not found:
                    val = x
                    ok = True
                    for a in entries:
                        d = val - a
                        if d in used:
                            ok = False
                            break
                    if ok:
                        candidates.append(val)
                        if len(candidates) >= window:
                            found = True
                    x += 1
                if rng.random() < rand_prob and len(candidates) > 1:
                    pick = rng.choice(candidates)
                else:
                    pick = candidates[0]
                g = pick - prefix
                gaps.append(g)
                prefix = pick
                entries.append(pick)
                for a in entries[:-1]:
                    used.add(pick - a)
            rows.append([0] + list(itertools_prefix(gaps)))
        return rows

    def itertools_prefix(gaps):
        out = []
        s = 0
        for g in gaps:
            s += g
            out.append(s)
        return out

    # ---------- Backtracking over gap variables with capacity ordering ----------
    def backtrack(n_rows, k, target_scope, time_limit, seed=0):
        rng = random.Random(seed)
        start = time.time()
        # entries[r][c] for c in 0..k ; entries[r][0] = 0
        entries = [[0] * row_len for _ in range(n_rows)]
        used = set()

        def prefix_row(r):
            # list of currently set entries in row r (contiguous from 0)
            return entries[r]

        def row_fill_count(r):
            # number of gaps already placed in row r
            cnt = 0
            for c in range(1, row_len):
                if entries[r][c] > 0 or (entries[r][c] == 0 and c == 0):
                    pass
            # entries are always increasing and start at 0; a gap is "placed"
            # iff entries[r][c] > entries[r][c-1]
            cnt = 0
            for c in range(1, row_len):
                if entries[r][c] > entries[r][c - 1]:
                    cnt += 1
            return cnt

        def all_filled():
            for r in range(n_rows):
                for c in range(1, row_len):
                    if entries[r][c] <= entries[r][c - 1]:
                        return False
            return True

        def candidate_gaps(r, c, max_val):
            # gap to place at position c (1-indexed entry) in row r
            prev_val = entries[r][c - 1]
            lower = prev_val + 1
            upper = min(max_val, target_scope)
            if lower > upper:
                return []
            # differences this entry must be distinct from all used
            # and from differences to earlier entries in same row (which become used)
            out = []
            scan_limit = min(upper, lower + 60)
            for val in range(lower, scan_limit + 1):
                ok = True
                for i in range(c):
                    d = val - entries[r][i]
                    if d in used:
                        ok = False
                        break
                if ok:
                    out.append(val)
                    if len(out) >= 20:
                        break
            if len(out) < 20 and scan_limit < upper:
                for val in range(scan_limit + 1, upper + 1):
                    ok = True
                    for i in range(c):
                        d = val - entries[r][i]
                        if d in used:
                            ok = False
                            break
                    if ok:
                        out.append(val)
                        if len(out) >= 20:
                            break
            return out

        def forward_check(r, c):
            # ensure every not-yet-filled gap has >=1 candidate
            for r2 in range(n_rows):
                fc = row_fill_count(r2)
                for c2 in range(fc + 1, row_len):
                    if len(candidate_gaps(r2, c2, target_scope)) == 0:
                        return False
            return True

        def solve():
            if time.time() > start + time_limit:
                return "timeout"
            if all_filled():
                return True
            # choose variable with fewest viable gaps (MRV on capacity)
            best = None
            best_count = 10 ** 9
            for r in range(n_rows):
                fc = row_fill_count(r)
                for c in range(fc + 1, row_len):
                    cands = candidate_gaps(r, c, target_scope)
                    if len(cands) == 0:
                        return False
                    if len(cands) < best_count:
                        best_count = len(cands)
                        best = (r, c, cands)
                        if best_count == 1:
                            break
                if best_count == 1:
                    break
            if best is None:
                return True
            r, c, cands = best
            # order: try values that keep scope small first, but spread a bit
            cands = sorted(cands)
            # mix: first few low values, then a couple random
            order = cands[:min(6, len(cands))]
            if len(cands) > 6:
                order += rng.sample(cands[6:], min(4, len(cands) - 6))
            for val in order:
                if time.time() > start + time_limit:
                    return "timeout"
                if val > target_scope:
                    continue
                entries[r][c] = val
                new_diffs = [val - entries[r][i] for i in range(c)]
                for d in new_diffs:
                    used.add(d)
                ok = forward_check(r, c)
                if ok:
                    res = solve()
                    if res is True:
                        return True
                    if res == "timeout":
                        # undo then report timeout
                        entries[r][c] = 0
                        for d in new_diffs:
                            used.discard(d)
                        return "timeout"
                # undo
                entries[r][c] = 0
                for d in new_diffs:
                    used.discard(d)
            return False

        res = solve()
        if res is True:
            return [list(entries[r]) for r in range(n_rows)]
        return None

    # ---------- Lightweight single-element polish (descent on scope) ----------
    def polish(rows, time_limit):
        start = time.time()
        best = [r[:] for r in rows]
        best_scope = get_scope(best)
        improved = True
        while time.time() < start + time_limit and improved:
            improved = False
            # compute fixed diffs once per outer pass
            for r_idx in range(n_rows):
                for c_idx in range(1, row_len):
                    if time.time() > start + time_limit:
                        return best, best_scope
                    row = best[r_idx]
                    cur = row[c_idx]
                    prev = row[c_idx - 1]
                    # all diffs
                    all_diffs = set()
                    for rr in best:
                        for i in range(len(rr)):
                            for j in range(i + 1, len(rr)):
                                all_diffs.add(rr[j] - rr[i])
                    rem = set()
                    for i in range(c_idx):
                        rem.add(cur - row[i])
                    for j in range(c_idx + 1, len(row)):
                        rem.add(row[j] - cur)
                    fixed = all_diffs - rem
                    found = False
                    for nv in range(cur - 1, prev, -1):
                        ok = True
                        for i in range(c_idx):
                            if (nv - row[i]) in fixed:
                                ok = False
                                break
                        if ok:
                            for j in range(c_idx + 1, len(row)):
                                if (row[j] - nv) in fixed:
                                    ok = False
                                    break
                        if not ok:
                            continue
                        left = [nv - row[i] for i in range(c_idx)]
                        right = [row[j] - nv for j in range(c_idx + 1, len(row))]
                        if len(set(left + right)) != len(left + right):
                            continue
                        row[c_idx] = nv
                        found = True
                        improved = True
                        break
                    if found:
                        best_scope = get_scope(best)
        return best, best_scope

    # ---------- Main driver ----------
    best_rows = greedy_init(seed=1)
    if not validate(best_rows):
        # fallback: rebuild with different seeds
        for s in range(2, 12):
            cand = greedy_init(seed=s)
            if validate(cand):
                best_rows = cand
                break
    best_scope = get_scope(best_rows)

    # quick polish
    if time.time() < deadline - 60:
        pr, ps = polish(best_rows, min(20.0, deadline - time.time() - 60))
        if validate(pr):
            best_rows, best_scope = pr, ps

    # iterative deepening down the target scope
    target = best_scope - 1
    seed_counter = 0
    lower_bound = 112
    while time.time() < deadline - 8 and target >= lower_bound:
        remaining = deadline - time.time() - 4
        if remaining < 2:
            break
        st = min(12.0, remaining)
        res = backtrack(n_rows, k, target, st, seed=seed_counter)
        seed_counter += 1
        if res and validate(res):
            ns = get_scope(res)
            if ns < best_scope:
                best_scope = ns
                best_rows = res
                target = best_scope - 1
            else:
                # no improvement at this target; try a couple more seeds then lower
                extra = 0
                while extra < 2 and time.time() < deadline - 8:
                    remaining = deadline - time.time() - 4
                    if remaining < 2:
                        break
                    res2 = backtrack(n_rows, k, target, min(8.0, remaining), seed=seed_counter)
                    seed_counter += 1
                    if res2 and validate(res2):
                        ns = get_scope(res2)
                        if ns < best_scope:
                            best_scope = ns
                            best_rows = res2
                            target = best_scope - 1
                            break
                    extra += 1
                else:
                    target -= 1
        else:
            # failed to find solution within target; lower target
            target -= 1

    # final polish with remaining time
    if time.time() < deadline - 2:
        pt = min(12.0, deadline - time.time() - 2)
        if pt > 2:
            pr, ps = polish(best_rows, pt)
            if validate(pr):
                best_rows, best_scope = pr, ps

    return best_rows


if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
