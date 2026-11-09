"""Structural change: replace randomized greedy row generation with a
systematic backtracking search using forward checking and constraint
propagation.  The key insight is that we should build rows element-by-element
with strong pruning, rather than generating complete rows and checking them
later.  We also use a tighter initial scope estimate and iterative deepening.
"""
import time
import random
import math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150 s kill
    K1 = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on scope: S(S+1)/2 >= total_diffs
    lb = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)

    def scope(rows):
        return max(max(r) for r in rows) if rows else 0

    def is_valid(rows):
        used = set()
        for r in rows:
            n = len(r)
            for i in range(n):
                for j in range(i + 1, n):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def row_diff_mask(r):
        m = 0
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                m |= 1 << (r[j] - r[i])
        return m

    # ---------- Systematic row construction with forward checking ----------
    def build_row(used_mask, max_val, rng, row_idx=0):
        """Build a single row of length K1 starting at 0 using backtracking
        with forward checking.  Returns the row or None."""
        row = [0]
        row_mask = 0  # differences within this row so far

        def bt(pos, cur, cur_mask):
            if time.time() > DEADLINE - 5:
                return None
            if len(cur) == K1:
                return cur[:]
            start = cur[-1] + 1
            end = min(max_val, start + 60)  # limit candidate window
            if start > max_val:
                return None

            # Generate candidates: mix of greedy (small) and random
            cands = list(range(start, end + 1))
            # Put smallest few first (greedy anchor), then randomize rest
            if len(cands) > 5:
                greedy_part = cands[:5]
                random_part = cands[5:]
                rng.shuffle(random_part)
                cands = greedy_part + random_part
            else:
                rng.shuffle(cands)

            for x in cands:
                # Check if x creates any conflict
                ok = True
                new_mask = 0
                for a in cur:
                    d = x - a
                    if d > max_val:
                        ok = False
                        break
                    if (cur_mask >> d) & 1 or (used_mask >> d) & 1:
                        ok = False
                        break
                    new_mask |= 1 << d
                if not ok:
                    continue
                # Forward check: ensure we can still complete the row
                # (light check: remaining positions must fit in max_val)
                remaining = K1 - len(cur) - 1
                if remaining > 0 and x + remaining > max_val:
                    continue
                res = bt(pos + 1, cur + [x], cur_mask | new_mask)
                if res is not None:
                    return res
            return None

        return bt(0, row, 0)

    # ---------- Build full solution with backtracking over rows ----------
    def build_solution(target_scope, rng, hint_rows=None, max_rows_to_try=None):
        """Build n_rows rows each <= target_scope with distinct differences."""
        rows = [None] * n_rows
        used_mask = 0
        max_attempts = max_rows_to_try if max_rows_to_try else 100

        def bt_row(idx):
            if time.time() > DEADLINE - 5:
                return False
            if idx == n_rows:
                return True

            attempts = 0
            # Try hint first if available
            if hint_rows and hint_rows[idx] is not None:
                hr = hint_rows[idx]
                if max(hr) <= target_scope:
                    m = row_diff_mask(hr)
                    if not (m & used_mask):
                        rows[idx] = hr
                        um = used_mask | m
                        if bt_row_helper(idx + 1, um):
                            return True
                        rows[idx] = None

            while attempts < max_attempts:
                attempts += 1
                if time.time() > DEADLINE - 5:
                    return False
                row = build_row(used_mask, target_scope, rng, idx)
                if row is None:
                    continue
                m = row_diff_mask(row)
                if m & used_mask:
                    continue
                rows[idx] = row
                um = used_mask | m
                if bt_row_helper(idx + 1, um):
                    return True
                rows[idx] = None
            return False

        def bt_row_helper(idx, um):
            if idx == n_rows:
                return True
            attempts = 0
            if hint_rows and hint_rows[idx] is not None:
                hr = hint_rows[idx]
                if max(hr) <= target_scope:
                    m = row_diff_mask(hr)
                    if not (m & um):
                        rows[idx] = hr
                        if bt_row_helper(idx + 1, um | m):
                            return True
                        rows[idx] = None

            while attempts < max_attempts:
                attempts += 1
                if time.time() > DEADLINE - 5:
                    return False
                row = build_row(um, target_scope, rng, idx)
                if row is None:
                    continue
                m = row_diff_mask(row)
                if m & um:
                    continue
                rows[idx] = row
                if bt_row_helper(idx + 1, um | m):
                    return True
                rows[idx] = None
            return False

        return bt_row(0)

    # ---------- Local search to reduce scope ----------
    def local_search(rows, deadline):
        best = [list(r) for r in rows]
        best_sc = scope(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to shrink each row's maximum element
            for i in range(n_rows):
                if time.time() > deadline:
                    break
                row = best[i]
                last = row[-1]
                if last <= lb:
                    continue
                # Collect all differences used by other rows
                others_mask = 0
                for j in range(n_rows):
                    if j == i:
                        continue
                    others_mask |= row_diff_mask(best[j])
                # Differences within row excluding last element
                inner_mask = 0
                for a in range(len(row) - 1):
                    for b in range(a + 1, len(row) - 1):
                        inner_mask |= 1 << (row[b] - row[a])

                for nv in range(last - 1, row[-2], -1):
                    if nv <= row[-2]:
                        break
                    # Check new differences
                    ok = True
                    for a in row[:-1]:
                        d = nv - a
                        if d > best_sc:
                            continue
                        if (others_mask >> d) & 1 or (inner_mask >> d) & 1:
                            ok = False
                            break
                    if ok:
                        new_row = row[:-1] + [nv]
                        if is_valid(best[:i] + [new_row] + best[i+1:]):
                            best[i] = new_row
                            new_sc = scope(best)
                            if new_sc < best_sc:
                                best_sc = new_sc
                                improved = True
                            break
            # Try replacing a row with a smaller one
            if not improved:
                for i in range(n_rows):
                    if time.time() > deadline:
                        break
                    others_mask = 0
                    for j in range(n_rows):
                        if j == i:
                            continue
                        others_mask |= row_diff_mask(best[j])
                    cur_max = max(best[i])
                    for trial_scope in range(cur_max - 1, lb, -1):
                        if trial_scope < cur_max - 3:  # limit search depth
                            break
                        new_row = build_row(others_mask, trial_scope, random.Random())
                        if new_row is not None:
                            m = row_diff_mask(new_row)
                            if not (m & others_mask):
                                candidate = best[:i] + [new_row] + best[i+1:]
                                if is_valid(candidate):
                                    best = candidate
                                    best_sc = scope(best)
                                    improved = True
                                    break
        return best, best_sc

    # ---------- Main search ----------
    rng = random.Random(42)
    best_rows = None
    best_sc = float('inf')

    # Phase 1: find any feasible solution with moderate scope
    init_scope = max(lb + 30, 120)
    for attempt in range(30):
        if time.time() > DEADLINE - 30:
            break
        res = build_solution(init_scope, rng, max_rows_to_try=50)
        if res is not None:
            best_rows = res
            best_sc = scope(res)
            break

    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < K1:
                x += 1
                nd = {x - a for a in row}
                if not nd & used:
                    row.append(x)
                    used |= nd
            rows.append(row)
        return rows

    # Phase 2: local search
    if time.time() < DEADLINE - 20:
        best_rows, best_sc = local_search(best_rows, DEADLINE - 20)

    # Phase 3: iteratively decrease scope
    cur_sc = best_sc
    while cur_sc > lb and time.time() < DEADLINE - 15:
        target = cur_sc - 1
        found = False
        for attempt in range(20):
            if time.time() > DEADLINE - 15:
                break
            res = build_solution(target, rng, hint_rows=best_rows, max_rows_to_try=80)
            if res is not None and scope(res) <= target:
                if time.time() < DEADLINE - 10:
                    res, ns = local_search(res, DEADLINE - 10)
                    if ns < best_sc:
                        best_rows = res
                        best_sc = ns
                        cur_sc = ns
                    else:
                        best_rows = res
                        best_sc = target
                        cur_sc = target
                else:
                    best_rows = res
                    best_sc = target
                    cur_sc = target
                found = True
                break
        if not found:
            break

    # Phase 4: final local search if time remains
    if time.time() < DEADLINE - 5:
        best_rows, best_sc = local_search(best_rows, DEADLINE - 5)

    return best_rows
