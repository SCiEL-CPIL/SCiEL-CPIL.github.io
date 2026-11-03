import time
import random
import sys
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 145.0
    m = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
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

    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    # ---- Row regeneration ----
    def gen_row(used, k_val, randomize, p_smallest=0.8, window=4, rng=None, cap=400):
        if rng is None:
            rng = random
        row = [0]
        cur = set()
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            forbidden = used | cur
            while x < cap:
                new_d = {x - a for a in row}
                if not (new_d & forbidden):
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            if not candidates:
                return None
            if randomize:
                if rng.random() < p_smallest:
                    nxt = candidates[0]
                else:
                    nxt = rng.choice(candidates)
            else:
                nxt = candidates[0]
            row.append(nxt)
            cur |= {nxt - a for a in row[:-1]}
        return row

    # ---- Deterministic tightening ----
    def tighten(rows, max_iter=200):
        rows = [list(r) for r in rows]
        for _ in range(max_iter):
            if time.time() > deadline - 20:
                break
            improved = False
            for r in range(n_rows):
                row = rows[r]
                for pos in range(len(row) - 1, 0, -1):
                    target = row[pos - 1] + 1
                    if target >= row[pos]:
                        continue
                    cand = row[:pos] + [target] + row[pos + 1:]
                    ok = True
                    for i in range(1, len(cand)):
                        if cand[i] <= cand[i - 1]:
                            ok = False
                            break
                    if not ok:
                        continue
                    idiffs = set()
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            d = cand[j] - cand[i]
                            if d in idiffs:
                                ok = False
                                break
                            idiffs.add(d)
                        if not ok:
                            break
                    if not ok:
                        continue
                    for idx in range(n_rows):
                        if idx == r:
                            continue
                        od = row_diffs(rows[idx])
                        if idiffs & od:
                            ok = False
                            break
                    if ok:
                        row[pos] = target
                        improved = True
            if not improved:
                break
        return rows

    # ---- Greedy initial solution ----
    def greedy_initial(rng):
        used = set()
        rows = []
        prev_first = 0
        for r in range(n_rows):
            row = None
            attempts = 0
            while row is None and attempts < 200:
                attempts += 1
                row = gen_row(used, k, randomize=True, p_smallest=0.85,
                              window=3, rng=rng, cap=400)
                if row is not None:
                    if r > 0 and row[1] <= prev_first:
                        row = None
                        continue
            if row is None:
                row = gen_row(used, k, randomize=False, rng=rng, cap=500)
            rows.append(row)
            prev_first = row[1]
            used |= row_diffs(row)
        return rows

    # ---- Bitset-accelerated backtracking with MRV ----
    def backtrack_scope_search(target_scope, rng, max_nodes=100000):
        """
        Backtracking search to find a solution with max entry <= target_scope.
        Uses boolean array for O(1) difference checking.
        """
        L = k + 1
        used_diffs = [False] * (target_scope + 1)
        rows = []
        nodes_visited = 0

        def build_row(pos, current_mark, row_so_far):
            nonlocal nodes_visited
            if time.time() > deadline - 5:
                return None
            nodes_visited += 1
            if nodes_visited > max_nodes:
                return None

            if pos == L:
                return list(row_so_far)

            # Prune: need at least (L - pos - 1) more values
            max_x = target_scope - (L - pos - 1)
            if max_x <= current_mark:
                return None

            # Find valid candidates - prefer smaller values
            valid_candidates = []
            for x in range(current_mark + 1, max_x + 1):
                conflict = False
                for a in row_so_far:
                    d = x - a
                    if used_diffs[d]:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
                if len(valid_candidates) >= 40:
                    break

            if not valid_candidates:
                return None

            # Try smaller candidates first with some randomness
            n_try = min(15, len(valid_candidates))
            if rng.random() < 0.7:
                candidates_to_try = valid_candidates[:n_try]
            else:
                candidates_to_try = rng.sample(valid_candidates, min(n_try, len(valid_candidates)))

            for x in candidates_to_try:
                # Check all diffs
                new_diffs = [x - a for a in row_so_far]
                # Verify no conflicts
                ok = True
                for d in new_diffs:
                    if used_diffs[d]:
                        ok = False
                        break
                if not ok:
                    continue

                # Mark diffs
                for d in new_diffs:
                    used_diffs[d] = True

                result = build_row(pos + 1, x, row_so_far + [x])
                if result is not None:
                    return result

                # Unmark diffs
                for d in new_diffs:
                    used_diffs[d] = False

            return None

        for r in range(n_rows):
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)

        return rows

    # ---- Aggressive compaction: pull each element to minimum feasible ----
    def compact_rows(rows, max_iter=500):
        """
        For each row, try to reduce each element to the smallest feasible value
        given the other rows' differences. This is a deterministic tightening
        that goes beyond simple tighten().
        """
        rows = [list(r) for r in rows]
        other_diffs_cache = []
        for r in range(n_rows):
            s = set()
            for i in range(n_rows):
                if i == r:
                    continue
                s |= row_diffs(rows[i])
            other_diffs_cache.append(s)

        for _ in range(max_iter):
            if time.time() > deadline - 15:
                break
            improved = False
            for r in range(n_rows):
                row = rows[r]
                other = other_diffs_cache[r]
                for pos in range(1, len(row)):
                    # Try to reduce row[pos]
                    min_val = row[pos - 1] + 1
                    if min_val >= row[pos]:
                        continue
                    found = False
                    for new_val in range(min_val, row[pos]):
                        # Check if new_val works
                        test_row = row[:]
                        test_row[pos] = new_val
                        # Check increasing
                        ok = True
                        for i in range(1, len(test_row)):
                            if test_row[i] <= test_row[i - 1]:
                                ok = False
                                break
                        if not ok:
                            continue
                        # Check internal diffs
                        idiffs = set()
                        for i in range(len(test_row)):
                            for j in range(i + 1, len(test_row)):
                                d = test_row[j] - test_row[i]
                                if d in idiffs:
                                    ok = False
                                    break
                                idiffs.add(d)
                            if not ok:
                                break
                        if not ok:
                            continue
                        # Check cross-row diffs
                        if idiffs & other:
                            continue
                        # Accept
                        row[pos] = new_val
                        found = True
                        improved = True
                        # Update other_diffs_cache
                        new_idiffs = set()
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                new_idiffs.add(row[j] - row[i])
                        for i in range(n_rows):
                            if i != r:
                                other_diffs_cache[i] |= new_idiffs
                        break
                    if found:
                        # Rebuild other_diffs_cache for this row
                        pass
                # Rebuild other_diffs_cache for row r
                if improved:
                    my_diffs = row_diffs(row)
                    for i in range(n_rows):
                        if i != r:
                            other_diffs_cache[i] = other_diffs_cache[i] | my_diffs
            if not improved:
                break

        return rows

    # ---- Simulated annealing with row regeneration ----
    def anneal(rows, rng, max_iter=20000):
        rows = [list(r) for r in rows]
        used = set()
        for r in rows:
            used |= row_diffs(r)
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        T = 8.0
        cooling = 0.998
        min_T = 0.3

        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            r_idx = rng.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_diffs = row_diffs(old_row)
            used -= old_diffs

            p_smallest = 0.6 + 0.3 * rng.random()
            window = 2 + rng.randint(0, 3)
            new_row = gen_row(used, k, randomize=True, p_smallest=p_smallest,
                              window=window, rng=rng, cap=300)

            if new_row is None:
                used |= old_diffs
                continue

            new_diffs = row_diffs(new_row)
            other_max = 0
            for i in range(n_rows):
                if i != r_idx:
                    other_max = max(other_max, rows[i][-1])
            new_global = max(other_max, new_row[-1])
            delta = new_global - current_max

            if delta < 0:
                rows[r_idx] = new_row
                used |= new_diffs
                current_max = new_global
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
            else:
                if delta == 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / max(T, 0.01))
                if rng.random() < prob:
                    rows[r_idx] = new_row
                    used |= new_diffs
                    current_max = new_global
                else:
                    used |= old_diffs

            T = max(min_T, T * cooling)

        return best_rows

    # ---- Local element perturbation ----
    def local_refine(rows, rng, max_iter=5000):
        rows = [list(r) for r in rows]
        used = set()
        for r in rows:
            used |= row_diffs(r)
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            for delta in [-1, -2, 1, 2]:
                new_val = row[pos] + delta
                if new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                test_row = row[:]
                test_row[pos] = new_val
                idiffs = set()
                ok = True
                for i in range(len(test_row)):
                    for j in range(i + 1, len(test_row)):
                        d = test_row[j] - test_row[i]
                        if d in idiffs:
                            ok = False
                            break
                        idiffs.add(d)
                    if not ok:
                        break
                if not ok:
                    continue
                conflict = False
                for idx in range(n_rows):
                    if idx == r_idx:
                        continue
                    if idiffs & row_diffs(rows[idx]):
                        conflict = True
                        break
                if conflict:
                    continue
                other_max = 0
                for i in range(n_rows):
                    if i != r_idx:
                        other_max = max(other_max, rows[i][-1])
                new_scope = max(other_max, test_row[-1])
                if new_scope <= current_max:
                    rows[r_idx] = test_row
                    current_max = new_scope
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r) for r in rows]
                    break
        return best_rows

    # ---- Main search ----
    rng = random.Random(42)

    # 1. Initial solution
    best_rows = greedy_initial(rng)
    if best_rows is None or not is_valid(best_rows):
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = gen_row(used, k, randomize=False, rng=rng, cap=600)
            best_rows.append(row)
            used |= row_diffs(row)

    # 2. Tighten
    best_rows = tighten(best_rows)
    if not is_valid(best_rows):
        best_rows = greedy_initial(rng)
    best_scope = scope_of(best_rows)

    # 3. Anneal
    rows = anneal(best_rows, rng, max_iter=15000)
    if is_valid(rows):
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s

    # 4. Local refine
    rows = local_refine(best_rows, rng, max_iter=4000)
    if is_valid(rows):
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s

    # 5. Tighten again
    best_rows = tighten(best_rows)
    if is_valid(best_rows):
        best_scope = scope_of(best_rows)

    # 5b. Aggressive compaction
    best_rows = compact_rows(best_rows)
    if is_valid(best_rows):
        best_scope = scope_of(best_rows)

    # 6. Bitset backtracking for scope reduction
    target = best_scope - 1
    max_bt_attempts = 25
    bt_attempts = 0

    while target > 100 and bt_attempts < max_bt_attempts and time.time() < deadline - 30:
        bt_attempts += 1
        rng_bt = random.Random(10000 + bt_attempts * 7919)

        time_left = deadline - 30 - time.time()
        est_nodes = int(time_left * 800)
        if est_nodes < 5000:
            est_nodes = 5000
        if est_nodes > 80000:
            est_nodes = 80000

        sol = backtrack_scope_search(target, rng_bt, max_nodes=est_nodes)

        if sol and is_valid(sol):
            new_scope = scope_of(sol)
            if new_scope < best_scope:
                best_rows = sol
                best_scope = new_scope
                best_rows = tighten(best_rows)
                if is_valid(best_rows):
                    best_scope = scope_of(best_rows)
                best_rows = compact_rows(best_rows)
                if is_valid(best_rows):
                    best_scope = scope_of(best_rows)
                rows_ref = local_refine(best_rows, rng_bt, max_iter=2000)
                if is_valid(rows_ref):
                    s_ref = scope_of(rows_ref)
                    if s_ref < best_scope:
                        best_rows = rows_ref
                        best_scope = s_ref
                target = best_scope - 1
            else:
                target -= 1
        else:
            target -= 1

    # 7. Diversified restarts
    for trial in range(8):
        if time.time() > deadline - 25:
            break
        rng2 = random.Random(1000 + trial * 137)
        rows = greedy_initial(rng2)
        if rows is None or not is_valid(rows):
            continue
        rows = tighten(rows)
        if not is_valid(rows):
            continue
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s
        rows2 = anneal(rows, rng2, max_iter=8000)
        if is_valid(rows2):
            s2 = scope_of(rows2)
            if s2 < best_scope:
                best_rows = rows2
                best_scope = s2
        rows3 = local_refine(rows2 if is_valid(rows2) else rows, rng2, max_iter=3000)
        if is_valid(rows3):
            s3 = scope_of(rows3)
            if s3 < best_scope:
                best_rows = rows3
                best_scope = s3

    # Final tightening
    best_rows = tighten(best_rows, max_iter=300)
    if not is_valid(best_rows):
        best_rows = greedy_initial(random.Random(7))
        best_rows = tighten(best_rows)

    # Final compaction
    best_rows = compact_rows(best_rows, max_iter=200)
    if not is_valid(best_rows):
        best_rows = tighten(best_rows, max_iter=100)
        if not is_valid(best_rows):
            best_rows = greedy_initial(random.Random(7))
            best_rows = tighten(best_rows)

    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
