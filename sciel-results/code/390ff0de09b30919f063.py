import time
import random
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    k1 = k + 1

    # Lower bound: need n_rows * k*(k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k1:
                return False
            if r[0] != 0:
                return False
            for i in range(k1):
                for j in range(i + 1, k1):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(k1):
                for j in range(i + 1, k1):
                    used.add(r[j] - r[i])
        return used

    def greedy_construct(seed_val, max_scope=None):
        """Construct rows using greedy with a compact strategy."""
        if max_scope is None:
            max_scope = lb + 100
        random.seed(seed_val)
        used = set()
        rows = []
        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k1:
                last = row[-1]
                found = False
                window = 20 + len(row) * 8
                search_end = min(last + window, max_scope)
                candidates = list(range(last + 1, search_end + 1))
                random.shuffle(candidates)
                for x in candidates:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                if not found:
                    for x in range(last + 1, max_scope + 1):
                        valid = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d in used:
                                valid = False
                                break
                            new_diffs.append(d)
                        if valid:
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                    if not found:
                        return None, float('inf')
            rows.append(row)
        scope = max(max(r) for r in rows)
        return rows, scope

    # 1. Get initial solution
    best_rows = None
    best_scope = float('inf')
    for s in range(40):
        if time.time() > deadline - 70:
            break
        rows, scope = greedy_construct(s)
        if rows and scope < best_scope:
            best_scope = scope
            best_rows = rows

    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k1:
                x = row[-1] + 1
                while True:
                    x += 1
                    if not any((x - a) in used for a in row):
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            best_rows.append(row)
        best_scope = max(max(r) for r in best_rows)

    # 2. Simulated Annealing
    current_rows = [list(r) for r in best_rows]
    diff_counts = {}
    conflict_count = 0

    def add_diff(d):
        nonlocal conflict_count
        if d in diff_counts:
            if diff_counts[d] == 1:
                conflict_count += 1
            diff_counts[d] += 1
        else:
            diff_counts[d] = 1

    def remove_diff(d):
        nonlocal conflict_count
        if d in diff_counts:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
            elif diff_counts[d] == 1:
                conflict_count -= 1

    for r in current_rows:
        for i in range(k1):
            for j in range(i + 1, k1):
                add_diff(r[j] - r[i])

    current_scope = max(max(r) for r in current_rows)
    current_cost = current_scope + 1000 * conflict_count
    best_cost = current_cost
    best_rows_sa = [list(r) for r in current_rows]
    best_scope_sa = current_scope

    T = 15.0
    T_min = 0.1
    cooling = 0.9995
    steps = 0
    while time.time() < deadline - 25:
        if T < T_min:
            T = 5.0
            cooling = 0.9997
            if steps > 150000:
                break
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        p_idx = random.randint(1, k1 - 1)
        old_val = row[p_idx]
        diffs_to_remove = [abs(row[i] - old_val) for i in range(k1) if i != p_idx]
        for d in diffs_to_remove:
            remove_diff(d)
        row.pop(p_idx)
        new_val = None
        new_r_idx = r_idx
        if random.random() < 0.7:
            max_search = min(best_scope_sa + 30, 250)
            found = False
            for _ in range(8):
                cand = random.randint(1, max_search)
                if cand in row:
                    continue
                valid = True
                new_diffs = []
                for a in row:
                    d = abs(cand - a)
                    if d in diff_counts:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    pos = 0
                    while pos < len(row) and row[pos] < cand:
                        pos += 1
                    row.insert(pos, cand)
                    for d in new_diffs:
                        add_diff(d)
                    new_val = cand
                    found = True
                    break
            if not found:
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue
        else:
            new_r_idx = random.randint(0, n_rows - 1)
            while new_r_idx == r_idx:
                new_r_idx = random.randint(0, n_rows - 1)
            cand = old_val
            target_row = current_rows[new_r_idx]
            if cand in target_row:
                cand = random.randint(1, best_scope_sa + 20)
                if cand in target_row:
                    pos = 0
                    while pos < len(row) and row[pos] < old_val:
                        pos += 1
                    row.insert(pos, old_val)
                    for d in diffs_to_remove:
                        add_diff(d)
                    continue
            valid = True
            new_diffs = []
            for a in target_row:
                d = abs(cand - a)
                if d in diff_counts:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                pos = 0
                while pos < len(target_row) and target_row[pos] < cand:
                    pos += 1
                target_row.insert(pos, cand)
                for d in new_diffs:
                    add_diff(d)
                new_val = cand
            else:
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue
        new_scope = max(max(r) for r in current_rows)
        new_cost = new_scope + 1000 * conflict_count
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            current_scope = new_scope
            if is_valid(current_rows):
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope_sa = current_scope
                    best_rows_sa = [list(r) for r in current_rows]
        else:
            if new_r_idx != r_idx:
                target_row = current_rows[new_r_idx]
                idx_new = target_row.index(new_val)
                target_row.pop(idx_new)
                for a in target_row:
                    remove_diff(abs(a - new_val))
            else:
                row = current_rows[r_idx]
                idx_new = row.index(new_val)
                row.pop(idx_new)
                for a in row:
                    remove_diff(abs(a - new_val))
            row = current_rows[r_idx]
            pos = 0
            while pos < len(row) and row[pos] < old_val:
                pos += 1
            row.insert(pos, old_val)
            for d in diffs_to_remove:
                add_diff(d)
        T *= cooling
        steps += 1

    # 3. Pull-to-smallest: for each row, reconstruct with the same difference
    # multiset at the minimal feasible scope. This is the key refinement.
    def pull_to_smallest(rows):
        """Reconstruct each row to minimal scope using the same difference multiset."""
        result = []
        for r in rows:
            # Compute the multiset of differences for this row
            diffs = []
            for i in range(k1):
                for j in range(i + 1, k1):
                    diffs.append(r[j] - r[i])
            # Reconstruct row greedily with smallest feasible values
            new_row = [0]
            # We need to pick k more values such that all pairwise diffs match the multiset
            # Greedy: at each step, find smallest x > last such that adding x
            # keeps consistency with the remaining difference multiset.
            # Simpler robust approach: use the known multiset to guide.
            # Use a backtracking with the difference multiset as a soft target.
            new_row = _reconstruct_row(diffs, lb)
            if new_row is None:
                new_row = list(r)
            result.append(new_row)
        return result

    def _reconstruct_row(diffs, max_scope):
        """Reconstruct a single row of k+1 values starting at 0 with the given
        difference multiset, minimizing the largest value."""
        # Build a counter of the difference multiset
        from collections import Counter
        target = Counter(diffs)
        row = [0]
        # The smallest value in the row is 0. The next value is the smallest
        # difference that can appear as a "first step". We try to build the row
        # by choosing the next element as the smallest feasible value.
        # A value x is feasible as the next element if, for all existing a in row,
        # (x-a) is present in target with remaining count, and we can complete.
        # We do a greedy with backtracking limited in depth.
        best = None
        def backtrack(cur_row, cur_target):
            nonlocal best
            if len(cur_row) == k1:
                # All differences used up?
                if sum(cur_target.values()) == 0:
                    scope = cur_row[-1]
                    if best is None or scope < best[0]:
                        best = (scope, list(cur_row))
                return
            last = cur_row[-1]
            # Candidate next values: from last+1 up to some bound
            # The next value must be such that diffs (x - a) for a in cur_row
            # are all in cur_target with positive count.
            # Bound: the remaining number of elements is k1 - len(cur_row).
            # The largest possible next value is bounded by max_scope.
            upper = max_scope
            found_any = False
            for x in range(last + 1, upper + 1):
                # Check all diffs x - a are in cur_target
                ok = True
                needed = []
                for a in cur_row:
                    d = x - a
                    if cur_target.get(d, 0) <= 0:
                        ok = False
                        break
                    needed.append(d)
                if not ok:
                    continue
                found_any = True
                # Consume
                for d in needed:
                    cur_target[d] -= 1
                backtrack(cur_row + [x], cur_target)
                for d in needed:
                    cur_target[d] += 1
                # Once we have a solution, we can stop exploring larger x
                # because we want the smallest scope. But we must be careful:
                # a smaller x here might lead to a larger final scope. So we
                # continue but prune if we already found a good one.
                if best is not None and x >= best[0]:
                    # no point continuing with larger x at this level
                    break
            # If we found a solution already, stop
            if best is not None:
                return
            if not found_any:
                return
        backtrack(row, Counter(target))
        if best is not None:
            return best[1]
        return None

    # Apply pull-to-smallest to the SA result
    improved = pull_to_smallest(best_rows_sa)
    improved_scope = max(max(r) for r in improved)
    if is_valid(improved) and improved_scope < best_scope_sa:
        best_rows_sa = improved
        best_scope_sa = improved_scope

    # 4. Backtracking search for lower target scopes
    def backtracking_search(max_scope, seed_offset=0):
        rng = random.Random(seed_offset * 12345 + 678)
        rows = []
        used_diffs = [False] * (max_scope + 1)

        def build_row(pos, current_mark, row_so_far):
            if time.time() > deadline - 3:
                return None
            if pos == k1:
                return list(row_so_far)
            max_x = max_scope - (k1 - pos - 1)
            if max_x <= current_mark:
                return None
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
            n_smallest = min(12, len(valid_candidates))
            candidates_to_try = valid_candidates[:n_smallest]
            if len(valid_candidates) > n_smallest and rng.random() < 0.25:
                extra = rng.sample(valid_candidates[n_smallest:], min(6, len(valid_candidates) - n_smallest))
                candidates_to_try = candidates_to_try + extra
            rng.shuffle(candidates_to_try)
            for x in candidates_to_try:
                new_diffs = []
                for a in row_so_far:
                    d = x - a
                    new_diffs.append(d)
                    used_diffs[d] = True
                res = build_row(pos + 1, x, row_so_far + [x])
                if res is not None:
                    return res
                for d in new_diffs:
                    used_diffs[d] = False
            return None

        for i in range(n_rows):
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            for j in range(k1):
                for i2 in range(j + 1, k1):
                    d = row[i2] - row[j]
                    if not used_diffs[d]:
                        used_diffs[d] = True
        return rows

    # Try target scopes below current best
    target = best_scope_sa - 1
    attempt = 0
    while time.time() < deadline - 6 and target >= lb:
        found = False
        for _ in range(30):
            if time.time() > deadline - 6:
                break
            sol = backtracking_search(target, attempt)
            if sol:
                ns = max(max(r) for r in sol)
                if ns < best_scope_sa:
                    best_scope_sa = ns
                    best_rows_sa = sol
                    found = True
                    # Pull-to-smallest on the new solution
                    imp = pull_to_smallest(best_rows_sa)
                    if is_valid(imp):
                        ns2 = max(max(r) for r in imp)
                        if ns2 < best_scope_sa:
                            best_scope_sa = ns2
                            best_rows_sa = imp
                    if best_scope_sa <= lb:
                        return best_rows_sa
                    break
            attempt += 1
        if not found:
            target -= 1
        else:
            target = best_scope_sa - 1

    # 5. Final polish: iterative pull-to-smallest
    for _ in range(20):
        if time.time() > deadline - 3:
            break
        imp = pull_to_smallest(best_rows_sa)
        if not is_valid(imp):
            break
        ns = max(max(r) for r in imp)
        if ns < best_scope_sa:
            best_scope_sa = ns
            best_rows_sa = imp
        else:
            break

    # Validate final result
    if not is_valid(best_rows_sa):
        if is_valid(best_rows):
            best_rows_sa = best_rows
        else:
            used = set()
            best_rows_sa = []
            for _ in range(n_rows):
                row = [0]
                while len(row) < k1:
                    x = row[-1] + 1
                    while True:
                        x += 1
                        if not any((x - a) in used for a in row):
                            break
                    row.append(x)
                    used.update({x - a for a in row if a != x})
                best_rows_sa.append(row)

    return best_rows_sa
