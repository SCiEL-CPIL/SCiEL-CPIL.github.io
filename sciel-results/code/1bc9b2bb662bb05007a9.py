import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs  # counting lower bound (scope must be at least this)
    k1 = k + 1

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def _compute_all_diffs(rows):
        """Compute the full set of all positive differences across all rows."""
        s = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
        return s

    def validate_rows(rows):
        """Check all differences are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_baseline():
        """Fast greedy: smallest valid x each step. Provides a quick starting point."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def greedy_construct(seed, strategy='basic'):
        """
        Construct rows using greedy with randomization.
        strategy: 'basic', 'random_order', 'compact', 'spread'
        """
        rng = random.Random(seed)
        used = set()
        rows = []

        if strategy == 'random_order':
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            row_slots = [None] * n_rows

            for row_idx in row_order:
                row = [0]
                while len(row) < k1:
                    last = row[-1]
                    found = False

                    if strategy == 'compact':
                        search_end = last + 30 + rng.randint(0, 20)
                    elif strategy == 'spread':
                        search_end = last + 50 + rng.randint(0, 40)
                    else:
                        search_end = last + 40 + rng.randint(0, 30)

                    search_end = min(search_end, lb + 200)

                    candidates = list(range(last + 1, search_end))
                    if rng.random() < 0.6:
                        for i in range(len(candidates) - 1, 0, -1):
                            if rng.random() < 0.1:
                                candidates[i], candidates[i-1] = candidates[i-1], candidates[i]
                    else:
                        rng.shuffle(candidates)

                    for x in candidates:
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break

                    if not found:
                        for x in range(last + 1, lb + 300):
                            new_diffs = [x - a for a in row]
                            if any(d in used for d in new_diffs):
                                continue
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                        if not found:
                            return None
                row_slots[row_idx] = row

            if any(r is None for r in row_slots):
                return None
            return row_slots
        else:
            for row_idx in range(n_rows):
                row = [0]
                while len(row) < k1:
                    last = row[-1]
                    found = False

                    if strategy == 'compact':
                        search_end = last + 25 + rng.randint(0, 15)
                    elif strategy == 'spread':
                        search_end = last + 50 + rng.randint(0, 40)
                    else:
                        search_end = last + 35 + rng.randint(0, 25)

                    search_end = min(search_end, lb + 200)

                    candidates = list(range(last + 1, search_end))
                    if rng.random() < 0.5:
                        pass
                    else:
                        rng.shuffle(candidates)

                    for x in candidates:
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break

                    if not found:
                        for x in range(last + 1, lb + 300):
                            new_diffs = [x - a for a in row]
                            if any(d in used for d in new_diffs):
                                continue
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                        if not found:
                            return None
                rows.append(row)
            return rows

    def backtracking_search(max_scope, seed, time_limit=3.0):
        """
        Backtracking search to build rows within a given max_scope.
        Uses a boolean array for difference tracking (faster than set for small ranges).
        """
        rng = random.Random(seed)
        used_diffs = [False] * (max_scope + 1)
        rows = []
        t_start = time.time()

        def build_row(pos, last_val, row_so_far):
            nonlocal used_diffs
            if time.time() - t_start > time_limit:
                return None
            if pos == k1:
                return list(row_so_far)

            # Prune: need (k1 - pos) more elements, each >= 1 apart
            max_x = max_scope - (k1 - pos - 1)
            if max_x <= last_val:
                return None

            valid = []
            for x in range(last_val + 1, max_x + 1):
                ok = True
                for a in row_so_far:
                    if used_diffs[x - a]:
                        ok = False
                        break
                if ok:
                    valid.append(x)

            if not valid:
                return None

            # Bias toward smaller values, but shuffle a window for diversity
            n_take = min(12, len(valid))
            window = valid[:n_take]
            if len(valid) > n_take and rng.random() < 0.4:
                extra = rng.sample(valid[n_take:], min(5, len(valid) - n_take))
                window.extend(extra)
            rng.shuffle(window)

            for x in window:
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

        for _ in range(n_rows):
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            for i in range(k1):
                for j in range(i + 1, k1):
                    used_diffs[row[j] - row[i]] = True
        return rows

    def local_improve(rows, seed, max_iters=2000):
        """Try to reduce scope by shifting elements left to smaller valid positions."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)

        used_set = _compute_all_diffs(rows)

        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break

            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue

            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue

            # Remove this element and its diffs from used_set
            for a in row[:elem_idx]:
                used_set.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used_set.discard(a - old_val)
            row.pop(elem_idx)

            # Try to re-insert at a smaller position
            found = False
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1

            for new_val in range(min_val, max_val + 1):
                valid = True
                for a in row[:elem_idx]:
                    d = new_val - a
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    for a in row[elem_idx:]:
                        d = a - new_val
                        if d in used_set:
                            valid = False
                            break
                if valid:
                    row.insert(elem_idx, new_val)
                    for a in row[:elem_idx]:
                        used_set.add(new_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.add(a - new_val)
                    found = True
                    break

            if found:
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used_set.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.add(a - old_val)

        return rows, scope

    def compact_pass(rows):
        """Systematic compaction: repeatedly try to shift each element to its smallest valid position."""
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        used_set = _compute_all_diffs(rows)

        improved_any = True
        max_rounds = 300

        for _ in range(max_rounds):
            if time.time() > deadline - 2:
                break
            if not improved_any:
                break
            improved_any = False

            for row_idx in range(n_rows):
                if time.time() > deadline - 2:
                    break
                row = rows[row_idx]
                for elem_pos in range(len(row) - 1, 0, -1):
                    old_val = row[elem_pos]
                    if old_val <= 1:
                        continue

                    for a in row[:elem_pos]:
                        used_set.discard(old_val - a)
                    for a in row[elem_pos + 1:]:
                        used_set.discard(a - old_val)
                    row.pop(elem_pos)

                    min_val = row[elem_pos - 1] + 1 if elem_pos > 0 else 1
                    found = False

                    for new_val in range(min_val, old_val):
                        valid = True
                        for a in row[:elem_pos]:
                            d = new_val - a
                            if d in used_set:
                                valid = False
                                break
                        if valid:
                            for a in row[elem_pos:]:
                                d = a - new_val
                                if d in used_set:
                                    valid = False
                                    break
                        if valid:
                            row.insert(elem_pos, new_val)
                            for a in row[:elem_pos]:
                                used_set.add(new_val - a)
                            for a in row[elem_pos + 1:]:
                                used_set.add(a - new_val)
                            found = True
                            break

                    if found:
                        new_scope = get_scope(rows)
                        if new_scope < scope:
                            scope = new_scope
                            improved_any = True
                    else:
                        row.insert(elem_pos, old_val)
                        for a in row[:elem_pos]:
                            used_set.add(old_val - a)
                        for a in row[elem_pos + 1:]:
                            used_set.add(a - old_val)

        return rows, scope

    def random_restart_construct(seed):
        """Build with completely random choices."""
        rng = random.Random(seed)
        used = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            while len(row) < k1:
                last = row[-1]
                found = False
                jump = rng.randint(1, 60)
                x = last + jump
                for candidate in range(x - 10, x + 10):
                    if candidate <= last:
                        continue
                    new_diffs = [candidate - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(candidate)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    for candidate in range(last + 1, last + 200):
                        new_diffs = [candidate - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(candidate)
                        used.update(new_diffs)
                        found = True
                        break
                    if not found:
                        return None
            rows.append(row)
        return rows

    # === Phase 1: Quick greedy baseline ===
    base_rows = greedy_baseline()
    if base_rows:
        base_scope = get_scope(base_rows)
        best_rows = [list(r) for r in base_rows]
        best_scope = base_scope

    # === Phase 2: Backtracking search at various target scopes ===
    # Try to find solutions at progressively tighter scopes
    target_scopes = []
    if best_scope < float('inf'):
        # Start from current best and go down toward lb
        for s in range(best_scope, lb - 1, -1):
            target_scopes.append(s)
    else:
        target_scopes = [lb + 50, lb + 30, lb + 20, lb + 10, lb]

    # Limit backtracking attempts to avoid spending too much time
    bt_budget = 40.0
    bt_start = time.time()
    for target in target_scopes:
        if time.time() - bt_start > bt_budget:
            break
        if target < lb:
            break
        # Try a few seeds at this target
        for seed in range(5):
            if time.time() - bt_start > bt_budget:
                break
            if time.time() > deadline - 30:
                break
            rows = backtracking_search(target, 1000 + seed * 137, time_limit=2.0)
            if rows is not None:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                    break  # Move to next (tighter) target
            # If we found a solution at this scope, try tighter
            if rows is not None and scope <= target:
                break

    # === Phase 3: Greedy constructions with various strategies ===
    strategies = ['basic', 'compact', 'spread', 'random_order']
    greedy_budget = 25.0
    greedy_start = time.time()
    for i in range(3000):
        if time.time() - greedy_start > greedy_budget:
            break
        if time.time() > deadline - 30:
            break
        strategy = strategies[i % len(strategies)]
        rows = greedy_construct(i * 31 + 7, strategy)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # === Phase 4: Local improvement on best found ===
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 42, max_iters=1500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # === Phase 5: Random restarts ===
    rr_budget = 15.0
    rr_start = time.time()
    for i in range(1000):
        if time.time() - rr_start > rr_budget:
            break
        if time.time() > deadline - 25:
            break
        rows = random_restart_construct(100000 + i * 97)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # === Phase 6: Local improvement again ===
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 12345, max_iters=1000)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # === Phase 7: Systematic compaction ===
    if best_rows is not None:
        compacted, comp_scope = compact_pass(best_rows)
        if comp_scope < best_scope:
            best_scope = comp_scope
            best_rows = compacted

    # === Phase 8: More backtracking at the current best scope (tighten) ===
    if best_rows is not None and time.time() < deadline - 15:
        # Try to find a solution at best_scope - 1, best_scope - 2, etc.
        for offset in range(1, 8):
            target = best_scope - offset
            if target < lb:
                break
            if time.time() > deadline - 10:
                break
            for seed in range(3):
                if time.time() > deadline - 8:
                    break
                rows = backtracking_search(target, 50000 + offset * 1000 + seed * 73, time_limit=1.5)
                if rows is not None:
                    scope = get_scope(rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                    break

    # === Phase 9: Final local improvement + compaction ===
    if best_rows is not None and time.time() < deadline - 5:
        improved, imp_scope = local_improve(best_rows, 99999, max_iters=800)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

        if time.time() < deadline - 3:
            compacted, comp_scope = compact_pass(best_rows)
            if comp_scope < best_scope:
                best_scope = comp_scope
                best_rows = compacted

    # Final validation
    if best_rows is not None:
        if not validate_rows(best_rows):
            # Fallback to greedy baseline if something went wrong
            fb = greedy_baseline()
            if fb:
                best_rows = fb
                best_scope = get_scope(fb)

    return best_rows


# Quick self-test
if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        # Validate
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
