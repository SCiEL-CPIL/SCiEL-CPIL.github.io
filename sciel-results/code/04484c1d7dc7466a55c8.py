import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    max_scope_bound = lb + 300  # upper bound for difference lookup

    best_rows = None
    best_scope = float('inf')

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

    def get_scope(rows):
        return max(max(r) for r in rows)

    def compute_used_set(rows):
        """Compute the set of all differences across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def row_diffs(row):
        """Compute all positive differences in a single row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_row(seed, strategy='basic', max_scope=None):
        """
        Construct a single row using greedy with randomization.
        Returns a row (list of k+1 increasing ints starting at 0) or None.
        """
        rng = random.Random(seed)
        row = [0]
        used = set()

        while len(row) < k + 1:
            last = row[-1]
            found = False

            if strategy == 'compact':
                search_end = last + 25 + rng.randint(0, 15)
            elif strategy == 'spread':
                search_end = last + 50 + rng.randint(0, 40)
            elif strategy == 'balanced':
                search_end = last + 35 + rng.randint(0, 25)
            else:
                search_end = last + 40 + rng.randint(0, 30)

            if max_scope is not None:
                search_end = min(search_end, max_scope)

            candidates = list(range(last + 1, search_end))
            if rng.random() < 0.5:
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
                # Wider search
                upper = max_scope if max_scope is not None else lb + 300
                for x in range(last + 1, upper):
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    return None

        return row

    def local_improve_rows(rows, seed, max_iters=200):
        """Try to reduce scope by shifting elements left."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)

        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break

            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue

            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue

            row.pop(elem_idx)

            used_set = set()
            for i, r in enumerate(rows):
                for a in range(len(r)):
                    for b in range(a + 1, len(r)):
                        used_set.add(r[b] - r[a])

            found = False
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1

            for new_val in range(max_val, min_val - 1, -1):
                valid = True
                for a in row[:elem_idx]:
                    d = new_val - a
                    if d in used_set:
                        valid = False
                        break
                if not valid:
                    continue
                for a in row[elem_idx:]:
                    d = a - new_val
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    row.insert(elem_idx, new_val)
                    found = True
                    break

            if found:
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                row.insert(elem_idx, old_val)

        return rows, scope

    def mis_solve(candidates, target_rows):
        """
        Find a set of target_rows non-conflicting candidates from the list,
        minimizing the maximum entry. Uses branch-and-bound with greedy coloring.
        candidates: list of (row, scope, diff_set)
        Returns (selected_rows, scope) or (None, inf)
        """
        if not candidates:
            return None, float('inf')

        # Sort candidates by scope (ascending) to try smaller ones first
        candidates = sorted(candidates, key=lambda x: x[1])

        n = len(candidates)
        if n < target_rows:
            return None, float('inf')

        # Build conflict graph: conflict[i][j] = True if candidates i and j share a diff
        # Use bitsets for efficiency
        conflict = [0] * n
        for i in range(n):
            di = candidates[i][2]
            for j in range(i + 1, n):
                dj = candidates[j][2]
                if di & dj:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)

        # Greedy coloring for upper bound
        def greedy_color(node_set):
            """Assign colors to nodes in node_set, return max color used."""
            colors = {}
            for node in node_set:
                used_colors = set()
                for other in node_set:
                    if other == node:
                        continue
                    if (conflict[node] >> other) & 1:
                        if other in colors:
                            used_colors.add(colors[other])
                c = 0
                while c in used_colors:
                    c += 1
                colors[node] = c
            return max(colors.values()) + 1 if colors else 0

        best_selection = None
        best_scope = float('inf')

        # Branch and bound
        def bnb(selected, remaining, current_scope):
            nonlocal best_selection, best_scope

            if time.time() > deadline - 5:
                return

            if len(selected) == target_rows:
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_selection = list(selected)
                return

            if not remaining:
                return

            # Pruning: if we can't complete, return
            needed = target_rows - len(selected)
            if len(remaining) < needed:
                return

            # Upper bound via greedy coloring
            ub = greedy_color(remaining)
            if len(selected) + ub < target_rows:
                return

            # Try nodes in order of scope
            for idx in remaining:
                if time.time() > deadline - 5:
                    return

                row, scope, diffs = candidates[idx]
                new_scope = max(current_scope, scope)

                # Prune if already worse
                if new_scope >= best_scope:
                    continue

                # Check compatibility with all selected
                compatible = True
                for sel_idx in selected:
                    if (conflict[idx] >> sel_idx) & 1:
                        compatible = False
                        break
                if not compatible:
                    continue

                # Recurse
                new_selected = selected + [idx]
                new_remaining = [r for r in remaining if r != idx]
                bnb(new_selected, new_remaining, new_scope)

        # Initial call
        all_indices = list(range(min(n, 200)))  # Limit to first 200 for speed
        bnb([], all_indices, 0)

        if best_selection is None:
            return None, float('inf')

        rows = [candidates[i][0] for i in best_selection]
        return rows, best_scope

    # Phase 1: Generate diverse row candidates
    strategies = ['basic', 'compact', 'spread', 'balanced']
    candidate_pool = []  # list of (row, scope, diff_set)
    seen_rows = set()

    max_candidates = 500
    for i in range(3000):
        if time.time() > deadline - 40:
            break
        if len(candidate_pool) >= max_candidates:
            break

        strategy = strategies[i % len(strategies)]
        seed = i * 31 + 7
        row = greedy_row(seed, strategy)
        if row is None:
            continue

        key = tuple(row)
        if key in seen_rows:
            continue
        seen_rows.add(key)

        scope = max(row)
        diffs = row_diffs(row)
        candidate_pool.append((row, scope, diffs))

    # Phase 2: Solve MIS on candidate pool
    if len(candidate_pool) >= n_rows:
        # Try with increasing scope limits
        scopes = sorted(set(c[1] for c in candidate_pool))

        # First try with all candidates
        rows, scope = mis_solve(candidate_pool, n_rows)
        if rows is not None:
            best_scope = scope
            best_rows = [list(r) for r in rows]

        # Try with progressively tighter scope bounds
        for max_s in scopes:
            if time.time() > deadline - 15:
                break
            if max_s >= best_scope:
                break

            filtered = [c for c in candidate_pool if c[1] <= max_s]
            if len(filtered) < n_rows:
                continue

            rows, scope = mis_solve(filtered, n_rows)
            if rows is not None and scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]

    # Phase 3: Local improvement
    if best_rows is not None:
        improved, imp_scope = local_improve_rows(best_rows, 42, max_iters=500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

        # More aggressive compaction
        current_rows = [list(r) for r in best_rows]
        current_scope = best_scope
        improved_any = True
        for round_num in range(30):
            if time.time() > deadline - 5:
                break
            if not improved_any:
                break
            improved_any = False

            for row_idx in range(n_rows):
                if time.time() > deadline - 3:
                    break
                row = current_rows[row_idx]
                for elem_pos in range(len(row) - 1, 0, -1):
                    old_val = row[elem_pos]
                    if old_val <= 1:
                        continue

                    row.pop(elem_pos)

                    used_set = set()
                    for i, r in enumerate(current_rows):
                        for a in range(len(r)):
                            for b in range(a + 1, len(r)):
                                used_set.add(r[b] - r[a])

                    found = False
                    min_val = row[elem_pos - 1] + 1 if elem_pos > 0 else 1
                    max_val = old_val - 1

                    for new_val in range(max_val, min_val - 1, -1):
                        valid = True
                        for a in row[:elem_pos]:
                            d = new_val - a
                            if d in used_set:
                                valid = False
                                break
                        if not valid:
                            continue
                        for a in row[elem_pos:]:
                            d = a - new_val
                            if d in used_set:
                                valid = False
                                break
                        if valid:
                            row.insert(elem_pos, new_val)
                            found = True
                            break

                    if found:
                        new_scope = get_scope(current_rows)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved_any = True
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [list(r) for r in current_rows]
                    else:
                        row.insert(elem_pos, old_val)

    # Phase 4: If no MIS solution, fallback to sequential greedy
    if best_rows is None:
        for i in range(1000):
            if time.time() > deadline - 5:
                break
            rng = random.Random(i * 13 + 5)
            used = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    last = row[-1]
                    found = False
                    for x in range(last + 1, lb + 300):
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    if not found:
                        success = False
                        break
                if not success:
                    break
                rows.append(row)
            if success:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows

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
