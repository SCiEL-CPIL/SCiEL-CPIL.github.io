"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Backtracking with forward checking ----------
    def backtrack_construct(n_rows, k, max_scope=None, timeout=30):
        """
        Try to construct n_rows rows of k+1 integers each, starting at 0,
        with all differences distinct across all rows, and max entry <= max_scope.
        Uses backtracking with forward checking and heuristic ordering.
        """
        if max_scope is None:
            max_scope = best_scope - 1
        
        end_time = time.time() + timeout
        used_diffs = set()
        rows = []
        
        def can_place(row, candidate):
            """Check if candidate can be added to row without conflicts."""
            new_diffs = {candidate - a for a in row}
            if new_diffs & used_diffs:
                return False, set()
            if len(new_diffs) != len(new_diffs):  # duplicate within row
                return False, set()
            return True, new_diffs
        
        def heuristic_order(row, max_scope):
            """Generate candidates in a heuristic order: prefer smaller gaps first."""
            if not row:
                return range(1, max_scope + 1)
            last = row[-1]
            # Prefer candidates that don't create very large gaps
            candidates = []
            for x in range(last + 1, max_scope + 1):
                gap = x - last
                candidates.append((gap, x))
            candidates.sort()
            return [x for _, x in candidates]
        
        def backtrack(row_idx, row, max_scope):
            if time.time() > end_time:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(row) == k + 1:
                rows.append(row)
                return backtrack(row_idx + 1, [0], max_scope)
            
            # Try candidates in heuristic order
            candidates = heuristic_order(row, max_scope)
            for candidate in candidates:
                if time.time() > end_time:
                    return False
                
                ok, new_diffs = can_place(row, candidate)
                if not ok:
                    continue
                
                # Forward checking: ensure we can still complete this row
                remaining_positions = k + 1 - len(row) - 1
                # Simple check: there must be enough room
                if candidate > max_scope:
                    continue
                
                row.append(candidate)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, row, max_scope):
                    return True
                
                row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        if backtrack(0, [0], max_scope):
            return rows
        return None

    # ---------- Try backtracking with decreasing max_scope ----------
    if time_left() > 30:
        for target_scope in range(best_scope - 1, max(50, best_scope - 30), -1):
            if time_left() < 20:
                break
            result = backtrack_construct(n_rows, k, target_scope, timeout=8)
            if result is not None and is_valid(result):
                best_rows = result
                best_scope = target_scope
                break

    # ---------- Local optimization: try to reduce a specific row ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=300):
        """Try to find a row at row_idx with smaller max entry (same k+1 size)."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            # Randomized greedy for this row to escape local minima in row construction
            jitter = 0.15 if attempt > 10 else 0.0
            while len(row) < k + 1:
                found = False
                # Look ahead a bit more for better choices
                search_limit = 300
                for trial_x in range(x + 1, x + search_limit):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if jitter and random.random() < jitter and trial_x - x > 3:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                best_row = list(row)
                best_max = max(row)
        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        while improved and time_left() > 10:
            improved = False
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 10:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=150)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    improved = True
        return rows

    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(best_rows)

    # ---------- Iterated Local Search with kicks ----------
    if time_left() > 30:
        # Order rows by descending scope: index 0 is the binding constraint.
        order = sorted(range(n_rows), key=lambda i: -max(best_rows[i]))
        
        def build_sa_state(initial_rows):
            """Build the conflict tracking state for SA/ILS."""
            sa_rows = [list(initial_rows[i]) for i in order]
            max_diff = max(max(r) for r in sa_rows) + 50
            conflict = [0] * (max_diff + 1)
            for row in sa_rows:
                for d in row_diffs_list(row):
                    if d >= len(conflict):
                        conflict.extend([0] * (d + 1 - len(conflict)))
                    conflict[d] += 1
            cur_conflict_count = sum(1 for c in conflict if c > 1)
            
            row_diffs_cache = [row_diffs_set(r) for r in sa_rows]
            
            return sa_rows, conflict, cur_conflict_count, row_diffs_cache

        def greedy_row_for(used_set, jitter=0.0):
            """Greedy row given a set of already-used differences (from other rows)."""
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used_set) and not (new & row_diffs):
                    if jitter and random.random() < jitter and x < 400:
                        continue
                    row.append(x)
                    row_diffs |= new
            return row

        def local_search_sa(sa_rows, conflict, cur_conflict_count, row_diffs_cache, time_budget):
            """Run SA with fast cooling on the given state. Returns best state found."""
            t = 1.0
            t_min = 0.01
            cooling = 0.998
            cur_conf = cur_conflict_count
            cur_scope = max(sa_rows[0])
            best_sa = [list(r) for r in sa_rows]
            best_conf = cur_conf
            best_scope_val = cur_scope
            best_cache = [set(s) for s in row_diffs_cache]
            
            end_time = time.time() + time_budget
            while time.time() < end_time and t > t_min:
                # Pick a row: bias toward the largest (index 0)
                if random.random() < 0.4:
                    idx = 0
                else:
                    idx = random.randrange(n_rows)

                old_row = sa_rows[idx]
                old_diffs = row_diffs_cache[idx]

                # Remove old row's differences.
                for d in old_diffs:
                    old_c = conflict[d]
                    if old_c == 2:
                        cur_conf -= 1
                    elif old_c == 1:
                        cur_conf += 0  # stays 0, no conflict
                    conflict[d] = old_c - 1
                conf_after_remove = cur_conf

                # Build used set from remaining rows
                used_set = set()
                for j in range(n_rows):
                    if j != idx:
                        used_set.update(row_diffs_cache[j])

                # Propose new row
                r = random.random()
                if r < 0.5:
                    new_row = greedy_row_for(used_set)
                elif r < 0.8:
                    new_row = greedy_row_for(used_set, jitter=0.15)
                else:
                    new_row = greedy_row_for(used_set, jitter=0.3)

                if len(new_row) != k + 1:
                    # Fallback: revert
                    for d in old_diffs:
                        conflict[d] += 1
                        if conflict[d] == 2:
                            cur_conf += 1
                    t *= cooling
                    continue

                new_diffs_set = row_diffs_set(new_row)
                
                # Add new row's differences
                for d in new_diffs_set:
                    if d >= len(conflict):
                        conflict.extend([0] * (d + 1 - len(conflict)))
                    old_c = conflict[d]
                    if old_c == 1:
                        cur_conf += 1
                    conflict[d] = old_c + 1
                conf_after_add = cur_conf

                new_scope = max(new_row) if idx == 0 else max(sa_rows[0])
                conf_delta = conf_after_add - conf_after_remove
                scope_delta = new_scope - cur_scope

                accept = False
                if conf_delta < 0 or (conf_delta == 0 and scope_delta < 0):
                    accept = True
                else:
                    penalty = conf_delta * 1000 + scope_delta
                    if penalty < 0:
                        accept = True
                    else:
                        if random.random() < math.exp(-penalty / max(t, 1e-6)):
                            accept = True

                if accept:
                    sa_rows[idx] = new_row
                    row_diffs_cache[idx] = new_diffs_set
                    cur_conf = conf_after_add
                    cur_scope = new_scope
                    if (cur_conf, cur_scope) < (best_conf, best_scope_val):
                        best_conf = cur_conf
                        best_scope_val = cur_scope
                        best_sa = [list(r) for r in sa_rows]
                        best_cache = [set(s) for s in row_diffs_cache]
                else:
                    # Revert
                    for d in new_diffs_set:
                        conflict[d] -= 1
                        if conflict[d] == 1:
                            cur_conf -= 1
                    for d in old_diffs:
                        if d >= len(conflict):
                            conflict.extend([0] * (d + 1 - len(conflict)))
                        old_c = conflict[d]
                        if old_c == 1:
                            cur_conf += 1
                        conflict[d] = old_c + 1
                    cur_conf = conf_after_remove

                t *= cooling

            return best_sa, best_conf, best_scope_val, best_cache

        # ILS loop
        sa_rows, conflict, cur_conflict_count, row_diffs_cache = build_sa_state(best_rows)
        
        # Initial local search
        if time_left() > 30:
            best_sa, best_conf, best_scope_val, best_cache = local_search_sa(
                sa_rows, conflict, cur_conflict_count, row_diffs_cache,
                min(20, time_left() - 20)
            )
            if best_conf == 0:
                restored = [None] * n_rows
                for i in range(n_rows):
                    restored[order[i]] = best_sa[i]
                if is_valid(restored):
                    s = scope(restored)
                    if s < best_scope:
                        best_rows = restored
                        best_scope = s

        # ILS: kick + local search
        while time_left() > 30:
            # Kick: perturb the current best state
            if best_conf == 0:
                # We have a valid solution; kick by replacing one row
                sa_rows = [list(r) for r in best_sa]
                conflict = [0] * (max(max(r) for r in sa_rows) + 50)
                for row in sa_rows:
                    for d in row_diffs_list(row):
                        if d >= len(conflict):
                            conflict.extend([0] * (d + 1 - len(conflict)))
                        conflict[d] += 1
                cur_conflict_count = sum(1 for c in conflict if c > 1)
                row_diffs_cache = [row_diffs_set(r) for r in sa_rows]
            else:
                # Start from a perturbed version of the current state
                sa_rows = [list(r) for r in best_sa]
                # Replace 1-2 random rows with new greedy rows
                for _ in range(random.randint(1, 2)):
                    idx = random.randrange(n_rows)
                    used = set()
                    for j in range(n_rows):
                        if j != idx:
                            used.update(row_diffs_cache[j])
                    sa_rows[idx] = greedy_row_for(used, jitter=0.2)
                    row_diffs_cache[idx] = row_diffs_set(sa_rows[idx])
                
                conflict = [0] * (max(max(r) for r in sa_rows) + 50)
                for row in sa_rows:
                    for d in row_diffs_list(row):
                        if d >= len(conflict):
                            conflict.extend([0] * (d + 1 - len(conflict)))
                        conflict[d] += 1
                cur_conflict_count = sum(1 for c in conflict if c > 1)

            # Local search on the kicked state
            if time_left() > 25:
                budget = min(15, time_left() - 20)
                new_best_sa, new_best_conf, new_best_scope_val, new_best_cache = local_search_sa(
                    sa_rows, conflict, cur_conflict_count, row_diffs_cache, budget
                )
                if new_best_conf < best_conf or (new_best_conf == best_conf and new_best_scope_val < best_scope_val):
                    best_sa = new_best_sa
                    best_conf = new_best_conf
                    best_scope_val = new_best_scope_val
                    best_cache = new_best_cache
                    if best_conf == 0:
                        restored = [None] * n_rows
                        for i in range(n_rows):
                            restored[order[i]] = best_sa[i]
                        if is_valid(restored):
                            s = scope(restored)
                            if s < best_scope:
                                best_rows = restored
                                best_scope = s

        # Final local polish
        if time_left() > 10:
            polished = local_polish(best_rows)
            if scope(polished) < best_scope:
                best_rows = polished
                best_scope = scope(polished)

    # Final validity guarantee.
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
