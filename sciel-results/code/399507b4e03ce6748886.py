"""Difference triangle set (4,7): minimize largest entry.
Combined: structural ILS from inspiration + SA refinement from current."""
import time
import random
import math

_DEADLINE = 145.0  # 5s safety margin under 150s kill
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def construct(n_rows, k):
    global _START
    _START = time.time()
    deadline = _START + 145.0

    def time_left():
        return deadline - time.time()

    # ---------- Fast difference utilities ----------
    def row_diff_list(row):
        """Return list of all positive differences in a row."""
        diffs = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diff_set(row):
        """Return set of all positive differences in a row."""
        n = len(row)
        return {row[j] - row[i] for i in range(n) for j in range(i + 1, n)}

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diff_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            row_diffs = set()
            x = 0
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
    if not is_valid(best_rows):
        # Shouldn't happen with greedy, but safety
        best_rows = greedy()
        best_scope = scope(best_rows)

    # ---------- Fast conflict tracking via count array ----------
    def build_conflict_array(rows, max_val):
        """Build conflict count array. Returns (conflict_arr, total_conflicts)."""
        # Determine max possible difference
        max_diff = max(max(r) for r in rows) + 1
        conflict = [0] * (max_diff + 1)
        for row in rows:
            for d in row_diff_list(row):
                if d >= len(conflict):
                    conflict.extend([0] * (d + 1 - len(conflict)))
                conflict[d] += 1
        total = sum(1 for c in conflict if c > 1)
        return conflict, total

    def remove_row_from_conflict(row, conflict):
        """Remove a row's differences from conflict array. Returns delta in conflict count."""
        delta = 0
        for d in row_diff_list(row):
            old = conflict[d]
            new = old - 1
            if old == 2 and new == 1:
                delta -= 1
            elif old == 1 and new == 0:
                delta -= 1  # was already non-conflict, stays non-conflict
            conflict[d] = new
        return delta

    def add_row_to_conflict(row, conflict):
        """Add a row's differences to conflict array. Returns delta in conflict count."""
        delta = 0
        for d in row_diff_list(row):
            if d >= len(conflict):
                conflict.extend([0] * (d + 1 - len(conflict)))
            old = conflict[d]
            new = old + 1
            if old == 1 and new == 2:
                delta += 1
            elif old == 0 and new == 1:
                delta += 0
            conflict[d] = new
        return delta

    # ---------- Per-row greedy rebuild ----------
    def greedy_row_avoiding(used_set, k, max_search=500):
        """Build one row of k+1 entries avoiding used_set differences. Returns row or None."""
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < k + 1:
            found = False
            for trial_x in range(x + 1, x + max_search):
                new = {trial_x - a for a in row}
                if not (new & used_set) and not (new & row_diffs):
                    row.append(trial_x)
                    row_diffs |= new
                    x = trial_x
                    found = True
                    break
            if not found:
                return None
        return row

    def greedy_row_with_jitter(used_set, k, jitter=0.03, max_search=500):
        """Greedy row with occasional random skips for diversity."""
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < k + 1:
            found = False
            for trial_x in range(x + 1, x + max_search):
                new = {trial_x - a for a in row}
                if not (new & used_set) and not (new & row_diffs):
                    if jitter > 0 and random.random() < jitter:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    x = trial_x
                    found = True
                    break
            if not found:
                return None
        return row

    # ---------- Local polish: re-optimize each row ----------
    def local_polish(rows, max_time=20.0):
        """Reduce scope by re-optimizing each row one at a time."""
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        while improved and (time.time() - start) < max_time and time_left() > 10:
            improved = False
            # Process rows in order of descending max entry
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for idx in order:
                if time_left() < 10:
                    break
                # Collect all differences from other rows
                used_other = set()
                for j in range(n_rows):
                    if j != idx:
                        used_other |= row_diff_set(rows[j])
                
                current_row = rows[idx]
                current_max = max(current_row)
                best_new_row = current_row
                best_new_max = current_max
                
                # Try multiple greedy rebuilds with different jitters
                for attempt in range(15):
                    if time_left() < 10:
                        break
                    jitter = 0.0 if attempt == 0 else 0.01 + 0.005 * attempt
                    new_row = greedy_row_with_jitter(used_other, k, jitter=jitter, max_search=400)
                    if new_row is not None:
                        new_max = max(new_row)
                        if new_max < best_new_max:
                            # Verify full validity
                            test_rows = [list(r) for r in rows]
                            test_rows[idx] = new_row
                            if is_valid(test_rows):
                                best_new_row = new_row
                                best_new_max = new_max
                
                if best_new_max < max(rows[idx]):
                    rows[idx] = best_new_row
                    improved = True
        
        return rows

    # Apply initial local polish
    if time_left() > 30:
        polished = local_polish(best_rows, max_time=min(20.0, time_left() - 25.0))
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # ---------- Simulated Annealing with fast conflict tracking ----------
    def sa_optimize(rows, time_limit=60.0):
        """Simulated annealing to reduce conflicts and scope."""
        # Order rows by descending scope
        order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
        sa_rows = [list(rows[i]) for i in order]
        
        max_val = max(max(r) for r in sa_rows) + 10
        conflict, cur_conflicts = build_conflict_array(sa_rows, max_val)
        
        cur_scope = max(sa_rows[0])  # binding constraint is first row
        best_sa = [list(r) for r in sa_rows]
        best_conflicts = cur_conflicts
        best_scope = cur_scope
        
        start_time = time.time()
        start_temp = 30.0
        end_temp = 0.5
        
        while time.time() - start_time < time_limit and time_left() > 8:
            elapsed = time.time() - start_time
            temp = start_temp * (end_temp / start_temp) ** (elapsed / time_limit)
            
            # Pick random row and position
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # positions 1..k (not 0)
            
            row = sa_rows[row_idx]
            old_val = row[pos]
            
            # Generate new value
            delta = random.randint(-8, 8)
            new_val = old_val + delta
            
            # Clamp to maintain increasing order
            if new_val <= row[pos - 1]:
                new_val = row[pos - 1] + 1
            if pos < k and new_val >= row[pos + 1]:
                new_val = row[pos + 1] - 1
            if new_val <= row[pos - 1] or (pos < k and new_val >= row[pos + 1]):
                continue
            
            # Remove old differences involving this position
            # Differences: row[pos] - row[i] for i < pos, and row[j] - row[pos] for j > pos
            old_diffs = []
            for i in range(pos):
                old_diffs.append(row[pos] - row[i])
            for j in range(pos + 1, len(row)):
                old_diffs.append(row[j] - row[pos])
            
            # Remove from conflict array
            conf_delta = 0
            for d in old_diffs:
                old_c = conflict[d]
                new_c = old_c - 1
                if old_c == 2 and new_c == 1:
                    conf_delta -= 1
                elif old_c == 1 and new_c == 0:
                    conf_delta -= 1
                conflict[d] = new_c
            
            # Apply new value
            row[pos] = new_val
            
            # Add new differences
            new_diffs = []
            for i in range(pos):
                new_diffs.append(new_val - row[i])
            for j in range(pos + 1, len(row)):
                new_diffs.append(row[j] - new_val)
            
            for d in new_diffs:
                if d >= len(conflict):
                    conflict.extend([0] * (d + 1 - len(conflict)))
                old_c = conflict[d]
                new_c = old_c + 1
                if old_c == 1 and new_c == 2:
                    conf_delta += 1
                elif old_c == 0 and new_c == 1:
                    conf_delta += 0
                conflict[d] = new_c
            
            new_conflicts = cur_conflicts + conf_delta
            new_scope = max(sa_rows[0]) if row_idx == 0 else cur_scope
            
            # Energy: primary = conflicts, secondary = scope
            old_energy = (cur_conflicts, cur_scope)
            new_energy = (new_conflicts, new_scope)
            
            if new_energy < old_energy:
                # Accept
                cur_conflicts = new_conflicts
                cur_scope = new_scope
                if new_conflicts == 0 and new_scope < best_scope:
                    best_sa = [list(r) for r in sa_rows]
                    best_conflicts = 0
                    best_scope = new_scope
                elif new_conflicts < best_conflicts:
                    best_sa = [list(r) for r in sa_rows]
                    best_conflicts = new_conflicts
                    best_scope = new_scope
            else:
                # Metropolis acceptance
                delta_e = (new_conflicts - cur_conflicts) + 0.05 * (new_scope - cur_scope)
                if delta_e <= 0 or random.random() < math.exp(-delta_e / max(temp, 0.01)):
                    cur_conflicts = new_conflicts
                    cur_scope = new_scope
                else:
                    # Revert
                    row[pos] = old_val
                    # Re-add old diffs, remove new diffs
                    for d in new_diffs:
                        old_c = conflict[d]
                        new_c = old_c - 1
                        if old_c == 2 and new_c == 1:
                            conf_delta -= 1
                        elif old_c == 1 and new_c == 0:
                            conf_delta -= 1
                        conflict[d] = new_c
                    for d in old_diffs:
                        old_c = conflict[d]
                        new_c = old_c + 1
                        if old_c == 1 and new_c == 2:
                            conf_delta += 1
                        elif old_c == 0 and new_c == 1:
                            conf_delta += 0
                        conflict[d] = new_c
                    cur_conflicts = cur_conflicts  # restored
        
        # Convert back to original ordering
        result = [None] * n_rows
        for new_idx, orig_idx in enumerate(order):
            result[orig_idx] = sa_rows[new_idx]
        
        if best_conflicts == 0:
            # Convert best_sa back
            result_best = [None] * n_rows
            for new_idx, orig_idx in enumerate(order):
                result_best[orig_idx] = best_sa[new_idx]
            return result_best
        return result

    # ---------- Iterated Local Search ----------
    def ils_search(rows, time_limit=80.0):
        """Iterated local search: kick + local polish."""
        current = [list(r) for r in rows]
        current_scope = scope(current)
        best = [list(r) for r in current]
        best_scope = current_scope
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit and time_left() > 15:
            # Kick: random perturbation
            kicked = [list(r) for r in current]
            # Perturb a random row
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            old_val = kicked[row_idx][pos]
            delta = random.randint(-15, 15)
            new_val = old_val + delta
            if new_val > kicked[row_idx][pos - 1] and (pos == k or new_val < kicked[row_idx][pos + 1]):
                kicked[row_idx][pos] = new_val
            
            # Local polish the kicked solution
            polished = local_polish(kicked, max_time=min(15.0, time_left() - 15.0))
            pol_scope = scope(polished)
            
            if pol_scope < best_scope:
                best = polished
                best_scope = pol_scope
                current = [list(r) for r in polished]
            elif random.random() < 0.3:
                # Occasionally accept worse to escape local optima
                current = [list(r) for r in polished]
        
        return best

    # Apply SA if we have time
    if time_left() > 40:
        sa_time = min(time_left() - 35.0, 50.0)
        if sa_time > 10:
            sa_result = sa_optimize(best_rows, time_limit=sa_time)
            if is_valid(sa_result):
                sa_scope = scope(sa_result)
                if sa_scope < best_scope:
                    best_rows = sa_result
                    best_scope = sa_scope
            # Polish the SA result
            if time_left() > 20:
                polished = local_polish(best_rows, max_time=min(10.0, time_left() - 18.0))
                if is_valid(polished) and scope(polished) < best_scope:
                    best_rows = polished
                    best_scope = scope(polished)

    # Apply ILS if we have time
    if time_left() > 30:
        ils_time = min(time_left() - 25.0, 60.0)
        if ils_time > 15:
            ils_result = ils_search(best_rows, time_limit=ils_time)
            if is_valid(ils_result):
                ils_scope = scope(ils_result)
                if ils_scope < best_scope:
                    best_rows = ils_result
                    best_scope = ils_scope

    # Final polish
    if time_left() > 15:
        final = local_polish(best_rows, max_time=min(8.0, time_left() - 12.0))
        if is_valid(final) and scope(final) < best_scope:
            best_rows = final
            best_scope = scope(final)

    # Ensure validity
    if not is_valid(best_rows):
        best_rows = greedy()
    
    return best_rows
