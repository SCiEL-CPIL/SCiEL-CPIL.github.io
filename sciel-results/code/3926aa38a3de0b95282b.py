"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
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

    # ---------- Gap-based representation and operations ----------
    def row_to_gaps(row):
        """Convert row [0, a1, a2, ...] to gaps [a1-0, a2-a1, ...]."""
        return [row[i+1] - row[i] for i in range(len(row)-1)]

    def gaps_to_row(gaps):
        """Convert gaps to row starting at 0."""
        row = [0]
        s = 0
        for g in gaps:
            s += g
            row.append(s)
        return row

    def all_diffs_from_gaps(gaps):
        """All positive differences from a gap sequence (optimized)."""
        row = [0]
        s = 0
        for g in gaps:
            s += g
            row.append(s)
        diffs = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i+1, n):
                diffs.add(row[j] - ri)
        return diffs

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

    # ---------- Gap-swapping stochastic local search ----------
    def gap_swap_search(rows, time_budget):
        """
        Represent each row by its gaps. Use swap moves between rows to balance span.
        Objective: minimize max prefix sum (i.e., the row's last element = sum of gaps).
        """
        gap_rows = [row_to_gaps(row) for row in rows]
        current_gaps = [list(g) for g in gap_rows]
        
        current_diffs = [all_diffs_from_gaps(g) for g in current_gaps]
        all_used_diffs = set()
        for d in current_diffs:
            all_used_diffs |= d
        
        current_spans = [sum(g) for g in current_gaps]
        current_scope = max(current_spans)
        
        best_gaps = [list(g) for g in current_gaps]
        best_scope = current_scope
        best_diffs = [set(d) for d in current_diffs]
        
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        end_time = time.time() + time_budget
        
        while time.time() < end_time and T > T_min:
            if time_left() < 5:
                break
            
            r1, r2 = random.sample(range(n_rows), 2)
            p1 = random.randint(0, k-1)
            p2 = random.randint(0, k-1)
            
            old_g1 = current_gaps[r1][p1]
            old_g2 = current_gaps[r2][p2]
            
            current_gaps[r1][p1] = old_g2
            current_gaps[r2][p2] = old_g1
            
            if current_gaps[r1][p1] <= 0 or current_gaps[r2][p2] <= 0:
                current_gaps[r1][p1] = old_g1
                current_gaps[r2][p2] = old_g2
                T *= cooling
                continue
            
            new_diffs_r1 = all_diffs_from_gaps(current_gaps[r1])
            new_diffs_r2 = all_diffs_from_gaps(current_gaps[r2])
            
            other_diffs = set()
            for j in range(n_rows):
                if j != r1 and j != r2:
                    other_diffs |= current_diffs[j]
            
            collision = False
            if new_diffs_r1 & other_diffs or new_diffs_r2 & other_diffs:
                collision = True
            elif new_diffs_r1 & new_diffs_r2:
                collision = True
            
            new_span_r1 = sum(current_gaps[r1])
            new_span_r2 = sum(current_gaps[r2])
            
            other_spans = [sum(current_gaps[j]) for j in range(n_rows) if j != r1 and j != r2]
            new_scope = max([new_span_r1, new_span_r2] + other_spans) if other_spans else max(new_span_r1, new_span_r2)
            
            lam = 10.0
            if collision:
                coll_count = len(new_diffs_r1 & other_diffs) + len(new_diffs_r2 & other_diffs) + len(new_diffs_r1 & new_diffs_r2)
                new_energy = new_scope + lam * coll_count
            else:
                new_energy = new_scope
            
            old_energy = current_scope
            
            accepted = False
            if new_energy < old_energy:
                accepted = True
            else:
                delta_e = new_energy - old_energy
                if random.random() < math.exp(-delta_e / T):
                    accepted = True
            
            if accepted:
                current_diffs[r1] = new_diffs_r1
                current_diffs[r2] = new_diffs_r2
                current_spans[r1] = new_span_r1
                current_spans[r2] = new_span_r2
                current_scope = new_scope
                
                if new_scope < best_scope and not collision:
                    best_scope = new_scope
                    best_gaps = [list(g) for g in current_gaps]
                    best_diffs = [set(d) for d in current_diffs]
            else:
                current_gaps[r1][p1] = old_g1
                current_gaps[r2][p2] = old_g2
            
            T *= cooling
        
        best_rows_out = [gaps_to_row(g) for g in best_gaps]
        return best_rows_out

    # ---------- Per-row local optimization (refinement machinery) ----------
    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def try_reduce_row(rows, row_idx, used_other, max_attempts=100):
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
            jitter = 0.1 if attempt > 10 else 0.0
            while len(row) < k + 1:
                found = False
                # Look ahead a bit
                for trial_x in range(x + 1, x + 50):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        # Apply jitter: sometimes skip a valid candidate to explore
                        if jitter > 0 and random.random() < jitter and trial_x < 200:
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
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        return best_row, best_max

    def high_precision_polish(rows):
        """Final high-precision polish: try to move each element to a smaller position."""
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 3
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            for i in range(n_rows):
                if time_left() < 3:
                    break
                row = rows[i]
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                
                # Try to reduce each element (except index 0 which is fixed at 0)
                for pos in range(1, k + 1):
                    if time_left() < 2:
                        break
                    current_val = row[pos]
                    # Try moving this element to a smaller value
                    # We need to check that all differences remain valid
                    for new_val in range(row[pos-1] + 1, current_val):
                        if new_val == current_val:
                            break
                        # Temporarily replace
                        old_val = row[pos]
                        row[pos] = new_val
                        
                        # Check if the row is still strictly increasing
                        if pos < k and new_val >= row[pos+1]:
                            row[pos] = old_val
                            continue
                        
                        # Check all differences in this row
                        row_diffs = set()
                        valid = True
                        for a in range(len(row)):
                            for b in range(a+1, len(row)):
                                d = row[b] - row[a]
                                if d in row_diffs or d in used_other:
                                    valid = False
                                    break
                                row_diffs.add(d)
                            if not valid:
                                break
                        
                        if valid:
                            improved = True
                            break
                        else:
                            row[pos] = old_val
        return rows

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        # Limit the number of full passes to avoid spending too much time here
        # if we are in an ILS loop
        max_passes = 5 
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=50)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        """Perturb the solution to escape local minima."""
        rows = [list(r) for r in rows]
        # Strategy: Rebuild 1 or 2 random rows with high randomness
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            # Calculate used diffs from other rows
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            # Rebuild the row with high jitter
            row, x = [0], 0
            current_row_diffs = set()
            success = False
            # Try a few times to find a valid row, even if larger
            for attempt in range(20):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                while len(row) < k + 1:
                    found = False
                    # Wider search range for kicks
                    for trial_x in range(x + 1, x + 100):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            # High jitter: skip valid candidates often
                            if random.random() < 0.4 and trial_x < 300:
                                continue
                            row.append(trial_x)
                            current_row_diffs |= new
                            found = True
                            break
                    if not found:
                        valid_row = False
                        break
                    x = row[-1]
                
                if valid_row:
                    # Check if this new row is valid with the rest (it should be by construction)
                    # But we must ensure it doesn't conflict with itself (handled) and others (handled)
                    rows[idx] = row
                    success = True
                    break
            
            if not success:
                # If we failed to rebuild, keep the original row
                # This might make the solution invalid temporarily, but local_polish will fix it
                # Or we revert to a greedy row
                greedy_row, x = [0], 0
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    x += 1
                    new = {x - a for a in greedy_row}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(x)
                        row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # High-precision polish after initial local search
        if time_left() > 15:
            hp_polished = high_precision_polish(best_rows)
            if scope(hp_polished) < best_scope:
                best_rows = hp_polished
                best_scope = scope(best_rows)

    # Gap-swap search phase
    if time_left() > 30:
        gap_budget = min(time_left() - 15, 60)
        if gap_budget > 10:
            gap_result = gap_swap_search(best_rows, gap_budget)
            if is_valid(gap_result):
                gap_scope = scope(gap_result)
                if gap_scope < best_scope:
                    best_rows = gap_result
                    best_scope = gap_scope
                    if time_left() > 10:
                        polished_gap = local_polish(best_rows)
                        if scope(polished_gap) < best_scope:
                            best_rows = polished_gap
                            best_scope = scope(polished_gap)

    # Iterated Local Search
    while time_left() > 20:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # 2. Kick
        if time_left() < 15:
            break
            
        kicked_rows = kick(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        # 4. High-precision polish on kicked rows if improved
        if scope(polished_kicked) < best_scope:
            if time_left() > 10:
                hp_kicked = high_precision_polish(polished_kicked)
                if scope(hp_kicked) < scope(polished_kicked):
                    polished_kicked = hp_kicked
        
        # 5. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)
        else:
            # If the kick didn't lead to improvement, we might stick to the previous best
            # and try a different kick next iteration.
            # This is the standard ILS behavior.
            pass

    # Final gap-swap polish
    if time_left() > 15:
        final_gap_budget = min(time_left() - 5, 20)
        if final_gap_budget > 5:
            final_gap_result = gap_swap_search(best_rows, final_gap_budget)
            if is_valid(final_gap_result):
                final_gap_scope = scope(final_gap_result)
                if final_gap_scope < best_scope:
                    best_rows = final_gap_result
                    best_scope = final_gap_scope
                    # High-precision polish after gap swap
                    if time_left() > 10:
                        hp_final = high_precision_polish(best_rows)
                        if scope(hp_final) < best_scope:
                            best_rows = hp_final
                            best_scope = scope(best_rows)

    # Final polish to ensure we squeeze out any last drops
    if time_left() > 5:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # Final high-precision polish
        if time_left() > 3:
            hp_final2 = high_precision_polish(best_rows)
            if scope(hp_final2) < best_scope:
                best_rows = hp_final2
                best_scope = scope(best_rows)

    # Final validity guarantee.
    if not is_valid(best_rows):
        # Fallback to a fresh greedy (always valid).
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
