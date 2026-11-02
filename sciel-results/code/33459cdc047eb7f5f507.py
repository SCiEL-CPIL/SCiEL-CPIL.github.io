import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    L = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1  # counting lower bound

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def build_used(rows):
        used = set()
        for row in rows:
            for i in range(L):
                ri = row[i]
                for j in range(i + 1, L):
                    used.add(row[j] - ri)
        return used

    def diffs_for_pos(row, pos):
        v = row[pos]
        s = set()
        for i in range(L):
            if i < pos:
                s.add(v - row[i])
            elif i > pos:
                s.add(row[i] - v)
        return s

    def can_place(row, pos, new_val, used_set):
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < L - 1 and new_val >= row[pos + 1]:
            return False
        old = diffs_for_pos(row, pos)
        other = used_set - old
        v = new_val
        for i in range(L):
            if i < pos:
                d = v - row[i]
                if d in other:
                    return False
            elif i > pos:
                d = row[i] - v
                if d in other:
                    return False
        return True

    def apply_set(row, pos, new_val, used_set):
        old = diffs_for_pos(row, pos)
        for d in old:
            used_set.discard(d)
        v = new_val
        for i in range(L):
            if i < pos:
                used_set.add(v - row[i])
            elif i > pos:
                used_set.add(row[i] - v)
        row[pos] = new_val

    def greedy_build(seed):
        """Build a complete valid assignment via randomized greedy with forward checking."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < L:
                last = row[-1]
                # Adaptive search window: wider when used set is large
                window = 20 + int(0.15 * len(used)) + rng.randint(0, 15)
                found = False
                # Try candidates in random order within window
                start = last + 1
                end = min(start + window, lb + 300)
                candidates = list(range(start, end))
                rng.shuffle(candidates)
                for x in candidates:
                    nd = [x - a for a in row]
                    if any(d in used for d in nd):
                        continue
                    row.append(x)
                    used.update(nd)
                    found = True
                    break
                if not found:
                    # Expand search
                    for x in range(start, lb + 500):
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

    def rebuild_row(rows, r_idx, used_set, rng, target_scope=None):
        """
        Rebuild row r_idx. 
        Returns (success, old_row).
        If success is False, old_row is already restored in rows[r_idx] and used_set.
        If success is True, rows[r_idx] is the new row, and old_row is returned for potential undo.
        """
        old_row = list(rows[r_idx])
        
        # 1. Remove old diffs
        for i in range(L):
            for j in range(i + 1, L):
                used_set.discard(old_row[j] - old_row[i])
        
        # 2. Greedily construct new row
        new_row = [0]
        limit = target_scope if target_scope else max(max(r) for r in rows) + 10
        
        for pos in range(1, L):
            last = new_row[-1]
            found = False
            max_val = min(limit, last + 100)
            
            for x in range(last + 1, max_val + 1):
                conflict = False
                for i in range(len(new_row)):
                    d = x - new_row[i]
                    if d in used_set:
                        conflict = True
                        break
                if not conflict:
                    new_row.append(x)
                    used_set.update([x - a for a in new_row[:-1]])
                    found = True
                    break
            
            if not found:
                for x in range(last + 1, limit + 1):
                    conflict = False
                    for i in range(len(new_row)):
                        d = x - new_row[i]
                        if d in used_set:
                            conflict = True
                            break
                    if not conflict:
                        new_row.append(x)
                        used_set.update([x - a for a in new_row[:-1]])
                        found = True
                        break
                
                if not found:
                    # Restore old row and return False
                    for i in range(L):
                        for j in range(i + 1, L):
                            used_set.add(old_row[j] - old_row[i])
                    rows[r_idx] = old_row
                    return False, old_row
        
        rows[r_idx] = new_row
        return True, old_row

    def simulated_annealing(rows_init, time_limit):
        """SA to minimize scope by moving individual elements or rebuilding rows."""
        rows = [list(r) for r in rows_init]
        used = build_used(rows)
        current_scope = get_scope(rows)
        best_local = current_scope
        best_rows_local = [list(r) for r in rows]
        
        T = 50.0  # initial temperature
        T_min = 0.1
        cooling = 0.997
        rng = random.Random(int(time.time() * 1000))
        
        while time.time() < time_limit and T > T_min:
            # With 20% probability, rebuild a random row
            if rng.random() < 0.2:
                r_idx = rng.randint(0, n_rows - 1)
                allowed_scope = best_local + 5 
                success, old_row = rebuild_row(rows, r_idx, used, rng, allowed_scope)
                if success:
                    new_scope = get_scope(rows)
                    delta = new_scope - current_scope
                    if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
                        current_scope = new_scope
                        if current_scope < best_local:
                            best_local = current_scope
                            best_rows_local = [list(r) for r in rows]
                    else:
                        # Reject: undo rebuild
                        # Remove new diffs
                        new_row = rows[r_idx]
                        for i in range(L):
                            for j in range(i + 1, L):
                                used.discard(new_row[j] - new_row[i])
                        # Restore old row and diffs
                        rows[r_idx] = old_row
                        for i in range(L):
                            for j in range(i + 1, L):
                                used.add(old_row[j] - old_row[i])
            
            else:
                # Original single element move logic
                r_idx = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, L - 1)  # skip pos 0 (fixed at 0)
                row = rows[r_idx]
                old_val = row[pos]
                
                # Determine move range
                lower = row[pos - 1] + 1
                upper = row[pos + 1] - 1 if pos < L - 1 else old_val + 30
                
                if lower >= upper:
                    T *= cooling
                    continue
                
                # Choose new value: biased toward reducing scope
                if pos == L - 1 and rng.random() < 0.6:
                    # Bias downward for last element
                    new_val = rng.randint(lower, min(old_val, upper))
                else:
                    new_val = rng.randint(lower, max(lower, upper))
                
                if new_val == old_val:
                    continue
                
                if not can_place(row, pos, new_val, used):
                    continue
                
                # Apply move
                apply_set(row, pos, new_val, used)
                new_scope = get_scope(rows)
                
                # Acceptance criterion
                delta = new_scope - current_scope
                if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
                    current_scope = new_scope
                    if current_scope < best_local:
                        best_local = current_scope
                        best_rows_local = [list(r) for r in rows]
                else:
                    # Reject: undo
                    apply_set(row, pos, old_val, used)
            
            T *= cooling
        
        return best_rows_local, best_local

    def hill_climb_compact(rows, time_limit):
        """Aggressive local compaction: repeatedly try to reduce each element."""
        work = [list(r) for r in rows]
        used_set = build_used(work)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Randomize order of rows/positions to avoid cycles
            order = [(r, p) for r in range(n_rows) for p in range(L - 1, 0, -1)]
            random.shuffle(order)
            for r_idx, pos in order:
                if time.time() >= time_limit:
                    break
                row = work[r_idx]
                old_val = row[pos]
                lower = row[pos - 1] + 1
                upper = row[pos + 1] - 1 if pos < L - 1 else old_val
                if lower >= old_val:
                    continue
                for new_val in range(lower, min(old_val, upper + 1)):
                    if can_place(row, pos, new_val, used_set):
                        apply_set(row, pos, new_val, used_set)
                        improved = True
                        break
        return work

    # Phase 1: Multiple greedy constructions
    num_greedy = 2000
    for i in range(num_greedy):
        if time.time() > deadline - 10:
            break
        rows = greedy_build(i * 131 + 17)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope <= lb + 5:
                    break

    if best_rows is None:
        # Fallback: try harder greedy with larger windows
        for i in range(500):
            if time.time() > deadline - 5:
                break
            rows = greedy_build(90000 + i * 73)
            if rows is not None:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]

    if best_rows is None:
        return None

    # Phase 2: Simulated annealing on best solution
    if time.time() < deadline - 8:
        sa_rows, sa_scope = simulated_annealing(best_rows, deadline - 8)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows

    # Phase 3: Hill climb compaction
    if time.time() < deadline - 4:
        compacted = hill_climb_compact(best_rows, deadline - 4)
        c_scope = get_scope(compacted)
        if c_scope < best_scope:
            best_scope = c_scope
            best_rows = compacted

    # Phase 4: Another SA round from compacted solution
    if time.time() < deadline - 3:
        sa2_rows, sa2_scope = simulated_annealing(best_rows, deadline - 3)
        if sa2_scope < best_scope:
            best_scope = sa2_scope
            best_rows = sa2_rows

    # Phase 5: Final compaction
    if time.time() < deadline - 1:
        final_compact = hill_climb_compact(best_rows, deadline - 1)
        f_scope = get_scope(final_compact)
        if f_scope < best_scope:
            best_scope = f_scope
            best_rows = final_compact

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
