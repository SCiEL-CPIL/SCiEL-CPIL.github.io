import time
import random
import math
import numpy as np

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 115.0  # Safety margin for 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            if len(row) != row_len:
                return None
            if row[0] != 0:
                return None
            for i in range(len(row)):
                ri = row[i]
                if i > 0 and ri <= row[i - 1]:
                    return None
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def compute_all_diffs(rows):
        all_diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    all_diffs.add(r[j] - r[i])
        return all_diffs

    def get_remove_set(rows, r_idx, c_idx):
        row = rows[r_idx]
        remove_set = set()
        curr_val = row[c_idx]
        for i in range(c_idx):
            remove_set.add(curr_val - row[i])
        for j in range(c_idx + 1, len(row)):
            remove_set.add(row[j] - curr_val)
        return remove_set

    def check_move_valid(rows, r_idx, c_idx, new_val, fixed_diffs):
        row = rows[r_idx]
        if c_idx > 0 and new_val <= row[c_idx - 1]:
            return False
        if c_idx < len(row) - 1 and new_val >= row[c_idx + 1]:
            return False
        new_left_diffs = [new_val - row[i] for i in range(c_idx)]
        new_right_diffs = [row[j] - new_val for j in range(c_idx + 1, len(row))]
        all_new = new_left_diffs + new_right_diffs
        if len(set(all_new)) != len(all_new):
            return False
        for d in all_new:
            if d in fixed_diffs:
                return False
        return True

    def targeted_descent(rows, budget):
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = False
        all_diffs = compute_all_diffs(cur)

        while time.time() - start < budget:
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                continue

            candidates = set()
            for v in range(lower, min(lower + 20, old_val)):
                candidates.add(v)
            for _ in range(15):
                candidates.add(rng.randint(lower, old_val - 1))

            remove_set = get_remove_set(cur, r, pos)
            fixed_diffs = all_diffs - remove_set

            for new_val in sorted(candidates, key=lambda x: -x):
                if new_val >= old_val:
                    continue
                if check_move_valid(cur, r, pos, new_val, fixed_diffs):
                    row[pos] = new_val
                    all_diffs -= remove_set
                    for i in range(pos):
                        all_diffs.add(new_val - row[i])
                    new_scope = get_scope(cur)
                    if new_scope < cur_scope:
                        cur_scope = new_scope
                        improved = True
                        break
                    else:
                        row[pos] = old_val
                        all_diffs -= {new_val - row[i] for i in range(pos)}
                        all_diffs |= remove_set

            if not improved:
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    break
                remove_set2 = get_remove_set(cur, r, pos2)
                fixed_diffs2 = all_diffs - remove_set2
                for _ in range(8):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    if check_move_valid(cur, r, pos2, nv, fixed_diffs2):
                        row[pos2] = nv
                        all_diffs -= remove_set2
                        for i in range(pos2):
                            all_diffs.add(nv - row[i])
                        for j in range(pos2 + 1, len(row)):
                            all_diffs.add(row[j] - nv)
                        new_scope = get_scope(cur)
                        if new_scope < cur_scope:
                            cur_scope = new_scope
                            improved = True
                            break
                        else:
                            row[pos2] = old2
                            all_diffs -= {nv - row[i] for i in range(pos2)}
                            all_diffs -= {row[j] - nv for j in range(pos2 + 1, len(row))}
                            all_diffs |= remove_set2
                if not improved:
                    break

        if improved:
            return cur, cur_scope
        return None, None

    def row_rebuild_ilc(cur, cur_scope, all_diffs, budget):
        """
        Iterated Local Construction: rebuild one entire row at a time using
        a constraint-aware greedy with randomization, then run targeted descent.
        This provides much larger moves than single-element SA.
        """
        start = time.time()
        best = [r[:] for r in cur]
        best_scope = cur_scope
        best_diffs = all_diffs.copy()

        while time.time() - start < budget:
            # Pick a row to rebuild
            r_idx = rng.randrange(n_rows)
            
            # Remove this row's diffs from the global set
            old_row = best[r_idx]
            row_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    row_diffs.add(old_row[j] - old_row[i])
            
            fixed_diffs = best_diffs - row_diffs
            
            # Rebuild the row with constraint-aware greedy + randomization
            new_row = [0]
            used_in_row = set()
            max_target = best_scope + rng.randint(0, 10)  # allow slight expansion
            
            for pos in range(1, row_len):
                # Determine valid range
                lo = new_row[-1] + 1
                hi = min(max_target, best_scope + 20)
                
                if hi <= lo:
                    # Try to expand scope slightly
                    hi = lo + rng.randint(1, 5)
                
                # Try candidates in random order, prefer smaller values
                candidates = list(range(lo, hi + 1))
                rng.shuffle(candidates)
                
                # Sort by a heuristic: prefer values that create less "pressure"
                # i.e., values whose diffs are less likely to conflict
                def conflict_score(v):
                    new_diffs = {v - a for a in new_row}
                    return len(new_diffs & fixed_diffs)
                
                # Partially sort: keep some randomness but bias toward low conflict
                candidates.sort(key=lambda v: (conflict_score(v), rng.random()))
                
                placed = False
                for v in candidates[:min(50, len(candidates))]:
                    new_diffs = {v - a for a in new_row}
                    if new_diffs & fixed_diffs:
                        continue
                    if len(new_diffs) != len(new_row):
                        continue
                    new_row.append(v)
                    used_in_row |= new_diffs
                    placed = True
                    break
                
                if not placed:
                    # Backtrack: try a wider search
                    for v in range(lo, hi + 1):
                        new_diffs = {v - a for a in new_row}
                        if new_diffs & fixed_diffs:
                            continue
                        if len(new_diffs) != len(new_row):
                            continue
                        new_row.append(v)
                        used_in_row |= new_diffs
                        placed = True
                        break
                
                if not placed:
                    # Can't place; revert and try another row
                    break
            
            if len(new_row) < row_len:
                continue
            
            # Validate the new row against fixed_diffs
            new_row_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_row_diffs.add(new_row[j] - new_row[i])
            
            if new_row_diffs & fixed_diffs:
                continue
            
            # Apply the new row
            best[r_idx] = new_row
            new_diffs_all = best_diffs - row_diffs | new_row_diffs
            new_scope = max(max(row) for row in best)
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_diffs = new_diffs_all
            
            # Run a short targeted descent
            desc_rows, desc_scope = targeted_descent(best, budget=0.5)
            if desc_rows is not None and desc_scope < best_scope:
                best = desc_rows
                best_scope = desc_scope
                best_diffs = compute_all_diffs(best)

        return best, best_scope

    # Main search
    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)
    rng = random.Random(42)

    # Phase 1: Simulated annealing with restarts
    max_restarts = 200
    steps_per_restart = 60000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 10
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)

        cur_scope = full_validate_scope(cur)
        if cur_scope is None:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        T = max(0.05, 1.0 - restart / max_restarts)
        all_diffs = compute_all_diffs(cur)

        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 20
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            remove_set = get_remove_set(cur, ri, pos)
            fixed_diffs = all_diffs - remove_set

            if check_move_valid(cur, ri, pos, new_val, fixed_diffs):
                cur[ri][pos] = new_val
                all_diffs -= remove_set
                for i in range(pos):
                    all_diffs.add(new_val - cur[ri][i])
                for j in range(pos + 1, len(cur[ri])):
                    all_diffs.add(cur[ri][j] - new_val)

                new_scope = get_scope(cur)

                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]

                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_scope = new_scope
                else:
                    cur[ri][pos] = old_val
                    all_diffs -= {new_val - cur[ri][i] for i in range(pos)}
                    all_diffs -= {cur[ri][j] - new_val for j in range(pos + 1, len(cur[ri]))}
                    all_diffs |= remove_set
            else:
                pass

            T *= 0.99995

        # Phase 2: ILC row rebuilds
        if time.time() < deadline - 10.0:
            ilc_rows, ilc_scope = row_rebuild_ilc(cur, cur_scope, all_diffs, budget=min(5.0, deadline - time.time() - 5.0))
            if ilc_scope < best_scope:
                best_scope = ilc_scope
                best_rows = ilc_rows

        # Final targeted descent
        if time.time() < deadline - 5.0:
            desc_rows, desc_scope = targeted_descent(best_rows, budget=min(2.0, deadline - time.time() - 2.0))
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows

    return best_rows
