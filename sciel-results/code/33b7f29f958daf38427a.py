"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math
import sys

_TIME_LIMIT = 120.0
_DEADLINE = None
_DEADLINE_SET = False


def _time_left():
    global _DEADLINE, _DEADLINE_SET
    if not _DEADLINE_SET:
        _DEADLINE = time.time() + _TIME_LIMIT
        _DEADLINE_SET = True
    return max(0.0, _DEADLINE - time.time())


def _scope(rows):
    if not rows:
        return 0
    m = 0
    for r in rows:
        if r:
            m = max(m, r[-1])
    return m


def _validate(rows):
    """Check that all positive differences within each row are distinct across all rows."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _row_diffs(row):
    """Compute set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _all_diffs_except(rows, skip_idx):
    """Compute set of all differences except those in rows[skip_idx]."""
    s = set()
    for idx, row in enumerate(rows):
        if idx == skip_idx:
            continue
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
    return s


def _greedy_row(used, k, rng=None, strategy="pure"):
    """
    Build one row greedily.
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        if strategy == "pure":
            x += 1
            while True:
                new = {x - a for a in row}
                if not new & used:
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        elif strategy == "shuffled":
            base = x + 1
            window = 50
            candidates = list(range(base, base + window))
            if rng:
                rng.shuffle(candidates)
            found = False
            for cand in candidates:
                new = {cand - a for a in row}
                if not new & used:
                    row.append(cand)
                    used |= {cand - a for a in row[:-1]}
                    x = cand
                    found = True
                    break
            if not found:
                cand = base + window
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= {cand - a for a in row[:-1]}
                        x = cand
                        break
                    cand += 1
        elif strategy == "lookahead":
            base = x + 1
            valid_cands = []
            for cand in range(base, base + 50):
                new = {cand - a for a in row}
                if not new & used:
                    valid_cands.append(cand)
                if len(valid_cands) >= 8:
                    break
            if not valid_cands:
                cand = base
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        valid_cands.append(cand)
                        break
                    cand += 1
            best_cand = valid_cands[0]
            best_score = -1
            for cand in valid_cands[:6]:
                new = {cand - a for a in row}
                next_count = 0
                for nc in range(cand + 1, cand + 25):
                    nd = {nc - a for a in row}
                    nd.add(nc - cand)
                    if not (nd & (used | new)):
                        next_count += 1
                    if next_count >= 4:
                        break
                score = next_count * 100 - cand
                if score > best_score:
                    best_score = score
                    best_cand = cand
            row.append(best_cand)
            used |= {best_cand - a for a in row[:-1]}
            x = best_cand
        elif strategy == "pressure":
            base = x + 1
            valid_cands = []
            for cand in range(base, base + 60):
                new = {cand - a for a in row}
                if not new & used:
                    valid_cands.append((cand, max(new)))
                if len(valid_cands) >= 10:
                    break
            if not valid_cands:
                cand = base
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        valid_cands.append((cand, max(new)))
                        break
                    cand += 1
            valid_cands.sort(key=lambda t: (t[1], t[0]))
            best_cand = valid_cands[0][0]
            row.append(best_cand)
            used |= {best_cand - a for a in row[:-1]}
            x = best_cand
    return row


def _construct_diverse(n_rows, k, rng, strategy):
    """Construct full set using a given strategy."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, strategy)
        rows.append(row)
    return rows


def _backtrack_row(used, k, max_scope, rng, max_nodes=500000):
    """
    Try to build one row with all entries <= max_scope using backtracking.
    Returns row or None.
    """
    row = [0]
    nodes = [0]
    timeout = time.time() + 2.5

    def bt():
        if time.time() > timeout or nodes[0] > max_nodes:
            return False
        nodes[0] += 1
        if len(row) == k + 1:
            return True
        start = row[-1] + 1
        end = min(max_scope + 1, start + 100)
        candidates = list(range(start, end))
        if rng and len(candidates) > 15:
            rng.shuffle(candidates[10:])
        for x in candidates:
            if x > max_scope:
                break
            new = {x - a for a in row}
            if new & used:
                continue
            row.append(x)
            used.update(new)
            if bt():
                return True
            row.pop()
            used.difference_update(new)
        return False

    if bt():
        return row
    return None


def _simulated_annealing(rows, n_rows, k, rng, max_iters=500000):
    """
    Simulated annealing with incremental penalty update.
    """
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    rows = [r[:] for r in rows]

    all_diffs = set()
    for row in rows:
        all_diffs |= _row_diffs(row)

    current_scope = _scope(rows)
    T = 30.0
    cooling = 0.99995

    iters = 0
    while iters < max_iters and _time_left() > 10.0:
        iters += 1

        if iters % 3000 == 0:
            T *= cooling

        row_idx = rng.randint(0, n_rows - 1)
        old_row = rows[row_idx][:]

        if rng.random() < 0.7:
            old_diffs = _row_diffs(old_row)
            all_diffs -= old_diffs

            pos = rng.randint(1, len(old_row) - 1)
            lower = old_row[pos - 1] + 1
            upper = old_row[pos + 1] - 1 if pos < len(old_row) - 1 else best_scope

            search_range = list(range(lower, upper + 1))
            if len(search_range) > 20:
                samples = rng.sample(search_range, 20)
                samples.sort()
            else:
                samples = search_range

            new_row = None
            for new_val in samples:
                candidate = old_row[:pos] + [new_val] + old_row[pos + 1:]
                if any(candidate[i] >= candidate[i + 1] for i in range(len(candidate) - 1)):
                    continue
                cand_diffs = _row_diffs(candidate)
                if len(cand_diffs) != len(candidate) * (len(candidate) - 1) // 2:
                    continue
                if cand_diffs & all_diffs:
                    continue
                new_row = candidate
                break

            if new_row is None:
                all_diffs |= old_diffs
                continue

            new_diffs = _row_diffs(new_row)
            rows[row_idx] = new_row
            new_scope = _scope(rows)
            all_diffs |= new_diffs
        else:
            old_diffs = _row_diffs(old_row)
            all_diffs -= old_diffs

            target = min(rows[row_idx][-1], best_scope)
            if rng.random() < 0.4:
                target = max(k + 1, target - rng.randint(1, 15))

            new_row = _backtrack_row(all_diffs, k, target, rng)
            if new_row is None:
                all_diffs |= old_diffs
                continue

            new_diffs = _row_diffs(new_row)
            if new_diffs & all_diffs:
                all_diffs |= old_diffs
                continue

            rows[row_idx] = new_row
            new_scope = _scope(rows)
            all_diffs |= new_diffs

        if new_scope < current_scope:
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
        else:
            delta = new_scope - current_scope
            if T > 0.1 and rng.random() < math.exp(-delta / T):
                current_scope = new_scope
            else:
                rows[row_idx] = old_row
                all_diffs -= _row_diffs(new_row)
                all_diffs |= old_diffs

    return best_rows, best_scope


def _local_search(rows, n_rows, k, rng, max_iters=100000):
    """
    Local search: try to reduce the largest entry by rebuilding rows or tweaking elements.
    """
    best_scope = _scope(rows)
    rows = [r[:] for r in rows]
    iters = 0

    while iters < max_iters and _time_left() > 5.0:
        iters += 1

        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]

        target = best_scope - 1

        if target < k + 1:
            break

        used_other = _all_diffs_except(rows, max_idx)

        new_row = _backtrack_row(used_other, k, target, rng)
        if new_row is not None:
            rows[max_idx] = new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            continue

        row = rows[max_idx]
        improved = False

        for new_last in range(current_max - 1, row[-2], -1):
            if new_last <= row[-2]:
                break
            new_row = row[:-1] + [new_last]
            internal = _row_diffs(new_row)
            if len(internal) != len(new_row) * (len(new_row) - 1) // 2:
                continue
            if internal & used_other:
                continue
            rows[max_idx] = new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            improved = True
            break

        if not improved:
            found_move = False
            for pos in range(len(row) - 2, 0, -1):
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                for new_val in range(old_val - 1, lower_bound - 1, -1):
                    if new_val < lower_bound:
                        break
                    new_row = row[:pos] + [new_val] + row[pos + 1:]
                    if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                        continue
                    internal = _row_diffs(new_row)
                    if len(internal) != len(new_row) * (len(new_row) - 1) // 2:
                        continue
                    if internal & used_other:
                        continue
                    rows[max_idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    found_move = True
                    break
                if found_move:
                    break

            if not found_move:
                if rng.random() < 0.5:
                    alt_idx = rng.randint(0, n_rows - 1)
                    if alt_idx != max_idx:
                        used_other_alt = _all_diffs_except(rows, alt_idx)
                        target_alt = min(rows[alt_idx][-1] - 1, best_scope - 1)
                        if target_alt >= k + 1:
                            new_row_alt = _backtrack_row(used_other_alt, k, target_alt, rng)
                            if new_row_alt is not None:
                                rows[alt_idx] = new_row_alt
                                new_scope = _scope(rows)
                                if new_scope < best_scope:
                                    best_scope = new_scope

    return rows, best_scope


def _iterative_tightening(rows, n_rows, k, rng):
    """
    Iteratively try to find solutions with smaller and smaller max scope.
    """
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)

    for target in range(best_scope - 1, k + 1, -1):
        if _time_left() < 10.0:
            break

        success = True
        new_rows = []
        used = set()

        for r_idx in range(n_rows):
            if _time_left() < 5.0:
                success = False
                break
            row = _backtrack_row(used, k, target, rng)
            if row is None:
                row = _backtrack_row(used, k, target + 2, rng)
                if row is None:
                    success = False
                    break
            new_rows.append(row)
            used |= _row_diffs(row)

        if success:
            scope = _scope(new_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = new_rows

    return best_rows, best_scope


def _csp_solve(n_rows, k, max_scope, rng):
    """
    Try to solve the full CSP: find n_rows rows of k+1 increasing integers
    starting at 0, all <= max_scope, with all pairwise differences distinct.
    
    Uses backtracking with constraint propagation (arc consistency).
    """
    rows = []
    used = set()
    total_needed = n_rows * (k + 1) * k // 2  # total number of distinct diffs needed
    
    if total_needed > max_scope * (max_scope - 1) // 2:
        return None, None
    
    timeout = time.time() + 15.0
    nodes = [0]
    max_nodes = 5000000
    
    def solve_row(row_idx):
        if time.time() > timeout or nodes[0] > max_nodes:
            return False
        nodes[0] += 1
        
        row = [0]
        
        def bt_row():
            if time.time() > timeout or nodes[0] > max_nodes:
                return False
            nodes[0] += 1
            if len(row) == k + 1:
                rows.append(row[:])
                used.update(_row_diffs(row))
                return True
            
            start = row[-1] + 1
            end = min(max_scope + 1, start + 120)
            candidates = list(range(start, end))
            rng.shuffle(candidates)
            
            for x in candidates:
                if x > max_scope:
                    break
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                if len(new_diffs) != len(row) + 1:
                    continue
                row.append(x)
                used.update(new_diffs)
                if bt_row():
                    return True
                row.pop()
                used.difference_update(new_diffs)
            return False
        
        if bt_row():
            if row_idx + 1 < n_rows:
                if solve_row(row_idx + 1):
                    return True
                # Backtrack: remove last row
                rows.pop()
                removed_diffs = _row_diffs(rows[-1] if rows else [])
                # Actually we need to be more careful
                # Let's redo: pop the row we just added
                # The row was already popped by bt_row if it failed... 
                # Actually bt_row returns False and doesn't add to rows
                # Wait, bt_row only adds to rows on success
                # So if solve_row(row_idx+1) failed, we need to undo what bt_row did
                # But bt_row already cleaned up on failure... 
                # Actually no: if bt_row succeeded (returned True), it appended to rows
                # and updated used. Then if the next row fails, we need to undo.
                if rows:
                    last_row = rows.pop()
                    used.difference_update(_row_diffs(last_row))
                return False
            else:
                return True
        
        return False
    
    if solve_row(0) and len(rows) == n_rows:
        return rows, max_scope
    return None, None


def construct(n_rows, k):
    global _DEADLINE, _DEADLINE_SET
    _DEADLINE = time.time() + _TIME_LIMIT
    _DEADLINE_SET = True

    rng = random.Random(42)

    # Phase 1: Diverse randomized constructions
    strategies = ["pure", "shuffled", "lookahead", "pressure", "shuffled", "lookahead", "pure", "pressure"]
    best_rows = None
    best_scope = float('inf')
    
    for i, strategy in enumerate(strategies):
        if _time_left() < 15.0:
            break
        rows = _construct_diverse(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
    
    # More randomized restarts
    attempts = 0
    while _time_left() > 20.0 and attempts < 80:
        attempts += 1
        strategy = rng.choice(["pure", "shuffled", "lookahead", "pressure"])
        rows = _construct_diverse(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            if best_scope <= 112:
                break

    # Phase 2: Simulated annealing
    if _time_left() > 30.0:
        rows_sa, scope_sa = _simulated_annealing(best_rows, n_rows, k, rng, max_iters=300000)
        if scope_sa < best_scope:
            best_scope = scope_sa
            best_rows = rows_sa
    
    # Phase 2b: Local search
    if _time_left() > 15.0:
        rows_ls, scope_ls = _local_search(best_rows, n_rows, k, rng, max_iters=50000)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls

    # Phase 3: Iterative tightening
    if _time_left() > 15.0:
        rows_it, scope_it = _iterative_tightening(best_rows, n_rows, k, rng)
        if scope_it < best_scope:
            best_scope = scope_it
            best_rows = rows_it
    
    # Phase 3b: Second SA pass
    if _time_left() > 25.0:
        rows_sa2, scope_sa2 = _simulated_annealing(best_rows, n_rows, k, rng, max_iters=200000)
        if scope_sa2 < best_scope:
            best_scope = scope_sa2
            best_rows = rows_sa2

    # Phase 4: Additional restarts
    if _time_left() > 25.0:
        extra_attempts = 0
        while _time_left() > 15.0 and extra_attempts < 40:
            extra_attempts += 1
            strategy = rng.choice(["pressure", "shuffled", "lookahead"])
            rows = _construct_diverse(n_rows, k, rng, strategy)
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= 112:
                    break
    
    # Phase 4b: Final SA polish
    if _time_left() > 12.0:
        rows_sa3, scope_sa3 = _simulated_annealing(best_rows, n_rows, k, rng, max_iters=100000)
        if scope_sa3 < best_scope:
            best_scope = scope_sa3
            best_rows = rows_sa3

    # Phase 5: Final local search
    if _time_left() > 8.0:
        rows_ls2, scope_ls2 = _local_search(best_rows, n_rows, k, rng, max_iters=50000)
        if scope_ls2 < best_scope:
            best_scope = scope_ls2
            best_rows = rows_ls2

    # Final validation
    if best_rows is None or not _validate(best_rows):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, None, "pure")
            rows.append(row)
        return rows

    return best_rows
