"""Difference triangle set (4,7) — constraint-satisfaction via backtracking with
strong pruning, MRV heuristics, and a two-phase local-search refinement."""
import time
import random
from math import comb

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope) such that all positive differences
    within rows are distinct across all rows."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound: we need total_diffs distinct positive differences in [1, scope]
    lower_bound = total_diffs

    # Get a quick greedy baseline
    greedy_rows = _greedy_construct(n_rows, k)
    if not greedy_rows:
        # Fallback: simple construction
        greedy_rows = [[0] * (k + 1) for _ in range(n_rows)]
        for r in range(n_rows):
            val = 0
            greedy_rows[r][0] = 0
            for j in range(1, k + 1):
                val += 1
                while any(val - greedy_rows[r][i] in _all_diffs(greedy_rows, r, greedy_rows[r]) 
                          for i in range(len(greedy_rows[r]))):
                    val += 1
                greedy_rows[r][j] = val
        # Rebuild properly
        greedy_rows = _greedy_construct(n_rows, k)

    best_rows = greedy_rows
    best_scope = max(max(r) for r in greedy_rows)

    # Phase 1: Backtracking to find optimal or near-optimal scope
    # Try scopes from lower_bound upward
    search_start = max(lower_bound, best_scope - 50)
    if search_start < lower_bound:
        search_start = lower_bound

    # Time allocation for phase 1
    phase1_budget = 60.0
    phase1_end = start_time + phase1_budget

    for scope in range(search_start, best_scope):
        if time.time() > phase1_end:
            break
        t_start = time.time()
        t_budget = min(15.0, phase1_end - t_start)
        if t_budget < 1.0:
            break

        rows = _backtrack_optimal(n_rows, k, scope, t_budget, start_time)
        if rows is not None:
            best_rows = rows
            best_scope = scope
            break  # Found optimal (smallest feasible scope tried)

    # Phase 2: Local search refinement if we have time
    remaining = deadline - time.time() - 10.0
    if remaining > 5.0:
        refined = _local_search_refine(best_rows, n_rows, k, remaining * 0.8)
        if refined is not None:
            new_scope = max(max(r) for r in refined)
            if new_scope < best_scope:
                best_rows = refined
                best_scope = new_scope

    # Phase 3: Final polish - try to reduce scope by 1
    if time.time() < deadline - 15.0 and best_scope > lower_bound:
        t_start = time.time()
        t_budget = min(10.0, deadline - t_start - 5.0)
        if t_budget > 2.0:
            rows = _backtrack_optimal(n_rows, k, best_scope - 1, t_budget, start_time)
            if rows is not None:
                best_rows = rows
                best_scope = best_scope - 1

    return best_rows


def _all_diffs(rows, exclude_row, row):
    """Get all differences used in all rows except possibly exclude_row."""
    diffs = set()
    for r_idx, r in enumerate(rows):
        if r_idx == exclude_row:
            continue
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diffs.add(r[j] - r[i])
    # Also include diffs within the given row
    if row is not None:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
    return diffs


def _greedy_construct(n_rows, k):
    """Fast greedy construction to get an initial upper bound."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
            x += 1
        rows.append(row)
    return rows


def _backtrack_optimal(n_rows, k, max_scope, time_budget, start_time):
    """Backtracking search with strong pruning for a valid construction 
    with max entry <= max_scope."""
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    if max_scope < total_diffs:
        return None

    # Use bitmask for differences (differences go up to max_scope)
    # We need differences 1..max_scope to be unique across all rows
    # Bit 0 is unused, bit d represents difference d
    diff_limit = max_scope
    # We need a bitmask that can hold bits up to diff_limit
    # For efficiency, use Python integers as arbitrary-precision bitmasks

    rows = []
    used_mask = 0  # Global used differences bitmask

    def count_available(row, last_val):
        """Count how many values x > last_val can be placed given current row and used_mask."""
        count = 0
        for x in range(last_val + 1, max_scope + 1):
            ok = True
            for a in row:
                d = x - a
                if (used_mask >> d) & 1:
                    ok = False
                    break
            if ok:
                count += 1
        return count

    def backtrack(row_idx, row, row_mask):
        """Try to construct row row_idx."""
        if time.time() - start_time > time_budget:
            return False

        if row_idx == n_rows:
            return True

        if len(row) == k + 1:
            # Row complete
            rows.append(row[:])
            success = backtrack(row_idx + 1, [0], 0)
            if success:
                return True
            rows.pop()
            return False

        last = row[-1] if row else 0

        # MRV: choose x that leaves the most options for remaining positions
        # Actually, let's just try values in order but with forward checking
        candidates = []
        for x in range(last + 1, max_scope + 1):
            # Check if x can be placed
            conflict = False
            for a in row:
                d = x - a
                if ((used_mask | row_mask) >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                # Forward check: can we still complete this row?
                # Simple check: remaining positions = k + 1 - len(row) - 1
                # Need at least that many valid x' > x
                rem = k + 1 - len(row) - 1
                if rem > 0:
                    # Quick estimate: count available values after x
                    avail = 0
                    for x2 in range(x + 1, min(x + 50, max_scope + 1)):
                        ok2 = True
                        for a in row + [x]:
                            d2 = x2 - a
                            if ((used_mask | row_mask) >> d2) & 1:
                                ok2 = False
                                break
                        if ok2:
                            avail += 1
                        if avail >= rem:
                            break
                    if avail < rem:
                        continue
                candidates.append(x)

        # Sort candidates: prefer values that create smaller differences first (pack tighter)
        # Actually, prefer values closer to last (to keep scope small)
        # Try in natural order first
        for x in candidates:
            # Place x
            row.append(x)
            new_row_mask = row_mask
            for a in row[:-1]:  # All elements except the newly added x
                d = x - a
                new_row_mask |= (1 << d)

            if backtrack(row_idx, row, new_row_mask):
                return True

            row.pop()

        return False

    success = backtrack(0, [0], 0)
    if success:
        return rows
    return None


def _local_search_refine(solution, n_rows, k, time_budget):
    """Local search to reduce the scope of a valid solution."""
    start_time = time.time()
    deadline = start_time + time_budget

    # Work on a copy
    sol = [r[:] for r in solution]
    current_scope = max(max(r) for r in sol)
    best_scope = current_scope
    best_sol = [r[:] for r in sol]

    # Compute all differences
    def compute_diffs(rows):
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.add(r[j] - r[i])
        return diffs

    current_diffs = compute_diffs(sol)

    # Local search: try to reduce individual elements
    # For each row, for each position, try to decrease the value
    iterations = 0
    no_improve_count = 0

    while time.time() < deadline - 1.0:
        improved = False
        random.shuffle(sol)  # Randomize order to escape local optima

        for r_idx in range(n_rows):
            if time.time() > deadline - 1.0:
                break
            for pos in range(1, len(sol[r_idx])):  # Skip position 0 (always 0)
                old_val = sol[r_idx][pos]
                # Try decreasing the value
                for new_val in range(old_val - 1, -1, -1):
                    # Check constraints:
                    # 1. Must be > previous element
                    if new_val <= sol[r_idx][pos - 1]:
                        break
                    # 2. Must be < next element (if exists)
                    if pos < len(sol[r_idx]) - 1 and new_val >= sol[r_idx][pos + 1]:
                        break
                    # 3. Check difference conflicts
                    valid = True
                    # Remove old differences involving this position
                    old_diffs = set()
                    row = sol[r_idx]
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        d = abs(row[pos] - row[i])
                        old_diffs.add(d)
                    
                    # Check if new value creates conflicts
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        d = abs(new_val - row[i])
                        new_diffs.add(d)
                    
                    # A difference d in new_diffs is a conflict if:
                    # - d is in current_diffs but not in old_diffs (it's used elsewhere)
                    # - OR d appears in new_diffs more than once (duplicate within same row)
                    # Actually, we need to check:
                    # For each d in new_diffs:
                    #   - Is d used in any other row?
                    #   - Is d used in this row by a pair not involving pos?
                    
                    conflict = False
                    diff_count_in_row = {}
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        d = abs(new_val - row[i])
                        diff_count_in_row[d] = diff_count_in_row.get(d, 0) + 1
                    
                    for d in diff_count_in_row:
                        if diff_count_in_row[d] > 1:
                            conflict = True
                            break
                        # Check if d is used in other rows
                        for other_r in range(n_rows):
                            if other_r == r_idx:
                                continue
                            orow = sol[other_r]
                            for i in range(len(orow)):
                                for j in range(i + 1, len(orow)):
                                    if orow[j] - orow[i] == d:
                                        conflict = True
                                        break
                                if conflict:
                                    break
                            if conflict:
                                break
                        if conflict:
                            break
                        # Check if d is used in this row by a pair not involving pos
                        if not conflict:
                            for i in range(len(row)):
                                for j in range(i + 1, len(row)):
                                    if i == pos or j == pos:
                                        continue
                                    if row[j] - row[i] == d:
                                        conflict = True
                                        break
                                if conflict:
                                    break
                    
                    if conflict:
                        continue
                    
                    # Valid move: apply it
                    row[pos] = new_val
                    current_diffs -= old_diffs
                    current_diffs |= new_diffs
                    improved = True
                    break  # Move on to next position
                
                if new_val != old_val:
                    pass  # Value was changed

        # Update best
        new_scope = max(max(r) for r in sol)
        if new_scope < best_scope:
            best_scope = new_scope
            best_sol = [r[:] for r in sol]
            no_improve_count = 0
        else:
            no_improve_count += 1

        # If no improvement for many iterations, try a different strategy
        if no_improve_count > 20:
            # Reset to best and try random perturbation
            sol = [r[:] for r in best_sol]
            current_diffs = compute_diffs(sol)
            # Perturb: randomly increase a random element
            for _ in range(3):
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, len(sol[r_idx]) - 1)
                sol[r_idx][pos] += random.randint(1, 5)
            current_diffs = compute_diffs(sol)
            no_improve_count = 0

        iterations += 1

    return best_sol


# EVOLVE-BLOCK-END
