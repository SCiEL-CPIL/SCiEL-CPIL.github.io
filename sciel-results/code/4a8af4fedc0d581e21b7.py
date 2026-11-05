"""
Difference triangle set (7,5) solver.

Strategy:
1. Column-wise list-scheduling construction for a fast initial solution.
2. Bounded backtracking to find solutions at progressively lower scopes.
3. Element-wise local search (greedy descent) for polish.
4. Simulated annealing to escape local minima.
5. Final polish pass.
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _row_diffs_list(row):
    """Return list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs


def _row_mask(row):
    """Return frozenset of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    all_diffs = []
    for r in rows:
        all_diffs.extend(_row_diffs_list(r))
    return len(all_diffs) == len(set(all_diffs))


def _compute_all_diffs(rows):
    """Return set of all differences used."""
    used = set()
    for r in rows:
        used |= _row_mask(r)
    return used


def _column_wise_construct(n_rows, k, deadline):
    """
    Column-wise list-scheduling construction.
    Build all rows simultaneously, processing positions 0..k.
    At each position, for each row, find the smallest value > previous
    that doesn't create duplicate differences globally.
    
    Returns list of n_rows rows, or None if stuck.
    """
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    for pos in range(1, k + 1):
        # For this position, find the earliest feasible value for each row
        # Process rows in order of their current last value (smallest first)
        row_order = sorted(range(n_rows), key=lambda i: rows[i][-1])
        
        for idx in row_order:
            row = rows[idx]
            prev_val = row[-1]
            
            # Try values starting from prev_val + 1
            x = prev_val + 1
            while True:
                # Check if this value creates any duplicate differences
                new_diffs = set()
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used_diffs:
                        break
                    new_diffs.add(d)
                else:
                    # No conflicts, use this value
                    row.append(x)
                    used_diffs |= new_diffs
                    break
                x += 1
                # Safety: if we've gone too far, return None
                if x > 500:
                    return None
    
    return rows


def _column_wise_construct_randomized(n_rows, k, rng, deadline):
    """
    Randomized column-wise construction with multiple strategies.
    Returns best solution found.
    """
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple randomizations
    for attempt in range(15):
        if time.time() > deadline:
            break
        
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        for pos in range(1, k + 1):
            # Randomly shuffle row processing order
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            
            for idx in row_order:
                row = rows[idx]
                prev_val = row[-1]
                
                # Find candidates: try a range of values
                # Bias toward smaller values but allow some randomness
                x = prev_val + 1
                found = False
                max_try = prev_val + 80
                
                while x <= max_try:
                    new_diffs = set()
                    conflict = False
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    
                    if not conflict:
                        # With some probability, skip this to explore larger values
                        if rng.random() < 0.75:
                            row.append(x)
                            used_diffs |= new_diffs
                            found = True
                            break
                        # Otherwise, try next value
                    x += 1
                
                if not found:
                    # Try harder: go further
                    for x in range(prev_val + 1, prev_val + 250):
                        new_diffs = set()
                        conflict = False
                        for i in range(len(row)):
                            d = x - row[i]
                            if d in used_diffs:
                                conflict = True
                                break
                            new_diffs.add(d)
                        if not conflict:
                            row.append(x)
                            used_diffs |= new_diffs
                            found = True
                            break
                
                if not found:
                    break
        
        if found and len(rows[0]) == k + 1:
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [r[:] for r in rows]
    
    return best_rows


def _try_move_element(rows, idx, pos, new_val, used):
    """
    Try moving element at rows[idx][pos] to new_val.
    Returns (success, new_used) if successful.
    """
    row = rows[idx]
    old_val = row[pos]
    
    old_diffs = set()
    new_diffs = set()
    for i in range(len(row)):
        if i == pos:
            continue
        if i < pos:
            old_diffs.add(old_val - row[i])
            new_diffs.add(new_val - row[i])
        else:
            old_diffs.add(row[i] - old_val)
            new_diffs.add(row[i] - new_val)
    
    remaining = used - old_diffs
    if new_diffs & remaining:
        return False, used
    
    new_used = (used - old_diffs) | new_diffs
    return True, new_used


def _element_local_search(rows, k, deadline):
    """
    Element-wise local search to reduce scope.
    """
    rows = [r[:] for r in rows]
    used = _compute_all_diffs(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for idx in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[idx]
            
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    success, new_used = _try_move_element(rows, idx, pos, new_val, used)
                    if success:
                        row[pos] = new_val
                        used = new_used
                        improved = True
                        break
    
    return rows


def _backtrack_row(used, max_val, row, depth, k, deadline):
    """
    Build one row via backtracking within max_val.
    row is a list, depth is current length.
    Returns completed row or None.
    """
    if time.time() > deadline:
        return None
    
    if depth == k + 1:
        return row[:]
    
    last = row[-1]
    start = last + 1
    
    # Try values in order (deterministic for reproducibility within a seed)
    for x in range(start, max_val + 1):
        if time.time() > deadline:
            return None
        
        new_diffs = set()
        conflict = False
        for a in row:
            d = x - a
            if d in used:
                conflict = True
                break
            new_diffs.add(d)
        
        if conflict:
            continue
        
        row.append(x)
        used_copy = used | new_diffs
        
        result = _backtrack_row(used_copy, max_val, row, depth + 1, k, deadline)
        if result is not None:
            return result
        
        row.pop()
    
    return None


def _backtrack_construct(max_val, n_rows, k, deadline, rng=None):
    """
    Try to construct all rows within max_val using backtracking.
    Returns list of rows or None.
    """
    used = set()
    rows = []
    
    # Determine row order - shuffle for diversity if rng provided
    row_indices = list(range(n_rows))
    if rng:
        rng.shuffle(row_indices)
    
    for idx in row_indices:
        if time.time() > deadline:
            return None
        
        row = [0]
        result = _backtrack_row(used, max_val, row, 1, k, deadline)
        if result is None:
            return None
        
        rows.append(result)
        # Add diffs to used
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                used.add(result[j] - result[i])
    
    # Sort rows back to original order (by max value or just keep as is)
    # Actually, the order doesn't matter for validity, but let's sort by max for consistency
    rows.sort(key=lambda r: r[-1])
    
    return rows


def _simulated_annealing(rows, k, deadline, rng):
    """
    Simulated annealing to escape local minima.
    Allows temporary scope increases to find better solutions.
    """
    rows = [r[:] for r in rows]
    used = _compute_all_diffs(rows)
    best_rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    
    T = 15.0
    cooling = 0.998
    
    while time.time() < deadline and T > 0.1:
        # Pick a random row and position
        idx = rng.randint(0, len(rows) - 1)
        pos = rng.randint(1, len(rows[idx]) - 1)
        
        row = rows[idx]
        old_val = row[pos]
        
        # Pick a random new value
        # Bias towards smaller values
        if rng.random() < 0.7:
            delta = rng.randint(-8, 3)
        else:
            delta = rng.randint(-3, 8)
        new_val = old_val + delta
        
        # Ensure validity: must be > previous and < next
        if pos > 0 and new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            T *= cooling
            continue
        if new_val <= 0:
            T *= cooling
            continue
        
        success, new_used = _try_move_element(rows, idx, pos, new_val, used)
        if not success:
            T *= cooling
            continue
        
        old_scope = _scope(rows)
        rows[idx][pos] = new_val
        used = new_used
        new_scope = _scope(rows)
        
        # Accept if better, or with probability exp(-delta/T)
        delta_scope = new_scope - old_scope
        if delta_scope <= 0 or rng.random() < math.exp(-delta_scope / T):
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
        else:
            # Reject: revert
            rows[idx][pos] = old_val
            # Recompute used
            used = _compute_all_diffs(rows)
        
        T *= cooling
    
    return best_rows


def _construct(n_rows, k, deadline):
    """
    Main construction: column-wise + backtracking + refinement.
    """
    start = time.time()
    
    # Step 1: Column-wise construction (fast initial solution)
    rows = _column_wise_construct(n_rows, k, deadline)
    
    if rows is None:
        # Fallback to pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_scope = _scope(rows)
    best_rows = [r[:] for r in rows]
    
    # Step 2: Randomized column-wise constructions
    if time.time() < deadline - 25:
        rng = random.Random(42)
        alt_rows = _column_wise_construct_randomized(n_rows, k, rng, deadline - 25)
        if alt_rows is not None:
            alt_scope = _scope(alt_rows)
            if alt_scope < best_scope:
                best_scope = alt_scope
                best_rows = alt_rows
    
    # Step 3: Bounded backtracking to find solutions at lower scopes
    # Start from current best and try to go lower
    current_max = best_scope
    lower_bound = n_rows * k * (k + 1) // 2  # Counting bound
    
    while time.time() < deadline - 30:
        target_scope = current_max - 1
        
        if target_scope < lower_bound:
            break
        
        # Try backtracking with this max_val
        deadline_bt = min(deadline - 5, time.time() + 15)
        rng = random.Random(int(time.time() * 1000) % 10000)
        
        result = _backtrack_construct(target_scope, n_rows, k, deadline_bt, rng)
        
        if result is not None:
            best_rows = result
            best_scope = target_scope
            current_max = target_scope
        else:
            # If backtracking fails, try a few more times with different seeds
            found = False
            for attempt in range(3):
                if time.time() > deadline - 10:
                    break
                rng = random.Random(attempt * 12345 + int(time.time() * 1000) % 10000)
                deadline_bt = min(deadline - 5, time.time() + 10)
                result = _backtrack_construct(target_scope, n_rows, k, deadline_bt, rng)
                if result is not None:
                    best_rows = result
                    best_scope = target_scope
                    current_max = target_scope
                    found = True
                    break
            
            if not found:
                # Give up on this target, try one lower
                current_max = target_scope
    
    # Step 4: Element local search
    if time.time() < deadline - 10:
        rows = _element_local_search(best_rows, k, deadline - 10)
        sc = _scope(rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = rows
    
    # Step 5: Simulated annealing for final polish
    if time.time() < deadline - 5:
        rng = random.Random(7)
        sa_rows = _simulated_annealing(best_rows, k, deadline - 5, rng)
        sc = _scope(sa_rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = sa_rows
    
    # Final element local search
    if time.time() < deadline - 2:
        rows = _element_local_search(best_rows, k, deadline - 2)
        sc = _scope(rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = rows
    
    return best_rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    rows = _construct(n_rows, k, deadline)
    
    # Validate
    if rows is None or len(rows) != n_rows:
        # Ultimate fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
    
    return rows

# EVOLVE-BLOCK-END
