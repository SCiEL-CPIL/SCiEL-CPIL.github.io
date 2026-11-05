"""
Difference triangle set (7,5) solver.

Strategy:
1. Use a backtracking row-construction approach (inspiration 1) with heuristic
   candidate ordering to build complete sets that minimize scope.
2. Use a column-wise list-scheduling approach (inspiration 2) as an alternative
   construction method that builds all rows simultaneously.
3. Apply local search and polish passes to reduce scope further.
4. Use bitmask representation for difference sets for speed.
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return frozenset of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _row_mask(row):
    """Return bitmask representation of differences for a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m


def _scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(r[-1] for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= set(_diffs_of_row(r))
    return used


def _greedy_construct(n_rows, k):
    """Pure greedy construction as fallback."""
    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d <= 0:
                        valid = False
                        break
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    break
                x += 1
            row.append(x)
            for d in new_diffs:
                used_mask |= (1 << d)
        rows.append(row)
    return rows


def _column_wise_construct(n_rows, k, rng, max_scope_hint=160):
    """
    List-scheduling style construction:
    - We build all rows simultaneously, column by column.
    - For each row, at each column, we pick the smallest feasible value that
      doesn't conflict with any previously chosen differences.
    """
    num_elements = k + 1
    rows = [[0] for _ in range(n_rows)]
    used_mask = 0
    
    for col in range(1, num_elements):
        # Process rows in random order to break symmetry
        order = list(range(n_rows))
        rng.shuffle(order)
        
        for r in order:
            row = rows[r]
            last_val = row[-1]
            found = False
            
            # Try values from last_val+1 up to a reasonable bound
            upper = max_scope_hint + 10
            for val in range(last_val + 1, upper + 1):
                # Check new differences
                new_diffs = []
                valid = True
                for prev in row:
                    d = val - prev
                    if d <= 0:
                        valid = False
                        break
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Accept this value
                row.append(val)
                for d in new_diffs:
                    used_mask |= (1 << d)
                found = True
                break
            
            if not found:
                # Fallback: jump to a value that works
                val = last_val + 1
                while True:
                    new_diffs = []
                    valid = True
                    for prev in row:
                        d = val - prev
                        if d <= 0:
                            valid = False
                            break
                        if (used_mask >> d) & 1:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        break
                    val += 1
                row.append(val)
                for d in new_diffs:
                    used_mask |= (1 << d)
    
    return rows


def _backtrack_construct(n_rows, k, target_scope, rng, deadline):
    """
    Backtracking solver with heuristic ordering to construct rows
    that fit within target_scope.
    """
    used_mask = 0
    rows = []
    
    def backtrack(row_idx, current_row):
        nonlocal used_mask, rows
        
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        if len(current_row) == k + 1:
            # Row complete
            rows.append(current_row)
            if backtrack(row_idx + 1, [0]):
                return True
            rows.pop()
            return False
        
        last_val = current_row[-1]
        
        # Generate candidates with heuristic ordering
        candidates = []
        for x in range(last_val + 1, target_scope + 1):
            new_diffs = []
            valid = True
            for a in current_row:
                d = x - a
                if d <= 0:
                    valid = False
                    break
                if (used_mask >> d) & 1:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Heuristic: prefer smaller differences (more constrained)
            cost = sum(1.0 / d for d in new_diffs)
            noise = rng.random() * 0.01
            candidates.append((cost + noise, x, new_diffs))
            
            if len(candidates) > 25:
                break
        
        if not candidates:
            return False
        
        candidates.sort(key=lambda item: item[0])
        
        for cost, x, new_diffs in candidates:
            old_mask = used_mask
            for d in new_diffs:
                used_mask |= (1 << d)
            
            if backtrack(row_idx, current_row + [x]):
                return True
            
            used_mask = old_mask
        
        return False
    
    return backtrack(0, [0])


def _local_search_scope(rows, k, rng, deadline):
    """
    Local search to reduce the scope (maximum entry).
    Tries to shrink elements in rows.
    """
    rows = [r[:] for r in rows]
    used_mask = 0
    for r in rows:
        used_mask |= _row_mask(r)
    scope = _scope(rows)

    improved = True
    num_passes = 0
    while improved and time.time() < deadline:
        improved = False
        num_passes += 1
        if num_passes % 3 == 0:
            row_indices = list(range(len(rows)))
            rng.shuffle(row_indices)
        elif num_passes % 3 == 1:
            row_indices = sorted(range(len(rows)), key=lambda i: -rows[i][-1])
        else:
            row_indices = sorted(range(len(rows)), key=lambda i: rows[i][-1])

        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]

            # Try to reduce the last element
            last = row[-1]
            for new_last in range(last - 1, row[-2], -1):
                old_diffs = {last - a for a in row[:-1]}
                new_diffs = {new_last - a for a in row[:-1]}
                
                # Check if new_diffs conflict with used_mask excluding old_diffs
                conflict = False
                for d in new_diffs:
                    if d not in old_diffs and (used_mask >> d) & 1:
                        conflict = True
                        break
                
                if not conflict:
                    # Update mask
                    for d in old_diffs:
                        used_mask &= ~(1 << d)
                    for d in new_diffs:
                        used_mask |= (1 << d)
                    row[-1] = new_last
                    scope = _scope(rows)
                    improved = True
                    break

            # Try to reduce other elements
            for pos in range(len(row) - 2, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    
                    conflict = False
                    for d in new_diffs:
                        if d not in old_diffs and (used_mask >> d) & 1:
                            conflict = True
                            break
                    
                    if not conflict:
                        for d in old_diffs:
                            used_mask &= ~(1 << d)
                        for d in new_diffs:
                            used_mask |= (1 << d)
                        row[pos] = new_val
                        scope = _scope(rows)
                        improved = True
                        break

    return rows


def _polish_pass(rows, k, deadline):
    """
    High-precision polish: try every possible reduction for each position.
    """
    rows = [r[:] for r in rows]
    used_mask = 0
    for r in rows:
        used_mask |= _row_mask(r)
    best_scope = _scope(rows)

    for _ in range(5):
        if time.time() > deadline:
            break
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
                        if time.time() > deadline:
                            break
                        old_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                old_diffs.add(old_val - row[i])
                            else:
                                old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                        
                        conflict = False
                        for d in new_diffs:
                            if d not in old_diffs and (used_mask >> d) & 1:
                                conflict = True
                                break
                        
                        if not conflict:
                            for d in old_diffs:
                                used_mask &= ~(1 << d)
                            for d in new_diffs:
                                used_mask |= (1 << d)
                            row[pos] = new_val
                            improved = True
                            break

    return rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Step 1: Try column-wise construction first (fast)
    rows = _column_wise_construct(n_rows, k, rng, max_scope_hint=150)
    best_scope = _scope(rows)
    best_rows = [r[:] for r in rows]

    # Step 2: Try greedy construction
    greedy_rows = _greedy_construct(n_rows, k)
    greedy_scope = _scope(greedy_rows)
    if greedy_scope < best_scope:
        best_scope = greedy_scope
        best_rows = [r[:] for r in greedy_rows]

    # Step 3: Try backtracking with decreasing target scopes
    for target in range(best_scope - 1, max(best_scope - 20, 100), -1):
        if time.time() > deadline - 30:
            break
        found = _backtrack_construct(n_rows, k, target, rng, deadline - 30)
        if found:
            # Verify the result
            all_diffs = []
            for r in rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        all_diffs.append(r[j] - r[i])
            if len(all_diffs) == len(set(all_diffs)):
                actual_scope = _scope(rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_rows = [r[:] for r in rows]
                break

    # Step 4: Local search to reduce scope
    if time.time() < deadline - 15:
        best_rows = _local_search_scope(best_rows, k, rng, deadline - 15)

    # Step 5: Additional local search iterations
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            best_rows = _local_search_scope(best_rows, k, rng, deadline - 10)

    # Step 6: Polish pass
    if time.time() < deadline - 3:
        best_rows = _polish_pass(best_rows, k, deadline - 3)

    return best_rows

# EVOLVE-BLOCK-END
