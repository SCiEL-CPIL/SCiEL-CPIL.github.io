"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0.
Uses deterministic backtracking with MRV heuristics to partition differences into disjoint Sidon sets.
"""
import time
import random
import sys
import math

_START_TIME = None
_DEADLINE = 145.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_diffs_set(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs_valid(rows, n_rows, k):
    """Check that all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        d = _compute_diffs_set(row)
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_build(n_rows, k, row_order, rng):
    """Greedy: build rows in given order, each row greedily picks smallest valid next value."""
    used = set()
    rows = [None] * n_rows
    for r in row_order:
        row = [0]
        used_in_row = set()
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                # Check new differences
                ok = True
                for a in row:
                    d = x - a
                    if d in used or d in used_in_row:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                        used_in_row.add(x - a)
                    break
                x += 1
        rows[r] = row
    return rows

def _backtracking_solve(n_rows, k, max_scope, rng):
    """
    Deterministic backtracking with MRV heuristics.
    We try to assign differences 1..max_scope to rows such that each row forms a valid Sidon set.
    
    Representation:
    - Each row is built incrementally. We track for each row the set of used differences.
    - MRV: at each step, pick the row that has the fewest available next values (tightest constraint).
    
    Actually, a cleaner formulation:
    - We have n_rows rows, each needs k+1 elements starting at 0.
    - The differences that must be assigned: for each row, we need to choose k elements (the non-zero ones).
    - The constraint is that all pairwise differences across all rows are distinct.
    
    Let's use a different approach: build rows one by one using backtracking.
    For each row, we choose k+1 elements from 0..max_scope such that:
    1. They start at 0 and increase strictly.
    2. All pairwise differences are distinct within the row.
    3. No difference conflicts with differences in previously built rows.
    
    MRV heuristic: when building a row, at each position, choose the value that minimizes
    the number of valid extensions for the remaining positions.
    """
    if _time_left() < 2.0:
        return None
    
    # Precompute: for a given row prefix and used differences, find valid next values
    used_global = set()  # all differences used so far across all rows
    rows = []
    
    def _build_row(prefix, used_in_row, pos, depth):
        """
        Recursively build a row. prefix is the list of chosen elements so far.
        used_in_row is the set of differences within this row.
        pos is the current position (next index to fill).
        depth is for recursion limit.
        """
        if _time_left() < 1.0:
            return None
        
        if pos == k + 1:
            return list(prefix)
        
        last = prefix[-1] if prefix else 0
        start_val = last + 1
        end_val = max_scope + 1
        
        # MRV: try values that minimize future constraints
        # For simplicity, try from start_val upward, but with some randomization
        candidates = list(range(start_val, end_val))
        
        # Heuristic: prefer values that leave more room for future elements
        # Actually, for MRV we want to try the most constrained first
        # Let's just try in order with some randomization
        rng.shuffle(candidates)
        
        for val in candidates:
            # Check if val creates valid differences
            new_diffs = set()
            ok = True
            for a in prefix:
                d = val - a
                if d in used_in_row or d in used_global:
                    ok = False
                    break
                new_diffs.add(d)
            
            if not ok:
                continue
            
            # Check internal validity: all diffs in new_diffs are distinct (they are by construction)
            # and don't conflict with used_in_row (checked above)
            
            # Recurse
            prefix.append(val)
            used_in_row_copy = used_in_row | new_diffs
            
            result = _build_row(prefix, used_in_row_copy, pos + 1, depth + 1)
            if result is not None:
                return result
            
            prefix.pop()
            used_in_row = used_in_row  # no change, we used a copy
        
        return None
    
    # Build rows one by one
    for r in range(n_rows):
        if _time_left() < 2.0:
            return None
        
        row = _build_row([0], set(), 1, 0)
        if row is None:
            return None
        
        # Update used_global
        rd = _compute_diffs_set(row)
        used_global |= rd
        rows.append(row)
    
    return rows

def _backtracking_with_restarts(n_rows, k, max_scope, rng, num_restarts=10):
    """Try backtracking multiple times with different random seeds."""
    for _ in range(num_restarts):
        if _time_left() < 3.0:
            break
        result = _backtracking_solve(n_rows, k, max_scope, rng)
        if result is not None:
            return result
    return None

def _local_search_reduce_max(rows, n_rows, k, rng, time_budget):
    """Local search: try to reduce the max entry by moving elements."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    # Precompute difference sets
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    while time.time() - start < time_budget and _time_left() > 3.0:
        # Find the row with the maximum element
        max_r = max(range(n_rows), key=lambda r: max(rows[r]))
        row = rows[max_r]
        
        # Try to reduce the last element
        old_val = row[-1]
        improved = False
        for new_val in range(old_val - 1, row[-2], -1):
            new_diffs = set()
            ok = True
            for a in row[:-1]:
                d = new_val - a
                if d in row_diffs[max_r] or d in all_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            
            # Check against other rows
            conflict = False
            for other_r in range(n_rows):
                if other_r == max_r:
                    continue
                if new_diffs & row_diffs[other_r]:
                    conflict = True
                    break
            if not conflict:
                row[-1] = new_val
                row_diffs[max_r] = _compute_diffs_set(row)
                all_diffs = set()
                for rr in range(n_rows):
                    all_diffs |= row_diffs[rr]
                cur_max = max(max(rows[rr]) for rr in range(n_rows))
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(rows[rr]) for rr in range(n_rows)]
                improved = True
                break
        
        if not improved:
            # Try random perturbation
            rr = rng.randrange(n_rows)
            pos = rng.randrange(1, len(rows[rr]))
            old = rows[rr][pos]
            delta = rng.randint(-5, 5)
            new = old + delta
            if new <= rows[rr][pos-1] or (pos < k and new >= rows[rr][pos+1]) or new <= 0:
                continue
            
            row_copy = list(rows[rr])
            row_copy[pos] = new
            nd = _compute_diffs_set(row_copy)
            if len(nd) != k * (k + 1) // 2:
                continue
            
            conflict = False
            for other_r in range(n_rows):
                if other_r == rr:
                    continue
                if nd & row_diffs[other_r]:
                    conflict = True
                    break
            if not conflict:
                rows[rr][pos] = new
                row_diffs[rr] = nd
                all_diffs = set()
                for x in range(n_rows):
                    all_diffs |= row_diffs[x]
                cur_max = max(max(rows[x]) for x in range(n_rows))
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(rows[x]) for x in range(n_rows)]
    
    return best_rows, best_max

def _simulated_annealing(rows, n_rows, k, rng, time_budget):
    """SA for further optimization."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    
    T = 50.0
    T_min = 0.5
    cooling = 0.999
    
    while T > T_min and time.time() - start < time_budget and _time_left() > 3.0:
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        row = rows[r]
        old_val = row[pos]
        delta = rng.randint(-10, 10)
        new_val = old_val + delta
        
        if new_val <= row[pos-1] or (pos < k and new_val >= row[pos+1]) or new_val <= 0:
            T *= cooling
            continue
        
        row_copy = list(row)
        row_copy[pos] = new_val
        new_rd = _compute_diffs_set(row_copy)
        
        if len(new_rd) != k * (k + 1) // 2:
            T *= cooling
            continue
        
        conflict_count = 0
        for other_r in range(n_rows):
            if other_r == r:
                continue
            conflict_count += len(new_rd & row_diffs[other_r])
        
        if conflict_count == 0:
            row[pos] = new_val
            row_diffs[r] = new_rd
            cur_max = max(max(rows[x]) for x in range(n_rows))
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(rows[x]) for x in range(n_rows)]
        else:
            if rng.random() < 0.001 * (T / 50.0) and conflict_count <= 2:
                row[pos] = new_val
                row_diffs[r] = new_rd
        
        T *= cooling
    
    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Greedy to get an initial solution
    best_rows = None
    best_max = float('inf')
    
    for trial in range(30):
        if _time_left() < 10.0:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, order, rng)
        if rows is not None:
            cur_max = max(max(r) for r in rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
    
    # Phase 2: Try backtracking with a slightly smaller scope
    # Estimate a target scope: try to reduce best_max
    target_scope = best_max - 1
    min_scope = int(math.sqrt(n_rows * k * (k + 1) / 2)) + 1  # counting bound
    
    for scope in range(target_scope, max(min_scope, target_scope - 20), -1):
        if _time_left() < 10.0:
            break
        result = _backtracking_with_restarts(n_rows, k, scope, rng, num_restarts=5)
        if result is not None:
            cur_max = max(max(r) for r in result)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(r) for r in result]
                # Continue trying smaller scopes
                target_scope = cur_max - 1
            else:
                break
    
    # Phase 3: Local search to polish
    if _time_left() > 10.0:
        budget = min(30.0, _time_left() - 5.0)
        ls_rows, ls_max = _local_search_reduce_max(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    
    # Phase 4: Simulated annealing
    if _time_left() > 15.0:
        budget = min(40.0, _time_left() - 5.0)
        sa_rows, sa_max = _simulated_annealing(best_rows, n_rows, k, rng, budget)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
    
    # Final local search
    if _time_left() > 5.0:
        budget = min(10.0, _time_left() - 2.0)
        ls_rows, ls_max = _local_search_reduce_max(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    
    # Final verification
    if not _all_diffs_valid(best_rows, n_rows, k):
        # Should not happen, but try to fix with greedy
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
    
    return best_rows
