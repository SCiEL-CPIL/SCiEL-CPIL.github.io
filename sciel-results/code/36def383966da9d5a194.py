"""
Difference Triangle Set (4,7) solver using a bitset-based backtracking with pruning.

Key structural change: instead of simulated annealing on a solution space, we use
a systematic backtracking search with aggressive pruning based on a bitset
representation of the used differences. This allows us to explore the solution
space more efficiently and find tighter bounds.
"""
import time
import random
from typing import List, Optional, Set, Tuple


def construct(n_rows: int, k: int) -> List[List[int]]:
    """
    Main entry point: construct n_rows rows of k+1 increasing integers starting at 0.
    
    Returns a list of n_rows rows, each containing k+1 strictly increasing integers
    starting at 0, such that all positive differences within rows are distinct
    across all rows.
    """
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin under 150s
    
    # For the specific case (4,7), we know the best known scope is 134.
    # We'll try to find a solution with scope as small as possible.
    
    # First, try a systematic backtracking approach with pruning
    result = _backtrack_solve(n_rows, k, deadline)
    
    # If backtracking didn't find a good solution, fall back to a greedy + local search
    if result is None:
        result = _greedy_with_local_search(n_rows, k, deadline)
    
    # Final validity check
    if not _check_valid(result):
        # Last resort: deterministic greedy
        result = _deterministic_greedy(n_rows, k)
    
    return result


def _check_valid(rows: List[List[int]]) -> bool:
    """Check if all positive differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True


def _deterministic_greedy(n_rows: int, k: int) -> List[List[int]]:
    """Deterministic greedy: build each row by scanning upward, picking smallest valid value."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
            x += 1
        rows.append(row)
    return rows


def _backtrack_solve(n_rows: int, k: int, deadline: float) -> Optional[List[List[int]]]:
    """
    Backtracking solver with aggressive pruning using bitsets.
    
    We build rows one at a time. For each row, we use a bitset to track which
    differences are already used, and we prune branches early when we detect
    conflicts.
    """
    # Maximum scope to try: start high and work down
    # The counting bound is 112, best known is 134
    # Try from 134 down to 100
    
    best_solution = None
    best_scope = 200  # Start high
    
    # Try scopes in decreasing order
    for max_scope in range(134, 99, -1):
        if time.time() > deadline - 10:
            break
        
        # Try to find a solution with this scope
        solution = _backtrack_with_scope(n_rows, k, max_scope, deadline)
        
        if solution is not None:
            best_solution = solution
            best_scope = max_scope
            # Don't continue; we want the smallest scope
            break
    
    if best_solution is not None:
        return best_solution
    
    # If we didn't find anything in the range, try a wider search
    for max_scope in range(160, 133, -1):
        if time.time() > deadline - 5:
            break
        
        solution = _backtrack_with_scope(n_rows, k, max_scope, deadline)
        
        if solution is not None:
            best_solution = solution
            best_scope = max_scope
            break
    
    return best_solution


def _backtrack_with_scope(n_rows: int, k: int, max_scope: int, deadline: float) -> Optional[List[List[int]]]:
    """
    Try to find a solution where all entries are at most max_scope.
    
    Uses backtracking with symmetry breaking and pruning.
    """
    # Bitset for used differences (differences can be up to max_scope)
    # We'll use a Python set for simplicity, but with careful pruning
    
    def solve(row_idx: int, used_diffs: Set[int], rows: List[List[int]], 
              symmetry_constraints: List[int]) -> Optional[List[List[int]]]:
        if time.time() > deadline - 5:
            return None
        
        if row_idx == n_rows:
            return list(rows)
        
        # Symmetry breaking: enforce row_idx's first element (after 0) to be >= 
        # the previous row's first element to avoid equivalent solutions
        min_first = symmetry_constraints[row_idx - 1] if row_idx > 0 else 1
        
        # Try to build row row_idx
        row_solution = _build_row_backtrack(
            k, max_scope, used_diffs, min_first, deadline
        )
        
        if row_solution is None:
            return None
        
        rows.append(row_solution)
        symmetry_constraints.append(row_solution[1] if len(row_solution) > 1 else 0)
        
        result = solve(row_idx + 1, used_diffs, rows, symmetry_constraints)
        
        if result is not None:
            return result
        
        # Backtrack
        rows.pop()
        symmetry_constraints.pop()
        return None
    
    used_diffs = set()
    rows = []
    symmetry_constraints = [0]  # For row 0, no constraint on first element except >= 1
    
    result = solve(0, used_diffs, rows, symmetry_constraints)
    return result


def _build_row_backtrack(k: int, max_scope: int, used_diffs: Set[int], 
                         min_first: int, deadline: float) -> Optional[List[int]]:
    """
    Build a single row of k+1 elements using backtracking.
    
    Returns None if no valid row can be built.
    """
    def backtrack(row: List[int], local_diffs: Set[int]) -> Optional[List[int]]:
        if time.time() > deadline - 5:
            return None
        
        if len(row) == k + 1:
            return list(row)
        
        last = row[-1]
        start = max(last + 1, min_first) if len(row) == 1 else last + 1
        
        # Try values from start to max_scope
        for val in range(start, max_scope + 1):
            new_diffs = {val - a for a in row}
            
            # Check against used_diffs and local_diffs
            if new_diffs & used_diffs:
                continue
            if new_diffs & local_diffs:
                continue
            
            # Pruning: check if we can still complete the row
            # The remaining elements must create differences that don't conflict
            # Simple pruning: ensure we have enough room
            remaining = k + 1 - len(row) - 1
            if remaining > 0:
                # The largest possible difference we can still create is max_scope - val
                # We need to create 'remaining' more differences
                # This is a loose bound
                if max_scope - val < remaining:
                    continue
            
            row.append(val)
            local_diffs.update(new_diffs)
            
            result = backtrack(row, local_diffs)
            if result is not None:
                return result
            
            row.pop()
            local_diffs.difference_update(new_diffs)
        
        return None
    
    row = [0]
    local_diffs = set()
    return backtrack(row, local_diffs)


def _greedy_with_local_search(n_rows: int, k: int, deadline: float) -> List[List[int]]:
    """
    Fallback: greedy construction followed by local search to improve scope.
    """
    # Start with greedy solution
    rows = _deterministic_greedy(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(rows)
    
    # Local search: try to reduce the scope
    t_start = time.time()
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        
        for row_idx in range(n_rows):
            # Try to improve this row
            used_others = set()
            for j in range(n_rows):
                if j != row_idx:
                    for i in range(len(rows[j])):
                        for j2 in range(i + 1, len(rows[j])):
                            used_others.add(rows[j][j2] - rows[j][i])
            
            current_row = rows[row_idx]
            current_max = max(current_row)
            
            # Try to find a better row with smaller max
            best_new_row = None
            best_new_max = current_max
            
            for attempt in range(50):
                new_row = _try_improve_row(
                    k, current_max - 1, used_others, deadline
                )
                if new_row is not None:
                    new_max = max(new_row)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_new_row = new_row
            
            if best_new_row is not None and best_new_max < current_max:
                rows[row_idx] = best_new_row
                improved = True
        
        if improved:
            no_improve = 0
            current_scope = _max_entry(rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            no_improve += 1
    
    return best_rows


def _try_improve_row(k: int, max_scope: int, used_diffs: Set[int], 
                     deadline: float) -> Optional[List[int]]:
    """Try to build a row with max entry at most max_scope."""
    def backtrack(row: List[int], local_diffs: Set[int]) -> Optional[List[int]]:
        if time.time() > deadline - 5:
            return None
        
        if len(row) == k + 1:
            return list(row)
        
        last = row[-1]
        
        for val in range(last + 1, max_scope + 1):
            new_diffs = {val - a for a in row}
            
            if new_diffs & used_diffs:
                continue
            if new_diffs & local_diffs:
                continue
            
            row.append(val)
            local_diffs.update(new_diffs)
            
            result = backtrack(row, local_diffs)
            if result is not None:
                return result
            
            row.pop()
            local_diffs.difference_update(new_diffs)
        
        return None
    
    row = [0]
    local_diffs = set()
    return backtrack(row, local_diffs)


def _max_entry(rows: List[List[int]]) -> int:
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows) if rows else 0
