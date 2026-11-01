"""Construct difference triangle sets with minimal scope using a structured search."""
import time
import sys
import random
import itertools

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry (scope).
    """
    deadline = time.time() + 110  # safety margin before 120s kill
    
    # Total differences per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Counting bound: we need total_diffs distinct differences from 1..M
    # where M is the scope (largest entry). Differences from a row with max M are in 1..M.
    # Minimum M such that M >= total_diffs is a lower bound, but also each row's diffs
    # must fit within 1..M.
    # Simple lower bound: M >= total_diffs (since we need that many distinct diffs in 1..M)
    lower_bound = total_diffs
    
    # Start searching from a reasonable initial guess
    # The greedy approach gives some upper bound; let's start near the counting bound
    # but we need to be careful: each row has diffs that are differences between its elements.
    
    # Let's first get a baseline with greedy
    greedy_result = _greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in greedy_result) if greedy_result else 0
    
    # Now try to improve using a more structured approach
    # We'll use a simulated annealing / local search approach on the set of rows
    
    best_rows = greedy_result
    best_score = best_scope
    
    # Try to improve by searching for smaller scopes
    # Start from lower_bound and go up
    for target_scope in range(lower_bound, best_scope + 1):
        if time.time() > deadline - 30:
            break
        
        # Try to construct rows with all entries <= target_scope
        result = _try_construct_within_scope(n_rows, k, target_scope)
        if result is not None:
            result_scope = max(max(row) for row in result)
            if result_scope < best_score:
                best_score = result_scope
                best_rows = result
            # Found a valid construction; try to go lower
            # Don't break; try smaller scopes
        else:
            # Could not construct within this scope; try next higher
            # Actually, if we can't construct within target_scope, we can't within smaller
            # So we should break
            break
    
    # If we still have time, try to improve further with local search
    if time.time() < deadline - 10 and best_rows is not None:
        improved = _local_search_improve(best_rows, n_rows, k, deadline)
        if improved is not None:
            improved_scope = max(max(row) for row in improved)
            if improved_scope < best_score:
                best_score = improved_scope
                best_rows = improved
    
    if best_rows is not None:
        return best_rows
    else:
        # Fallback to greedy
        return greedy_result


def _greedy_construct(n_rows, k):
    """Greedy construction as in the seed program."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _try_construct_within_scope(n_rows, k, max_scope, max_time=10.0):
    """
    Try to construct n_rows rows of k+1 elements each, all in [0, max_scope],
    such that all differences are distinct across rows.
    Uses backtracking with MRV heuristic.
    Returns None if not found within time limit.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    diffs_per_row = k * (k + 1) // 2
    
    # State: 
    # - rows: list of lists, each row is a list of k+1 ints (first is 0)
    # - used_diffs: set of all differences used so far
    # - row_idx: current row being built
    # - pos_in_row: position in current row (0 = first element which is 0)
    
    # For MRV, we need to track for each row position, what values are still possible
    # This is complex; let's use a simpler backtracking with ordering
    
    # Alternative: use a random greedy with restarts
    best_found = None
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        result = _random_greedy_with_repair(n_rows, k, max_scope, time.time() < deadline - 5)
        if result is not None:
            # Verify
            if _validate(result, n_rows, k, max_scope):
                return result
    
    return best_found


def _validate(rows, n_rows, k, max_scope):
    """Validate that rows are correct."""
    if len(rows) != n_rows:
        return False
    used_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if any(row[i] >= row[i+1] for i in range(len(row)-1)):
            return False
        if any(x > max_scope for x in row):
            return False
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used_diffs:
                    return False
                used_diffs.add(d)
    return True


def _random_greedy_with_repair(n_rows, k, max_scope, keep_going):
    """
    Random greedy construction with some repair heuristics.
    """
    random.seed()  # Use system random seed
    
    # Try to build rows one by one
    used_diffs = set()
    rows = []
    
    for r in range(n_rows):
        row = _build_single_row(k, max_scope, used_diffs, random_choice=True)
        if row is None:
            return None
        rows.append(row)
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    return rows


def _build_single_row(k, max_scope, used_diffs, random_choice=False):
    """
    Build a single row of k+1 elements starting at 0, all in [0, max_scope],
    such that its differences don't conflict with used_diffs.
    """
    row = [0]
    available = list(range(1, max_scope + 1))
    
    if random_choice:
        random.shuffle(available)
    
    while len(row) < k + 1:
        found = False
        # Try values in order (or shuffled order)
        for x in available:
            if x <= row[-1]:
                continue
            # Check if x can be added
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                # Check that x doesn't create future issues (optional heuristic)
                row.append(x)
                # Also mark these diffs as tentatively used
                for d in new_diffs:
                    used_diffs.add(d)
                found = True
                break
        
        if not found:
            # Backtrack: undo the last addition
            if len(row) > 1:
                last = row.pop()
                for a in row:
                    used_diffs.discard(last - a)
                # If we can't find any value, fail
                # This is a simplification; proper backtracking would be more complex
                return None
            else:
                return None
    
    return row


def _local_search_improve(initial_rows, n_rows, k, deadline):
    """
    Local search to try to reduce the scope.
    This is a placeholder for a more sophisticated local search.
    """
    # For now, just return the initial rows
    # A real implementation would try to modify rows to reduce max entry
    return None

# EVOLVE-BLOCK-END
