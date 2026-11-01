"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from functools import lru_cache

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    best_solution = None
    best_max = float('inf')
    
    # Multiple random restarts with different strategies
    for attempt in range(200):
        solution = _construct_single(n_rows, k, attempt)
        if solution is None:
            continue
        max_val = max(max(row) for row in solution)
        if max_val < best_max:
            best_max = max_val
            best_solution = solution
            if best_max <= 134:  # Best known bound
                break
    
    if best_solution is None:
        # Fallback to greedy if all else fails
        best_solution = _greedy_fallback(n_rows, k)
    
    return best_solution

def _greedy_fallback(n_rows, k):
    """Original greedy approach as fallback."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _construct_single(n_rows, k, seed):
    """
    Single construction attempt using backtracking with randomized choices.
    """
    rng = random.Random(seed * 12345 + 67890)
    
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        row = _construct_row_backtrack(used, k, rng, max_tries=50)
        if row is None:
            return None
        rows.append(row)
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    return rows

def _construct_row_backtrack(used, k, rng, max_tries=50):
    """
    Construct a single row using backtracking.
    Returns a list of k+1 integers starting at 0, or None if failed.
    """
    for attempt in range(max_tries):
        row = [0]
        if _extend_row(row, used, k, rng, depth=0):
            return row
    return None

def _extend_row(row, used, k, rng, depth, max_depth=20):
    """
    Recursively extend the row with backtracking.
    """
    if len(row) == k + 1:
        return True
    
    if depth > max_depth:
        return False
    
    last = row[-1]
    # Try different gaps to add next element
    # Use randomized ordering to explore different solutions
    gap_options = list(range(1, 30))
    rng.shuffle(gap_options)
    
    for gap in gap_options:
        candidate = last + gap
        new_diffs = {candidate - a for a in row}
        
        if new_diffs & used:
            continue
        
        # Check if this choice is "safe" - not too restrictive
        row.append(candidate)
        used.update(new_diffs)
        
        if _extend_row(row, used, k, rng, depth + 1):
            return True
        
        # Backtrack
        used.difference_update(new_diffs)
        row.pop()
    
    return False
# EVOLVE-BLOCK-END
