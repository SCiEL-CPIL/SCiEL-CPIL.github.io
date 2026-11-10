"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    total_diffs = n_rows * (k + 1) * k // 2
    # We need differences from 1..M where M is the largest entry
    # Each row has k+1 entries, so C(k+1,2) = (k+1)*k//2 differences per row
    
    # Try increasing bounds M until we find a valid assignment
    # Lower bound: we need total_diffs distinct positive differences
    # Max difference in any row is M (from 0 to M), so we need enough room
    # Simple lower bound: M >= ceil(sqrt(2 * total_diffs)) roughly, but tighter:
    # The differences available up to M are 1..M, so M >= total_diffs
    # But actually differences can repeat across the range, just not within same row
    # Wait - all differences must be DISTINCT over all rows. So we need total_diffs
    # distinct values from {1, ..., M}. So M >= total_diffs.
    
    min_M = total_diffs  # absolute lower bound
    # But we can do better: the greedy gives some upper bound
    # Let's first get a greedy solution as initial upper bound
    greedy_rows = _greedy(n_rows, k)
    best_M = max(max(r) for r in greedy_rows)
    
    # Try to improve with CSP backtracking
    best_solution = greedy_rows
    best_scope = best_M
    
    # Try bounds from min_M up to best_M - 1 (we already have best_M)
    # Actually, let's try from a tighter estimate upward
    # The counting bound mentioned is 105 for (7,5), let's use that as hint
    # For general case, try from min_M to best_M - 1
    
    for M in range(min_M, best_M):
        if time.time() > deadline:
            break
        result = _csp_solve(n_rows, k, M, deadline)
        if result is not None:
            best_solution = result
            best_scope = M
            break  # found smallest M, done
    
    return best_solution

def _greedy(n_rows, k):
    """Original greedy construction."""
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

def _csp_solve(n_rows, k, M, deadline):
    """
    CSP backtracking: assign rows one at a time.
    A row is a set of k+1 integers from {0, ..., M} starting at 0, strictly increasing,
    such that all C(k+1,2) pairwise differences are unused.
    
    We select the next row to assign based on difficulty:
    try rows in order, but use backtracking if stuck.
    
    For efficiency, we precompute which difference sets are compatible.
    """
    total_diffs_per_row = (k + 1) * k // 2
    all_diffs = set(range(1, M + 1))
    
    # We'll build rows incrementally using backtracking
    # State: list of assigned rows, set of used differences
    # To make this tractable, we use a smarter approach:
    # Generate candidate rows (subsets of size k+1 from {0,...,M} containing 0)
    # and select n_rows of them with disjoint difference sets.
    
    # This is still huge. Instead, use incremental construction with backtracking.
    
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx, current_row, pos, last_val):
        """
        Build row row_idx. current_row is the partial row so far.
        pos is how many elements we've placed (including 0).
        last_val is the last element placed.
        """
        if time.time() > deadline:
            return False
        
        if pos == k + 1:
            # Row complete
            rows.append(current_row[:])
            if row_idx + 1 == n_rows:
                return True
            return backtrack(row_idx + 1, [0], 1, 0)
        
        # Try next element from last_val+1 to M
        for v in range(last_val + 1, M + 1):
            # Check all differences v - a for a in current_row are unused
            diffs = {v - a for a in current_row}
            if diffs & used_diffs:
                continue
            # Also check: can we still complete this row?
            # We need (k+1 - pos) more elements, each adding (pos) new differences
            # Total remaining diffs for this row: sum of pos, pos+1, ..., k
            # = (k+1-pos)*pos + (k+1-pos)*(k-pos+1)//2 ... let's just check feasibility
            # The max additional difference we can use is M - 0 = M
            # We need enough unused differences remaining
            # For now, just try
            
            # Pruning: the largest difference in this row will be at most M
            # We need all diffs to be <= M, which is guaranteed since v <= M and a >= 0
            
            current_row.append(v)
            used_diffs.update(diffs)
            
            if backtrack(row_idx, current_row, pos + 1, v):
                return True
            
            current_row.pop()
            used_diffs.difference_update(diffs)
        
        return False
    
    # Difficulty ordering: assign the "hardest" row first
    # A row is harder if it has fewer valid completions given current used_diffs
    # For simplicity, start with row 0, but we could reorder
    
    # Actually, let's try a different strategy: 
    # Pre-generate all valid rows (for small M), then do matching
    # But M can be up to ~112, and rows are subsets of size k+1 from {0..M}
    # That's C(M+1, k+1) which is too many for large M
    
    # Better: use the backtracking but with better variable ordering
    # Process rows in an order that maximizes constraint propagation
    
    # For now, try sequential with backtracking
    # Reset state
    used_diffs.clear()
    rows.clear()
    
    success = backtrack(0, [0], 1, 0)
    
    if success:
        return rows
    return None
# EVOLVE-BLOCK-END
