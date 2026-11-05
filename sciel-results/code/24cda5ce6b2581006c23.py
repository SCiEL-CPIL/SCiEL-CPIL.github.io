"""Difference triangle set (7,5) – randomized bitmask backtracking with deadline guard."""
import time
import random
import sys

# We only need to handle the specific task parameters, but keep it general.
# n_rows = 7, k = 5  -> each row has k+1 = 6 entries.
# Total differences per row: C(6,2) = 15.
# Total differences across 7 rows: 105.
# Max possible difference value: if largest entry is M, max diff is M.
# We need 105 distinct positive integers, so M >= 105 (counting bound).
# Best known is 112.

# Bitmask approach:
# A difference d (1..M) can be represented as bit (d-1) in an integer bitmask.
# If M is up to ~200, we can use Python's arbitrary-precision integers as bitmasks.

def construct(n_rows, k):
    """
    Returns a list of n_rows rows, each a list of k+1 strictly increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # Number of entries per row
    m = k + 1  # 6
    # Number of differences per row
    diffs_per_row = m * (m - 1) // 2  # 15
    total_diffs = n_rows * diffs_per_row  # 105

    # Lower bound on scope from counting: we need total_diffs distinct positive ints
    # so scope >= total_diffs = 105
    lower_bound = total_diffs
    # Best known upper bound
    upper_bound = 112

    best_result = None
    best_scope = upper_bound + 10  # start above known best

    # Try scopes from lower_bound upward
    for scope in range(lower_bound, min(upper_bound + 15, lower_bound + 30)):
        if time.time() > deadline - 5:
            break

        # Try to find a solution with this scope
        result = _solve_with_scope(n_rows, m, scope, deadline)
        if result is not None:
            best_result = result
            best_scope = scope
            # Since we're going upward, first valid is best for this scope
            # We can break if we want the first found, but let's see if we can do better
            # Actually, we're iterating scope upward, so first valid scope is optimal
            break

    if best_result is not None:
        return best_result

    # Fallback: use greedy to get a valid (possibly suboptimal) result
    return _greedy_construct(n_rows, k)


def _solve_with_scope(n_rows, m, scope, deadline):
    """
    Try to construct n_rows rows of m entries (starting at 0, strictly increasing, max <= scope)
    such that all positive differences are distinct.
    Uses randomized backtracking with bitmask for difference tracking.
    """
    # Precompute: for a row with entries [0, a1, a2, ..., a_{m-1}], the differences are
    # a_j - a_i for 0 <= i < j <= m-1. Since a_0 = 0, differences are a_j - a_i.
    
    # We'll build rows one by one. For each row, we choose m-1 entries (a1..a_{m-1}) from 1..scope
    # such that all differences are new (not used by previous rows) and within the row are distinct.
    
    # Bitmask: bit d-1 set means difference d is used.
    # scope up to ~120, so bitmask is an integer with up to 120 bits.
    
    used_mask = 0
    rows = []
    
    # We'll use backtracking for each row. For efficiency, we try to build rows in a way
    # that leaves flexibility. Randomize the order of candidate entries.
    
    rng = random.Random()
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        row = _build_row(m, scope, used_mask, row_idx, deadline, rng)
        if row is None:
            return None
        rows.append(row)
        # Update used_mask
        for i in range(m):
            for j in range(i + 1, m):
                d = row[j] - row[i]
                used_mask |= (1 << (d - 1))
    
    return rows


def _build_row(m, scope, used_mask, row_idx, deadline, rng):
    """
    Build a single row of m entries starting at 0, strictly increasing, max <= scope,
    such that all differences are not in used_mask and are distinct within the row.
    Returns the row list or None if not found.
    """
    # We need to choose m-1 values from 1..scope
    # Differences within the row must be distinct and not in used_mask.
    
    # Backtracking: choose a1, a2, ..., a_{m-1} in order.
    # For speed, we can use randomized restarts within this function.
    
    max_attempts = 5000
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        row = _backtrack_row(m, scope, used_mask, row_idx, deadline, rng)
        if row is not None:
            return row
    
    return None


def _backtrack_row(m, scope, used_mask, row_idx, deadline, rng):
    """
    Single backtracking attempt to build a row.
    """
    row = [0]
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == m:
            return True
        
        # Choose a value for row[pos] from 1..scope, greater than row[pos-1]
        # We'll randomize the order of candidates for diversity
        start_val = row[-1] + 1
        candidates = list(range(start_val, scope + 1))
        rng.shuffle(candidates)
        
        for val in candidates:
            # Check if adding val creates valid differences
            # Differences: val - row[i] for i in 0..pos-1
            valid = True
            new_diffs = []
            for i in range(pos):
                d = val - row[i]
                # d must be positive (guaranteed since val > row[i])
                # d must not be in used_mask
                if used_mask & (1 << (d - 1)):
                    valid = False
                    break
                # d must not equal any previous difference in this row
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            row.append(val)
            if backtrack(pos + 1):
                return True
            row.pop()
        
        return False
    
    if backtrack(1):
        return row
    return None


def _greedy_construct(n_rows, k):
    """
    Fallback greedy construction (same as original seed).
    """
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
