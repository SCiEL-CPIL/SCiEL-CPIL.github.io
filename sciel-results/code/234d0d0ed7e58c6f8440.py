"""Difference triangle set solver using gap-based backtracking with bitset pruning."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, subject to all positive differences
    within rows being globally distinct.
    """
    deadline = time.time() + 110.0  # safety margin before 120s kill
    
    # Greedy baseline
    best_rows = _greedy(n_rows, k)
    best_scope = _scope(best_rows)
    
    # Try to improve via backtracking with gap representation
    # We search for a solution with scope < best_scope
    # Use a time-bounded search
    
    # The number of differences per row is k*(k+1)//2
    # Total differences needed: n_rows * k*(k+1)//2
    # Minimum scope from counting: need at least total_diffs distinct positive integers
    # So scope >= total_diffs (since differences range from 1 to scope)
    
    total_diffs = n_rows * k * (k + 1) // 2
    min_scope = total_diffs  # counting bound
    
    # Try scopes from max(min_scope, best_scope - 50) up to best_scope - 1
    # Actually, let's try to find any solution with scope < best_scope
    # We'll try a few target scopes
    
    # First, let's do a quick local search / hill climbing to improve the greedy
    improved = _local_search(best_rows, n_rows, k, deadline)
    if improved is not None:
        s = _scope(improved)
        if s < best_scope:
            best_rows = improved
            best_scope = s
    
    # Now try backtracking for even better solutions
    # We'll try to find a solution with scope = best_scope - 1, then -2, etc.
    # But that's expensive. Let's try a smarter approach.
    
    # Backtracking with gap representation and bitset
    # We'll try to find any valid configuration with scope < best_scope
    # by searching in a limited way
    
    for target_scope in range(best_scope - 1, max(min_scope - 1, 0), -1):
        if time.time() > deadline - 20:
            break
        result = _backtrack_gaps(n_rows, k, target_scope, deadline)
        if result is not None:
            s = _scope(result)
            if s < best_scope:
                best_rows = result
                best_scope = s
            break  # Found the best we can for this target
    
    return best_rows


def _scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(row) for row in rows)


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


def _local_search(rows, n_rows, k, deadline):
    """Simple local search: try to reduce scope by adjusting entries."""
    # This is a placeholder for a more sophisticated local search
    # For now, return None to keep it simple
    return None


def _backtrack_gaps(n_rows, k, max_scope, deadline):
    """
    Backtracking search using gap representation.
    Each row is defined by k gaps (differences between consecutive elements).
    The row entries are cumulative sums of gaps.
    All positive differences between any two entries in a row must be globally unique.
    
    We search for a valid configuration where all rows have span <= max_scope.
    """
    # Total differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Bitset for used differences (differences range from 1 to max_scope)
    used_diffs = 0  # bit i set means difference i is used
    
    # We'll build rows one by one
    # For each row, we choose k gaps such that their cumulative sums give entries
    # and all pairwise differences are unused
    
    # To make this tractable, we'll use a heuristic: try to construct each row
    # greedily but with backtracking when stuck
    
    rows = []
    
    def build_row(row_idx, current_entries, used_diffs_local):
        """
        Build row row_idx. current_entries is the list of entries so far.
        Returns (success, new_used_diffs, new_entries) or (False, None, None)
        """
        if len(current_entries) == k + 1:
            return True, used_diffs_local, current_entries
        
        if time.time() > deadline - 5:
            return False, None, None
        
        last = current_entries[-1]
        # Next entry must be > last and <= max_scope
        # Also, the new differences with all previous entries must be unused
        
        # Try next entries in increasing order
        for next_val in range(last + 1, max_scope + 1):
            # Check if all new differences are unused
            new_diffs = [next_val - e for e in current_entries]
            valid = True
            for d in new_diffs:
                if d <= 0 or d > max_scope:
                    valid = False
                    break
                if (used_diffs_local >> d) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Also check that the new differences don't conflict with each other
            # (they can't since they're all from the same next_val to different entries)
            
            new_used = used_diffs_local
            for d in new_diffs:
                new_used |= (1 << d)
            
            success, new_used2, new_entries = build_row(
                row_idx, current_entries + [next_val], new_used
            )
            if success:
                return True, new_used2, new_entries
        
        return False, None, None
    
    # Build all rows
    for r in range(n_rows):
        if time.time() > deadline - 5:
            return None
        
        success, used_diffs, entries = build_row(r, [0], used_diffs)
        if not success:
            return None
        rows.append(entries)
    
    return rows
# EVOLVE-BLOCK-END
