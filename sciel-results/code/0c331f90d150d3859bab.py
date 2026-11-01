import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 140.0  # 10s safety margin for 150s limit
    
    # The number of differences in one row is k * (k+1) / 2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Get initial upper bound from greedy
    initial_rows = _greedy_construct(n_rows, k)
    best_rows = initial_rows
    best_M = _max_entry(initial_rows)
    
    # Lower bound: need total_diffs_needed distinct differences, each <= M
    # So M >= total_diffs_needed
    M_lower = max(total_diffs_needed, k)
    
    # Try increasing M from lower bound to best_M - 1
    for M in range(M_lower, best_M):
        if time.time() > deadline - 5:
            break
        
        # Try to find a packing with max entry <= M
        rows = _solve_packing(n_rows, k, M, deadline)
        
        if rows is not None:
            best_rows = rows
            best_M = M
            break
    
    # If we still have time, try to improve further with local search
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        improved = _local_search_improve(best_rows, k, deadline)
        if improved is not None:
            imp_M = _max_entry(improved)
            if imp_M < best_M:
                best_rows = improved
                best_M = imp_M
    
    return best_rows

def _solve_packing(n_rows, k, M, deadline):
    """
    Try to find n_rows rows of k+1 integers from {0, ..., M} starting at 0,
    such that all pairwise differences are distinct.
    Uses randomized greedy with backtracking.
    """
    used_diffs = set()
    rows = []
    
    def try_greedy_row(used_diffs, M, k, random_bias=0.5):
        """Try to construct a single valid row using greedy approach."""
        row = [0]
        row_diffs = set()
        
        # Try multiple random starts
        for attempt in range(50):
            row = [0]
            row_diffs = set()
            
            while len(row) < k + 1:
                # Determine range for next value
                # Must be > last element
                min_next = row[-1] + 1
                max_next = M
                
                if min_next > max_next:
                    break
                    
                # Try values in a biased order
                # Sometimes prefer smaller, sometimes larger
                if attempt % 2 == 0:
                    candidates = range(min_next, max_next + 1)
                else:
                    candidates = range(max_next, min_next - 1, -1)
                
                found = False
                for x in candidates:
                    # Check if adding x creates conflicts
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d <= 0:
                            valid = False
                            break
                        if d in row_diffs or d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        row.append(x)
                        row_diffs.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    break
            
            if len(row) == k + 1:
                return row, row_diffs
        
        return None, None
    
    # Try multiple restarts
    for restart in range(30):
        if time.time() > deadline - 2:
            break
            
        random.seed(restart * 7919 + M)  # Prime multiplier for diversity
        used_diffs.clear()
        rows.clear()
        
        success = True
        for i in range(n_rows):
            row, row_diffs = try_greedy_row(used_diffs, M, k)
            
            if row is None:
                success = False
                break
            
            rows.append(row)
            used_diffs.update(row_diffs)
        
        if success:
            return rows
    
    return None

def _local_search_improve(rows, k, deadline):
    """
    Try to reduce the maximum entry by local search.
    """
    if rows is None:
        return None
    
    current_rows = [row[:] for row in rows]
    current_M = _max_entry(current_rows)
    best_rows = [row[:] for row in current_rows]
    best_M = current_M
    
    # Try to reduce M by swapping or modifying rows
    for M_target in range(current_M - 1, max(_greedy_lower_bound(n_rows, k), 0), -1):
        if time.time() > deadline - 3:
            break
        
        # Try to reconstruct all rows with max entry <= M_target
        new_rows = _solve_packing(len(rows), k, M_target, deadline)
        if new_rows is not None:
            best_rows = new_rows
            best_M = M_target
            break
    
    return best_rows if best_M < current_M else None

def _greedy_lower_bound(n_rows, k):
    """Simple lower bound on M."""
    diffs_per_row = k * (k + 1) // 2
    return max(n_rows * diffs_per_row, k)

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False
            
        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True
