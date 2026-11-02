import time
import random
import sys

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 integers
    starting at 0, strictly increasing, with all positive differences distinct
    across all rows. Minimize the largest entry.
    """
    # Total number of differences needed
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound on M: M(M+1)/2 >= total_diffs_needed
    M_lower = int((2 * total_diffs_needed) ** 0.5)
    while M_lower * (M_lower + 1) // 2 < total_diffs_needed:
        M_lower += 1
    
    # Start from a slightly higher M to give some room
    M = M_lower
    
    deadline = time.time() + 115.0
    best_rows = None
    best_M = None
    
    # Use a fixed seed for reproducibility in the first attempt
    random.seed(42)
    
    while time.time() < deadline:
        # Try to construct a valid set with the current M
        rows = _construct_set(M, n_rows, k)
        if rows is not None:
            if best_M is None or M < best_M:
                best_M = M
                best_rows = rows
                # If we hit the lower bound, we're done
                if M == M_lower:
                    return best_rows
                # Try to go lower
                M = max(M_lower, M - 1)
                random.seed()  # New random seed for different exploration
            else:
                # Same M, try again with different randomization
                random.seed()
        else:
            # Failed to construct with current M, increase M
            if M < M_lower + 10:
                M += 1
            else:
                # If M is too high, try a different approach
                # Use greedy construction as fallback
                rows = _greedy_construct(n_rows, k)
                if rows is not None:
                    if best_M is None:
                        best_M = max(row[-1] for row in rows)
                        best_rows = rows
                break
        
        # If we're running low on time, break
        if time.time() > deadline - 5.0:
            break
    
    if best_rows is not None:
        return best_rows
    
    # Final fallback: greedy construction
    return _greedy_construct(n_rows, k)

def _construct_set(M, n_rows, k):
    """
    Try to construct a valid set of n_rows rows, each with k+1 integers
    in [0, M], with all differences distinct.
    Uses randomized backtracking with constraint propagation.
    """
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = _find_row(used_diffs, M, k)
        if row is None:
            return None
        rows.append(row)
        # Update used differences
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    return rows

def _find_row(used_diffs, M, k):
    """
    Find a row of k+1 integers starting at 0, strictly increasing, max <= M,
    such that all differences are not in used_diffs and are distinct within the row.
    Uses randomized backtracking.
    """
    row = [0]
    current_diffs = set()
    
    def dfs():
        if len(row) == k + 1:
            return True
        
        last = row[-1]
        candidates = []
        for x in range(last + 1, M + 1):
            valid = True
            new_diffs = set()
            for a in row:
                d = x - a
                if d in used_diffs or d in current_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                candidates.append((x, new_diffs))
        
        if not candidates:
            return False
        
        random.shuffle(candidates)
        
        for x, new_diffs in candidates:
            row.append(x)
            current_diffs.update(new_diffs)
            
            if dfs():
                return True
            
            current_diffs.difference_update(new_diffs)
            row.pop()
        
        return False
    
    if dfs():
        return row
    return None

def _greedy_construct(n_rows, k):
    """
    Greedy construction: for each row, pick the smallest possible next value
    that doesn't conflict with existing differences.
    """
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
