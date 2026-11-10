import time
import random

# Global deadline
DEADLINE = 0
TIME_LIMIT = 110  # 150s limit, keep 40s buffer for safety/overhead

def construct(n_rows, k):
    global DEADLINE
    DEADLINE = time.time() + TIME_LIMIT
    
    # 1. Get a baseline solution using the original greedy method
    # This ensures we always return a valid result if the search fails
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows) if baseline_rows else 0
    
    # 2. Attempt to find a better solution via DFS with increasing M
    # The theoretical lower bound for D(7,5) is 105
    # We search from 105 up to the baseline scope
    for M in range(105, baseline_scope + 1):
        if time.time() > DEADLINE:
            break
        
        # Reset state for this M
        used_diffs = set()
        rows = []
        
        # Try to construct n_rows rows with max element M
        if dfs_construct_rows(rows, used_diffs, n_rows, k, M):
            return rows
            
    # If loop finishes or times out, return the best found (baseline or last valid)
    # Note: The DFS only returns if it finds a complete valid set.
    # So if we didn't return in the loop, we return the baseline.
    return baseline_rows

def greedy_construct(n_rows, k):
    """Original greedy algorithm for baseline."""
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

def dfs_construct_rows(rows, used_diffs, remaining_rows, k, M):
    """
    Recursively constructs 'remaining_rows' rows.
    Returns True if a valid set of rows is found.
    """
    if time.time() > DEADLINE:
        return False
        
    if remaining_rows == 0:
        return True
        
    # Pruning: Check if enough differences remain
    # Each row generates k*(k+1)/2 differences. For k=5, that's 15.
    diffs_needed = remaining_rows * (k * (k + 1) // 2)
    # The total available differences are those in [1, M] not in used_diffs
    # Max possible diffs is M. 
    # So if (M - len(used_diffs)) < diffs_needed, it's impossible.
    if (M - len(used_diffs)) < diffs_needed:
        return False
        
    # Try to build one row
    current_row = [0]
    row_diffs = set()
    if build_row(current_row, used_diffs, row_diffs, k, M):
        # If build_row succeeded, it has updated current_row and used_diffs
        # We need to save the state to backtrack
        # Actually, build_row should handle backtracking internally or we copy state.
        # Let's make build_row return the row if successful, and modify used_diffs.
        # We need to undo the modification if the recursive call fails.
        
        # Save state
        old_used_diffs = set(used_diffs)
        rows.append(current_row)
        
        if dfs_construct_rows(rows, used_diffs, remaining_rows - 1, k, M):
            return True
            
        # Backtrack
        rows.pop()
        used_diffs.clear()
        used_diffs.update(old_used_diffs)
        
    return False

def build_row(current_row, used_diffs, row_diffs, k, M):
    """
    Builds a single row starting with 0, of length k+1, with max element <= M.
    Modifies used_diffs if successful.
    Returns True if successful.
    """
    if time.time() > DEADLINE:
        return False
        
    if len(current_row) == k + 1:
        return True
        
    last_val = current_row[-1]
    # Candidate next values
    # We can try values from last_val + 1 to M
    # Heuristic: Try smaller values first? Or random?
    # For finding *any* solution, smaller is usually more constrained but leads to denser packing.
    # Let's try a mix: sort candidates by how many "conflicts" they cause?
    # Simpler: Just iterate.
    
    # Optimization: If we are at the last step, we can be more specific.
    
    for x in range(last_val + 1, M + 1):
        # Check differences with all previous elements in the row
        new_diffs = {x - a for a in current_row}
        
        # Check if these diffs are already used globally
        if new_diffs & used_diffs:
            continue
            
        # Check if these diffs are distinct from each other (always true for distinct a)
        # But we must ensure they don't collide with *existing* diffs in the *current* row?
        # No, `used_diffs` contains diffs from *previous* rows.
        # What about diffs within the *current* row?
        # The set `new_diffs` contains diffs between x and all previous elements in the row.
        # Are there collisions within `new_diffs`? No, because a are distinct.
        # Are there collisions between `new_diffs` and diffs already in the *current* row?
        # The current row's diffs are NOT in `used_diffs` yet.
        # So we must check if `new_diffs` intersects with the set of diffs already generated by `current_row`.
        
        # Let's compute existing diffs in current_row
        # Actually, it's easier to just check if any diff in `new_diffs` is in the set of all diffs of current_row.
        # But we don't store that set explicitly.
        # However, if a diff $d$ is in `new_diffs`, it means $x - a_i = d$.
        # If there was a previous pair $(a_j, a_l)$ in current_row such that $a_l - a_j = d$,
        # then we have a collision within the row.
        
        # Efficient check:
        # We can maintain a set of diffs for the current row as we build it?
        # Let's pass `row_diffs` as a parameter to build_row?
        # Or just recompute? k=5, so max 15 diffs. Recomputing is cheap.
        
        existing_row_diffs = set()
        for i in range(len(current_row)):
            for j in range(i+1, len(current_row)):
                existing_row_diffs.add(current_row[j] - current_row[i])
                
        if new_diffs & existing_row_diffs:
            continue
            
        # If we pass all checks, tentatively add x
        current_row.append(x)
        
        # Update used_diffs? 
        # Wait, `used_diffs` is global. We shouldn't add to it until the row is complete?
        # No, if we add to it, we must backtrack.
        # It's safer to only commit to `used_diffs` when the row is complete.
        # But then we might pick x that collides with a diff that *would* be generated by a future element in the same row?
        # No, the check `new_diffs & existing_row_diffs` handles collisions with *past* elements.
        # Collisions with *future* elements will be handled when those future elements are picked.
        # So we do NOT need to update `used_diffs` during row construction.
        # We only need to check against `used_diffs` (global) and `existing_row_diffs` (local).
        
        if build_row(current_row, used_diffs, row_diffs, k, M):
            # Success: The row is complete.
            # Now we commit the diffs of this entire row to `used_diffs`
            # We can compute them from current_row
            for i in range(len(current_row)):
                for j in range(i+1, len(current_row)):
                    used_diffs.add(current_row[j] - current_row[i])
            return True
        
        # Backtrack
        current_row.pop()
        
    return False
