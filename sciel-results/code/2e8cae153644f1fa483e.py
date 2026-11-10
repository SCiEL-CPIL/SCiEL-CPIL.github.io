import time
import random

# Global deadline
DEADLINE = 0
TIME_LIMIT = 115  # 150s limit, keep 35s buffer for safety/overhead

def construct(n_rows, k):
    global DEADLINE
    DEADLINE = time.time() + TIME_LIMIT
    
    # 1. Get a baseline solution using a fast greedy method
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows) if baseline_rows else 0
    
    # 2. Attempt to find a better solution via DFS with increasing M
    # The theoretical lower bound for D(7,5) is 105
    # We search from 105 up to the baseline scope
    best_found = baseline_rows
    best_scope = baseline_scope
    
    for M in range(105, baseline_scope + 1):
        if time.time() > DEADLINE:
            break
        
        # Reset state for this M
        used_diffs = set()
        rows = []
        
        # Try to construct n_rows rows with max element M
        if dfs_construct_rows(rows, used_diffs, n_rows, k, M):
            return rows
            
    # If loop finishes or times out, return the best found (baseline)
    return best_found

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
    row_diffs = set()  # Track differences within the current row
    
    if build_row(current_row, used_diffs, k, M, row_diffs):
        # If build_row succeeded, it has updated current_row and row_diffs
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

def build_row(current_row, used_diffs, k, M, row_diffs):
    """
    Builds a single row starting with 0, of length k+1, with max element <= M.
    Modifies used_diffs if successful.
    Returns True if successful.
    """
    if time.time() > DEADLINE:
        return False
        
    if len(current_row) == k + 1:
        # Row is complete, commit its differences to used_diffs
        used_diffs.update(row_diffs)
        return True
        
    last_val = current_row[-1]
    # Candidate next values
    # We can try values from last_val + 1 to M
    # Heuristic: Try smaller values first? Or random?
    # For finding *any* solution, smaller is usually more constrained but leads to denser packing.
    # Let's try a mix: sort candidates by how many "conflicts" they cause?
    # Simpler: Just iterate.
    
    # Optimization: If we are at the last step, we can be more specific.
    
    # Collect valid candidates
    candidates = []
    
    for x in range(last_val + 1, M + 1):
        # Check differences with all previous elements in the row
        new_diffs = {x - a for a in current_row}
        
        # Check if these diffs are already used globally
        if new_diffs & used_diffs:
            continue
            
        # Check if these diffs collide with existing diffs in the current row
        if new_diffs & row_diffs:
            continue
            
        # Heuristic: prefer values that leave more room for remaining positions
        # and that use "smaller" differences (to save larger ones for flexibility)
        # Score: prefer smaller x (tighter packing), but also consider future feasibility
        candidates.append((x, new_diffs))
    
    # Sort candidates: try smaller values first (denser packing tends to work)
    # But also try a few larger ones to avoid getting stuck
    # Use a mixed strategy: sort by x, but with some randomness in ordering
    candidates.sort(key=lambda c: c[0])
    
    # Limit candidates to avoid excessive branching
    # For each position, try at most ~30 candidates
    max_candidates = 30
    if len(candidates) > max_candidates:
        # Keep the smallest, but also sample some from the rest
        kept = candidates[:15]
        rest = candidates[15:]
        # Sample some from the rest for diversity
        if rest:
            sample_size = min(15, len(rest))
            kept.extend(random.sample(rest, sample_size))
        candidates = kept
    
    for x, new_diffs in candidates:
        # Tentatively add x
        current_row.append(x)
        row_diffs.update(new_diffs)
        
        if build_row(current_row, used_diffs, k, M, row_diffs):
            # Success: The row is complete.
            # Now we commit the diffs of this entire row to `used_diffs`
            used_diffs.update(row_diffs)
            return True
        
        # Backtrack
        current_row.pop()
        row_diffs.difference_update(new_diffs)
    
    return False
