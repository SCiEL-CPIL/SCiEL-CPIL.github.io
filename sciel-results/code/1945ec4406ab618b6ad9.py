import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. Bitset-based local search for scope reduction
    best_solution = _local_search_bitset(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    # 3. Randomized restarts with bitset backtracking
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline - 3.0:
        attempt += 1
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
        
        # Try to find a solution with scope <= target_scope
        solution = _backtracking_bitset(n_rows, k, target_scope, deadline, rng)
        if solution:
            new_scope = max(max(row) for row in solution)
            if new_scope < best_scope:
                best_scope = new_scope
                best_solution = solution
                # Polish with local search
                improved = _local_search_bitset(best_solution, n_rows, k, deadline)
                new_scope = max(max(row) for row in improved)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = improved
                if best_scope <= lower_bound:
                    return best_solution
    
    return best_solution

def greedy_construct(n_rows, k):
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

def _make_bitset(max_scope):
    """Create a bitset to represent used differences up to max_scope."""
    # We need bits 1..max_scope to represent differences
    # Use Python's arbitrary precision integers as bitsets
    return 0  # 0 means no differences used

def _diff_bit(d):
    """Return the bit mask for difference d."""
    return 1 << d

def _check_diffs_bitset(diffs, used_bitset):
    """Check if any difference in diffs conflicts with used_bitset."""
    mask = 0
    for d in diffs:
        mask |= (1 << d)
    return (mask & used_bitset) != 0

def _local_search_bitset(solution, n_rows, k, deadline):
    """
    Aggressive local search using bitset representation.
    Tries to reduce scope by moving elements down.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    
    # Build used differences bitset
    used_bitset = 0
    for row in best:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_bitset |= (1 << (row[j] - row[i]))
    
    rng = random.Random(12345)
    
    # Iterative improvement: try to move each element down
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        
        # Try each row, each position (from end to start for better packing)
        for row_idx in range(n_rows):
            if time.time() > deadline - 1.0:
                break
            row = best[row_idx]
            for pos in range(len(row) - 1, 0, -1):
                original = row[pos]
                # Try moving down to smallest feasible value
                for new_val in range(row[pos - 1] + 1, original):
                    # Compute new differences involving position pos
                    new_diffs_mask = 0
                    for i in range(pos):
                        new_diffs_mask |= (1 << (new_val - row[i]))
                    for j in range(pos + 1, len(row)):
                        new_diffs_mask |= (1 << (row[j] - new_val))
                    
                    # Compute old differences involving position pos
                    old_diffs_mask = 0
                    for i in range(pos):
                        old_diffs_mask |= (1 << (original - row[i]))
                    for j in range(pos + 1, len(row)):
                        old_diffs_mask |= (1 << (row[j] - original))
                    
                    # Check for conflicts: new_diffs should not conflict with (used - old)
                    conflict = new_diffs_mask & (used_bitset & ~old_diffs_mask)
                    if conflict:
                        continue
                    
                    # Also check new_diffs don't have internal conflicts (shouldn't happen if row is valid)
                    # Apply the change
                    used_bitset = (used_bitset & ~old_diffs_mask) | new_diffs_mask
                    row[pos] = new_val
                    improved = True
                    break
    
    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]

def _backtracking_bitset(n_rows, k, max_scope, deadline, rng):
    """
    Backtracking search with bitset for fast conflict checking.
    Builds rows one at a time, choosing values to keep scope small.
    """
    used_bitset = 0
    rows = []
    
    def build_row(row_idx):
        if time.time() > deadline:
            return None
        
        row = [0]
        used_local = 0  # bitset for differences in this row
        
        for pos in range(1, k + 1):
            if time.time() > deadline:
                return None
            
            # Find valid candidates
            valid = []
            for x in range(row[-1] + 1, max_scope + 1):
                # Check differences with all previous elements in this row
                row_diffs_mask = 0
                for a in row:
                    row_diffs_mask |= (1 << (x - a))
                
                # Check conflicts with global used and row-local used
                if (row_diffs_mask & used_bitset) or (row_diffs_mask & used_local):
                    continue
                
                # Ensure we can complete the row: need (k+1 - pos - 1) more elements
                # Each needs at least +1, so x + (k+1-pos-1) <= max_scope
                if x + (k + 1 - pos - 1) > max_scope:
                    continue
                
                valid.append(x)
            
            if not valid:
                return None
            
            # Strategy: prefer smaller values but add some randomness
            # Take a mix of smallest and random
            n_smallest = min(5, len(valid))
            smallest = valid[:n_smallest]
            
            if len(valid) > n_smallest and rng.random() < 0.4:
                extra_pool = valid[n_smallest:]
                extra = rng.sample(extra_pool, min(3, len(extra_pool)))
                candidates = smallest + extra
            else:
                candidates = smallest
            
            rng.shuffle(candidates)
            
            found = False
            for x in candidates:
                row_diffs_mask = 0
                for a in row:
                    row_diffs_mask |= (1 << (x - a))
                
                # Check again (in case of duplicates in candidates)
                if (row_diffs_mask & used_bitset) or (row_diffs_mask & used_local):
                    continue
                
                # Tentatively add
                row.append(x)
                used_local |= row_diffs_mask
                
                if pos == k:
                    # Row complete
                    return list(row)
            
            return None
    
    for i in range(n_rows):
        if time.time() > deadline:
            return None
        
        row = build_row(i)
        if row is None:
            return None
        
        rows.append(row)
        # Update global used bitset
        for a in row:
            for b in row:
                if b > a:
                    used_bitset |= (1 << (b - a))
    
    return rows
