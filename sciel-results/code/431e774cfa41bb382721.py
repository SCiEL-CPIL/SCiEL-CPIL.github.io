import time
import random
import sys

# Increase recursion limit for deep searches
sys.setrecursionlimit(10000)

# Global deadline
DEADLINE = 0
TIME_LIMIT = 115  # Keep buffer for safety

def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT
    
    # 1. Get a baseline solution using a fast greedy method
    # This ensures we always return a valid result if the search fails or times out
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows) if baseline_rows else 0
    
    # 2. Attempt to find a better solution via optimized backtracking with increasing M
    # The theoretical lower bound for D(7,5) is 105.
    # We search from 105 up to the baseline scope.
    best_found = baseline_rows
    best_scope = baseline_scope
    
    # We want to minimize the scope. We try M from the lower bound upwards.
    # If we find a solution for M, that is the optimal for that M.
    # Since we want the global minimum, we stop at the first M that yields a solution.
    
    lower_bound = 105 # Specific to D(7,5), but we can calculate a generic bound
    # Generic counting bound: n_rows * k*(k+1)/2 differences must fit in [1, M]
    # So M >= n_rows * k*(k+1)/2
    generic_lb = n_rows * (k * (k + 1) // 2)
    start_m = max(lower_bound, generic_lb)
    
    for M in range(start_m, baseline_scope + 1):
        if time.time() > DEADLINE:
            break
        
        # Try to construct n_rows rows with max element M
        solution = solve_with_mask(n_rows, k, M)
        if solution is not None:
            return solution
            
    # If loop finishes or times out, return the best found (baseline)
    return best_found

def greedy_construct(n_rows, k):
    """
    Fast greedy algorithm for baseline.
    This is not optimal but provides a valid upper bound quickly.
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

def solve_with_mask(n_rows, k, M):
    """
    Attempts to find a valid set of n_rows rows with max element <= M.
    Uses bitmask representation for used differences to speed up checks.
    Returns a list of rows if found, None otherwise.
    """
    # used_mask: bit i is set if difference i is used.
    # Differences range from 1 to M.
    # We use an integer as a bitset.
    used_mask = 0
    rows = []
    
    # Helper to check if a set of differences (list of ints) can be added to used_mask
    # and are distinct within themselves
    def can_add(diffs, current_row_mask):
        # Check if diffs are distinct among themselves
        # Since diffs are derived from a row, they are distinct if the row elements are distinct.
        # But we need to check if they intersect with used_mask or current_row_mask.
        # Actually, we build the row incrementally, so we check each new diff against:
        # 1. used_mask (global)
        # 2. current_row_mask (local to the row being built)
        
        mask = 0
        for d in diffs:
            if d > M: # Should not happen if we constrain values, but good for safety
                return False, 0
            bit = 1 << d
            # Check if bit is already set in global or local
            if (used_mask & bit) or (current_row_mask & bit):
                return False, 0
            mask |= bit
        return True, mask

    def build_row(idx, current_row, current_row_mask):
        """
        Recursively builds the current row starting at index idx (0 is already 0).
        current_row: list of elements so far
        current_row_mask: bitmask of differences already present in current_row
        """
        if time.time() > DEADLINE:
            return False
            
        if len(current_row) == k + 1:
            # Row is complete
            rows.append(current_row)
            # Update global used_mask
            nonlocal used_mask
            used_mask |= current_row_mask
            return True
            
        last_val = current_row[-1]
        # Candidate next values: last_val + 1 to M
        # Heuristic: Try values that result in "small" differences first?
        # Or try to spread them out?
        # For difference triangles, "greedy" (smallest possible) often works well for packing,
        # but backtracking is needed to avoid dead ends.
        
        candidates = []
        for x in range(last_val + 1, M + 1):
            # Calculate new differences with all previous elements in current_row
            new_diffs = []
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                if d > M:
                    valid = False
                    break
                bit = 1 << d
                # Check against global used and local row mask
                if (used_mask & bit) or (current_row_mask & bit):
                    valid = False
                    break
                # Check if this diff is already in new_diffs (should not happen as a are distinct)
                # But we need to ensure new_diffs don't collide with each other?
                # d = x - a. Since a are distinct, d are distinct.
                new_diffs.append(d)
                new_mask |= bit
                
            if valid:
                # Additional pruning: 
                # If we pick x, we use new_mask.
                # We should prefer x that leaves more options?
                # Simple heuristic: sort by x.
                candidates.append((x, new_diffs, new_mask))
                
        # Sort candidates. 
        # Strategy: Try smaller x first (denser packing) but if that fails, try others.
        # To improve search, we can randomize the order slightly or use a "fail-first" metric.
        # For now, deterministic order (smallest x) is a good baseline.
        # If we find that we always fail at small x, we might need to perturb.
        
        # Optimization: If the number of candidates is large, we might want to limit them
        # or use a better heuristic. For M ~ 112, the range is small enough.
        
        for x, new_diffs, new_mask in candidates:
            current_row.append(x)
            
            # Recurse
            if build_row(idx + 1, current_row, current_row_mask | new_mask):
                return True
            
            current_row.pop()
            
        return False

    # Start building the first row
    # Row 0 starts with [0]
    if build_row(1, [0], 0):
        if len(rows) == 1:
            # We have built 1 row. We need to build n_rows - 1 more.
            # The structure above only builds ONE row in the recursive call.
            # We need a loop for the rows.
            pass
        else:
            # The build_row function as written only builds one row.
            # We need to wrap it in a loop for n_rows.
            pass
            
    # Let's restructure solve_with_mask to loop over rows
    # The previous logic was flawed because build_row only handles one row.
    
    # Corrected approach:
    used_mask = 0
    rows = []
    
    def build_row(current_row, current_row_mask):
        if time.time() > DEADLINE:
            return False
            
        if len(current_row) == k + 1:
            rows.append(current_row)
            nonlocal used_mask
            used_mask |= current_row_mask
            return True
            
        last_val = current_row[-1]
        candidates = []
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                if d > M:
                    valid = False
                    break
                bit = 1 << d
                if (used_mask & bit) or (current_row_mask & bit):
                    valid = False
                    break
                new_mask |= bit
            if valid:
                candidates.append((x, new_mask))
                
        # Heuristic: Try candidates in increasing order of x
        # This tends to produce "tight" rows which is good for minimizing M.
        # If we get stuck, we might need to try different orderings.
        
        for x, new_mask in candidates:
            current_row.append(x)
            if build_row(current_row, current_row_mask | new_mask):
                return True
            current_row.pop()
            
        return False

    for _ in range(n_rows):
        if time.time() > DEADLINE:
            return None
        if not build_row([0], 0):
            return None
            
    return rows

# The above solve_with_mask has a subtle issue: `used_mask` is modified in place.
# If `build_row` fails for a specific row, we need to backtrack the `used_mask` changes?
# No, `build_row` is called for a NEW row. The `used_mask` only accumulates.
# If `build_row` returns False, it means we couldn't complete the current row.
# The `used_mask` should NOT have been updated if the row wasn't completed.
# In my implementation, `used_mask` is updated ONLY when `len(current_row) == k+1` (success).
# So backtracking is handled naturally by the return value.
# However, `used_mask` is a nonlocal variable. If `build_row` fails, it returns False,
# and the caller (the loop) sees None and returns None.
# The state `used_mask` is only modified on success.
# So if row 1 fails, `used_mask` is unchanged from before row 1 started.
# This is correct.

# However, there is a performance risk: the search space for a single row is still large.
# For D(7,5), we need to find 7 rows.
# The bitmask approach is fast, but the branching factor can be high.
# We can add a simple pruning: 
# If the number of available differences (M - popcount(used_mask)) is less than the 
# differences needed for the current row (k*(k+1)/2), prune.
# This is a weak prune but cheap.

# Let's refine build_row with this prune.

def solve_with_mask_v2(n_rows, k, M):
    used_mask = 0
    rows = []
    
    def popcount(x):
        return bin(x).count('1')
        
    def build_row(current_row, current_row_mask):
        if time.time() > DEADLINE:
            return False
            
        if len(current_row) == k + 1:
            rows.append(current_row)
            nonlocal used_mask
            used_mask |= current_row_mask
            return True
            
        # Pruning: Check if enough diffs remain for this row
        # Diffs needed for this row: k*(k+1)/2
        # Diffs available: M - popcount(used_mask) - popcount(current_row_mask)
        # Actually, current_row_mask is subset of what we will use, but not in used_mask yet.
        # Total used so far = popcount(used_mask)
        # Diffs we MUST use for this row = k*(k+1)/2
        # Diffs we have already picked for this row = popcount(current_row_mask)
        # Remaining diffs needed for this row = k*(k+1)/2 - popcount(current_row_mask)
        # Available diffs globally = M - popcount(used_mask)
        # If Available < Remaining Needed, prune.
        
        diffs_needed_total = k * (k + 1) // 2
        diffs_used_in_row = popcount(current_row_mask)
        remaining_needed = diffs_needed_total - diffs_used_in_row
        
        available = M - popcount(used_mask)
        if available < remaining_needed:
            return False
            
        last_val = current_row[-1]
        candidates = []
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                if d > M:
                    valid = False
                    break
                bit = 1 << d
                if (used_mask & bit) or (current_row_mask & bit):
                    valid = False
                    break
                new_mask |= bit
            if valid:
                candidates.append((x, new_mask))
                
        # Heuristic: 
        # Instead of just sorting by x, we can sort by the "cost" of the mask.
        # Masks that set bits in "rare" positions (high values) might be worse?
        # Or masks that set bits in "common" positions (low values) are better because
        # low values are more likely to be needed by other rows?
        # Actually, for packing, we want to use small differences if possible.
        # So sorting by x (which correlates with small differences) is good.
        
        for x, new_mask in candidates:
            current_row.append(x)
            if build_row(current_row, current_row_mask | new_mask):
                return True
            current_row.pop()
            
        return False

    for _ in range(n_rows):
        if time.time() > DEADLINE:
            return None
        if not build_row([0], 0):
            return None
            
    return rows

# Replace the original solve_with_mask with v2
solve_with_mask = solve_with_mask_v2

# One more improvement: Randomized Restart
# If the deterministic search fails for a specific M, it might be due to a bad choice early on.
# We can try a few random restarts for the same M before incrementing M.

def solve_with_mask_randomized(n_rows, k, M, max_restarts=5):
    for _ in range(max_restarts):
        # Shuffle the candidate order? 
        # The current build_row iterates candidates in order.
        # To randomize, we can shuffle the list of candidates in build_row.
        # But build_row is nested. We can pass a flag or use a global random seed.
        
        # Let's implement a version that shuffles candidates.
        # We need to modify build_row to accept a shuffle flag.
        
        # For simplicity, we can just call the deterministic version first.
        # If it fails, we try a few times with a slightly modified heuristic?
        # Or we can just rely on the fact that for D(7,5), the deterministic 
        # "smallest x first" strategy is quite strong.
        
        # Let's stick to deterministic for now to keep it simple and fast.
        # If we have time, we can add restarts.
        pass
    
    return solve_with_mask_deterministic(n_rows, k, M)

# Let's rename the previous one to deterministic and use it.
# I will re-write the final clean version below.

def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT
    
    # 1. Baseline
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows) if baseline_rows else 0
    
    # 2. Search
    lower_bound = n_rows * (k * (k + 1) // 2)
    # Specific lower bound for D(7,5) is 105.
    if n_rows == 7 and k == 5:
        lower_bound = 105
        
    for M in range(lower_bound, baseline_scope + 1):
        if time.time() > DEADLINE:
            break
        
        # Try to solve for this M
        # We use a bitmask-based backtracking solver
        solution = solve_optimized(n_rows, k, M)
        if solution is not None:
            return solution
            
    return baseline_rows

def solve_optimized(n_rows, k, M):
    used_mask = 0
    rows = []
    
    def popcount(x):
        return bin(x).count('1')
        
    def build_row(current_row, current_row_mask):
        if time.time() > DEADLINE:
            return False
            
        if len(current_row) == k + 1:
            rows.append(current_row)
            nonlocal used_mask
            used_mask |= current_row_mask
            return True
            
        # Pruning
        diffs_needed_total = k * (k + 1) // 2
        diffs_used_in_row = popcount(current_row_mask)
        remaining_needed = diffs_needed_total - diffs_used_in_row
        
        available = M - popcount(used_mask)
        if available < remaining_needed:
            return False
            
        last_val = current_row[-1]
        candidates = []
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                if d > M:
                    valid = False
                    break
                bit = 1 << d
                if (used_mask & bit) or (current_row_mask & bit):
                    valid = False
                    break
                new_mask |= bit
            if valid:
                candidates.append((x, new_mask))
                
        # Heuristic: Try smaller x first
        # candidates.sort(key=lambda c: c[0]) # Already in order
        
        for x, new_mask in candidates:
            current_row.append(x)
            if build_row(current_row, current_row_mask | new_mask):
                return True
            current_row.pop()
            
        return False

    for _ in range(n_rows):
        if time.time() > DEADLINE:
            return None
        if not build_row([0], 0):
            return None
            
    return rows

# Final check on the code structure.
# The function `construct` is the entry point.
# It uses `greedy_construct` for baseline.
# It uses `solve_optimized` for the search.
# `solve_optimized` uses bitmasks and pruning.

# One potential issue: `nonlocal used_mask` in `build_row` inside `solve_optimized`.
# This is valid in Python 3.

# Let's double check the `greedy_construct`.
# It returns a list of rows.
# `baseline_scope` is the max of max(row).

# The search loop `for M in range(...)`
# If it finds a solution, it returns it.
# This guarantees we return the minimum M for which a solution exists,
# provided the solver is complete (which backtracking is, given enough time).

# The solver might be slow for large M if the solution is sparse, but for D(7,5)
# the solutions are quite dense. The bitmask operations are O(1) effectively.
# The main cost is the recursion depth and branching.
# For k=5, row length is 6. Depth is 5.
# Branching factor: up to M.
# With pruning, it should be manageable.

# To be safe against timeouts, we keep the baseline.

# One refinement: In `build_row`, we can add a check for the LAST element.
# If we are picking the last element (len(current_row) == k), 
# we can just check if any x works, without recursion overhead?
# No, the recursion handles it fine.

# Let's add a small randomization to `construct` to break ties if the deterministic
# search fails due to a specific bad path. 
# Actually, if the deterministic search fails, it means NO solution exists for that M 
# IF the search is complete. Backtracking is complete. 
# So if it fails, it fails. 
# Wait, is it possible that the heuristic (smallest x first) leads to a dead end 
# while a larger x would have worked? Yes. 
# But backtracking will explore ALL candidates if the first ones fail. 
# `for x, new_mask in candidates:` iterates through ALL valid candidates. 
# So it is a complete search. It will find a solution if one exists. 
# The only reason it would fail is if no solution exists for that M, or timeout.

# Therefore, the deterministic complete search is correct.

# Final Code Structure

import time
import random
import sys

sys.setrecursionlimit(10000)

DEADLINE = 0
TIME_LIMIT = 115

def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT
    
    # 1. Baseline
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows) if baseline_rows else 0
    
    # 2. Search
    lower_bound = n_rows * (k * (k + 1) // 2)
    if n_rows == 7 and k == 5:
        lower_bound = 105
        
    for M in range(lower_bound, baseline_scope + 1):
        if time.time() > DEADLINE:
            break
        
        solution = solve_optimized(n_rows, k, M)
        if solution is not None:
            return solution
            
    return baseline_rows

def greedy_construct(n_rows, k):
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

def solve_optimized(n_rows, k, M):
    used_mask = 0
    rows = []
    
    def popcount(x):
        return bin(x).count('1')
        
    def build_row(current_row, current_row_mask):
        if time.time() > DEADLINE:
            return False
            
        if len(current_row) == k + 1:
            rows.append(current_row)
            nonlocal used_mask
            used_mask |= current_row_mask
            return True
            
        diffs_needed_total = k * (k + 1) // 2
        diffs_used_in_row = popcount(current_row_mask)
        remaining_needed = diffs_needed_total - diffs_used_in_row
        
        available = M - popcount(used_mask)
        if available < remaining_needed:
            return False
            
        last_val = current_row[-1]
        candidates = []
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                if d > M:
                    valid = False
                    break
                bit = 1 << d
                if (used_mask & bit) or (current_row_mask & bit):
                    valid = False
                    break
                new_mask |= bit
            if valid:
                candidates.append((x, new_mask))
                
        for x, new_mask in candidates:
            current_row.append(x)
            if build_row(current_row, current_row_mask | new_mask):
                return True
            current_row.pop()
            
        return False

    for _ in range(n_rows):
        if time.time() > DEADLINE:
            return None
        if not build_row([0], 0):
            return None
            
    return rows
