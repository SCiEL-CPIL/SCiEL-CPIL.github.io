"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a hybrid approach:
    1. Fast greedy to get an initial upper bound
    2. Iterated local search / restarts to try to improve
    3. Deadline guard to return best found before timeout
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # First, get a quick greedy solution as baseline
    best_rows = _greedy_construct(n_rows, k)
    best_max = max(max(row) for row in best_rows)
    
    # Try to improve with various strategies
    # Strategy 1: Greedy with different ordering heuristics
    for _ in range(50):
        if time.time() - start_time > DEADLINE:
            break
        rows = _greedy_construct(n_rows, k, random_seed=random.randint(0, 10**9))
        cur_max = max(max(row) for row in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
    
    # Strategy 2: Try to reduce the maximum by local search
    # We can try to "tighten" rows by shifting elements
    for attempt in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Start from a fresh greedy or slightly perturbed version
        rows = _greedy_construct(n_rows, k, random_seed=random.randint(0, 10**9))
        cur_max = max(max(row) for row in rows)
        
        # Local search: try to reduce the largest element in each row
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            for i in range(n_rows):
                row = rows[i]
                if len(row) == 0:
                    continue
                # Try to reduce the last element (largest) by finding a smaller valid value
                # or rearrange the row to have a smaller max
                new_row = _try_reduce_row_max(row, rows, i, n_rows, k)
                if new_row is not None and max(new_row) < max(row):
                    rows[i] = new_row
                    cur_max = max(max(r) for r in rows)
                    improved = True
                    break  # restart row iteration after change
        
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
    
    # Strategy 3: More aggressive - try to construct with a target max
    # Binary search on the maximum value
    lower = _counting_bound(n_rows, k)
    upper = best_max
    
    while lower < upper and time.time() - start_time < DEADLINE:
        mid = (lower + upper) // 2
        if _can_construct_with_max(n_rows, k, mid):
            upper = mid
        else:
            lower = mid + 1
    
    if upper < best_max and time.time() - start_time < DEADLINE:
        # Try to actually construct with the upper bound
        result = _construct_with_max_constraint(n_rows, k, upper)
        if result is not None:
            best_max = upper
            best_rows = result
    
    return best_rows


def _greedy_construct(n_rows, k, random_seed=None):
    """
    Greedy construction: each row starts at 0 and takes the smallest next mark
    whose differences to the earlier marks of the row are all still unused.
    """
    if random_seed is not None:
        rng = random.Random(random_seed)
    else:
        rng = random.Random()
    
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


def _try_reduce_row_max(row, all_rows, row_idx, n_rows, k):
    """
    Try to find a version of this row with a smaller maximum value.
    This is a simplified local search.
    """
    current_max = max(row)
    # Try to reduce the maximum by 1, 2, etc.
    for new_max in range(current_max - 1, 0, -1):
        if new_max < k:  # can't have max less than k since we need k+1 distinct values starting from 0
            break
        
        # Check if we can form a valid row with max = new_max
        # This is expensive, so only try for small reductions
        if current_max - new_max > 5:
            break
        
        # Try to find a subset of size k+1 from [0, new_max] that includes 0
        # and has all differences distinct and not in used set
        used = set()
        for i, r in enumerate(all_rows):
            if i != row_idx:
                for j in range(len(r)):
                    for l in range(j+1, len(r)):
                        used.add(r[l] - r[j])
        
        # Backtracking to find a valid row
        result = _backtrack_row(0, k, new_max, used)
        if result is not None:
            return result
    
    return None


def _backtrack_row(current, k, max_val, used_diffs):
    """
    Backtrack to find a row of length k+1 starting at 0, with max <= max_val,
    such that all differences are not in used_diffs and are distinct.
    """
    if len(current) == k + 1:
        return current
    
    last = current[-1] if current else 0
    for x in range(last + 1, max_val + 1):
        new_diffs = {x - a for a in current}
        if not new_diffs & used_diffs and len(new_diffs) == len(new_diffs):  # all distinct within new
            # Also check that new_diffs don't have internal duplicates (they can't since x > a)
            result = _backtrack_row(current + [x], k, max_val, used_diffs)
            if result is not None:
                return result
    
    return None


def _counting_bound(n_rows, k):
    """
    Simple counting bound: we need n_rows * (k*(k+1)//2) distinct differences.
    The differences range from 1 to M where M is the max entry.
    So we need M >= n_rows * k*(k+1)//2.
    """
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    return total_diffs


def _can_construct_with_max(n_rows, k, max_val):
    """
    Check if we can construct n_rows rows with max entry <= max_val.
    Uses a quick greedy check.
    """
    # Quick check: is max_val large enough?
    if max_val < k:
        return False
    
    # Try greedy with constraint
    used = set()
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_val:
                return False
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        # If we got here, this row is valid
        pass
    
    return True


def _construct_with_max_constraint(n_rows, k, max_val):
    """
    Try to construct n_rows rows with max entry <= max_val.
    Returns the rows if successful, None otherwise.
    """
    # Multiple attempts with different strategies
    for attempt in range(20):
        used = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if x > max_val:
                    success = False
                    break
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            
            if not success:
                break
            
            rows.append(row)
        
        if success:
            return rows
    
    return None
# EVOLVE-BLOCK-END
