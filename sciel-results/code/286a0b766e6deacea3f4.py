import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a difference triangle set (n_rows, k) minimizing the largest entry.
    Uses a backtracking search with forward checking, iterating over possible
    maximum entries M from the lower bound upwards.
    """
    deadline = time.time() + 145.0  # Safety margin
    
    # 1. Generate a valid baseline using a fast greedy method
    # This ensures we always have a valid result to return.
    baseline = _greedy_construct(n_rows, k)
    best_rows = baseline
    best_score = _max_entry(baseline)
    
    # For the specific task (7, 5), the lower bound is 105.
    # We can start searching from a value slightly below the baseline.
    # The counting bound for m(7,5) is 105.
    # Let's start from 105 up to best_score - 1.
    
    # If the problem is trivial, return immediately
    if n_rows <= 1 or k <= 1:
        return baseline
    
    # To handle general cases, we estimate a lower bound.
    # Total differences needed: n_rows * (k * (k+1) / 2)
    # Max difference is M. So M must be at least total_diffs.
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Heuristic lower bound: We need at least total_diffs distinct values in [1, M].
    # So M >= total_diffs.
    # Also, the row itself has k+1 elements, so M >= k.
    lb = max(total_diffs, k)
    
    # We search M from lb to best_score - 1
    # If we find a solution for a smaller M, we update best.
    
    # Optimization: If the gap is small, just try values.
    # If the gap is large, we might miss the optimal if we only go up to best_score-1,
    # but since we have a valid baseline, we just want to improve it.
    
    # To be safe and potentially find the optimal (112 or lower), we start from lb.
    # However, if lb is much smaller than the true optimal, we waste time.
    # For (7,5), lb=105, optimal=112. Gap is 7. This is small.
    
    # For larger instances, we might want to start closer to the baseline.
    # Let's start from max(lb, best_score - 20) to balance risk and time.
    # Actually, for (7,5), best_score from greedy is likely ~120-130.
    # Let's start from lb if lb < best_score.
    
    start_m = lb
    end_m = best_score - 1
    
    # If the range is too large, we might not finish.
    # But for (7,5), it's small.
    
    for M in range(start_m, end_m + 1):
        if time.time() > deadline - 5:
            break
        
        # Try to find a valid configuration with max entry <= M
        # We use backtracking.
        # To make it faster, we can randomize the row order or mark selection slightly?
        # Backtracking is deterministic if variable ordering is fixed.
        # Let's use a fixed variable ordering: Rows 0..n_rows-1, Marks 0..k.
        # But we can randomize the initial state or use multiple restarts with different random seeds for tie-breaking?
        # Actually, pure backtracking with good pruning is deterministic.
        # If it fails for M, it fails.
        # If it succeeds, we found a solution.
        
        # To avoid getting stuck in one specific bad path if the problem is harder,
        # we could add a bit of randomization in the order we try candidates for the first few rows?
        # No, let's keep it simple first. If it's too slow, we'll see.
        
        # The search space for M=112 is small enough for a well-pruned backtracker.
        
        solution = _backtrack_solve(n_rows, k, M)
        
        if solution is not None:
            # Valid solution found with max entry <= M
            # Since we iterate M upwards, the first one found is the minimum M for this specific search path?
            # No, backtracking finds A solution. It doesn't guarantee minimality of max entry within the M bound,
            # but since we fix M as the upper bound for all entries, any solution found has max entry <= M.
            # We want the smallest M. So if we find one for M, we can stop and return it.
            # Wait, the solution might have max entry < M. That's fine, it's still <= M.
            # But we want to minimize the max entry.
            # If we find a solution for M=105, the max entry is <= 105.
            # If we find one for M=106, max entry <= 106.
            # So the first M that yields a solution gives the best possible max entry (which is <= M).
            # Actually, the max entry of the found solution might be less than M.
            # We should record the actual max entry of the found solution.
            
            actual_max = _max_entry(solution)
            if actual_max < best_score:
                best_score = actual_max
                best_rows = solution
                # Since we are iterating M upwards, if we found a solution for M,
                # could there be a solution for M-1 with a smaller max entry?
                # No, we already checked M-1 and it failed (returned None).
                # So this is the minimal M for which a solution exists.
                # The actual max entry of the solution is the best we can get for this M.
                # We can stop here because any larger M will yield max entry >= current M > previous M.
                # Wait, if we found a solution for M=112 with max entry 110, and M=111 failed,
                # then 110 is the best possible?
                # Yes, because any solution with max entry <= 109 would have been found when M=109 (or lower).
                # Since M=109..111 failed, no solution exists with max entry <= 111.
                # So the best possible max entry is at least 112.
                # We found one with max entry 110? That's a contradiction.
                # If a solution exists with max entry 110, it is also a valid solution for M=110.
                # So _backtrack_solve(n_rows, k, 110) should have found it.
                # Therefore, if M=111 fails, no solution with max entry <= 111 exists.
                # So the first successful M gives the true optimal max entry?
                # Yes, the optimal max entry is the smallest M such that a solution exists in [0, M].
                # So we can stop at the first successful M.
                break
        
    return best_rows

def _backtrack_solve(n_rows, k, M):
    """
    Backtracking solver to find n_rows rows of k+1 integers in [0, M]
    such that all differences are distinct.
    """
    # Precompute all valid differences for a given M?
    # Total differences available: M
    # Differences needed: n_rows * k * (k+1) / 2
    
    # State:
    # used_diffs: set of used differences
    # rows: list of current rows (list of lists)
    # current_row_idx: index of the row currently being constructed
    # current_mark_idx: index of the mark in the current row
    
    # To speed up, we can precompute candidate marks for each position?
    # Or just iterate.
    
    # Variable ordering:
    # We construct rows one by one.
    # Within a row, we pick marks in increasing order.
    
    # Heuristic:
    # Try to pack tightly.
    
    def backtrack(used_diffs, rows, row_idx, mark_idx, current_row):
        if row_idx == n_rows:
            return rows[:]  # Found a solution
        
        # If we are finishing the current row
        if mark_idx == k + 1:
            return backtrack(used_diffs, rows + [current_row], row_idx + 1, 0, [0])
        
        # Determine the range for the next mark
        # If mark_idx == 0, the mark is 0 (fixed).
        # Else, the mark must be > current_row[-1] and <= M.
        
        if mark_idx == 0:
            # Mark is 0
            new_row = [0]
            # No new differences added for mark 0 (diffs with 0 are 0, but we only consider positive diffs)
            # Wait, the problem says "positive differences a_j - a_i (i < j)".
            # So diffs involving 0 are just the other elements.
            # When we add mark x, we add diffs x - a for all a in current_row.
            # For mark 0, there are no previous marks, so no diffs added.
            return backtrack(used_diffs, rows, row_idx, 1, new_row)
        
        prev_mark = current_row[-1]
        
        # Forward Checking / Pruning:
        # How many more differences do we need to fill the rest of the problem?
        # Remaining marks in this row: k + 1 - mark_idx
        # Each new mark adds (mark_idx) new differences (diffs with all previous marks in this row).
        # Wait, if we add mark at index i (0-indexed), it adds i differences (with indices 0..i-1).
        # So for mark_idx (which is the index of the NEXT mark to be added),
        # the number of new diffs added is mark_idx.
        
        # Total diffs needed from this point:
        # 1. For the rest of the current row: sum_{i=mark_idx}^{k} i
        # 2. For all subsequent rows: n_rows - row_idx - 1 rows, each with sum_{i=1}^{k} i diffs?
        #    No, each row has k(k+1)/2 diffs.
        
        # Let's calculate the minimum number of distinct differences required.
        # This is a lower bound on the size of the union of differences.
        # But we are checking if specific diffs are available.
        
        # A simpler pruning:
        # The largest possible difference is M.
        # If we can't fit the remaining required differences in the available pool [1, M] \ used_diffs, fail.
        
        # Available diffs count: M - len(used_diffs)
        # Required diffs count:
        req_diffs = 0
        # Rest of current row
        for i in range(mark_idx, k + 1):
            req_diffs += i
        # Future rows
        diffs_per_row = k * (k + 1) // 2
        for r in range(row_idx + 1, n_rows):
            req_diffs += diffs_per_row
            
        if len(used_diffs) + req_diffs > M:
            return None
            
        # Iterate through possible next marks
        start = prev_mark + 1
        # We can also bound the end.
        # The mark x must be such that x - a is not in used_diffs for all a in current_row.
        
        # To improve performance, we can limit the search range.
        # If we are at the last mark, we just need one valid x.
        
        for x in range(start, M + 1):
            # Check if x is valid
            new_diffs = set()
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Recursive call
            new_used = used_diffs | new_diffs
            res = backtrack(new_used, rows, row_idx, mark_idx + 1, current_row + [x])
            if res is not None:
                return res
                
            # Optional: Randomization to escape local minima?
            # Backtracking is exhaustive. If it fails, it fails.
            # However, the order in which we try x might affect speed, not correctness.
            # We can randomize the order of x?
            # If we randomize, we might find a solution faster if one exists,
            # but if none exists, we still explore the whole space.
            # For (7,5), the space is small.
            
        return None

    # Initial call
    # We can randomize the order of rows? No, rows are indistinguishable.
    # But the construction order matters for the "used_diffs" set growth.
    # Constructing "tighter" rows first might be better?
    # Or "looser" rows first?
    # Usually, greedy (tightest first) is good for packing.
    
    # Let's try multiple random permutations of row construction order?
    # No, rows are identical in structure. The only difference is the specific values.
    # The "order" of construction just means which row gets the "first pick" of differences.
    # Since all rows start at 0, the first row will always take the smallest available diffs.
    # So the order of rows doesn't change the set of solutions, just the path.
    # Wait, if we swap the labels of the rows, the solution is the same set.
    # So we just need to find ANY set.
    
    # To speed up, we can add a simple randomization:
    # Shuffle the candidate x values?
    # Let's try a deterministic search first.
    
    return backtrack(set(), [], 0, 0, [0])

def _greedy_construct(n_rows, k):
    """
    Fast greedy construction to get a valid baseline.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _is_valid(rows, n_rows, k):
    """
    Validates the constructed rows.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        # Check differences
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    """
    Returns the largest entry in the rows.
    """
    return max(max(row) for row in rows)

# EVOLVE-BLOCK-END
