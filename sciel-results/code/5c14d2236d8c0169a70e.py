"""Difference triangle set (7,5) solver with Genetic Algorithm over difference multisets."""
import time
import random
import sys
import math

# Deadline guard: return best result before 145s (5s safety margin under 150s kill)
_DEADLINE = 145.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _row_diffs(row):
    """Return the list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _is_valid(rows, k):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _diffs_to_row(diffs, k):
    """Reconstruct a strictly increasing row from its difference multiset.
    
    Args:
        diffs: list of positive differences (should have exactly k*(k+1)//2 elements)
        k: number of intervals (row length is k+1)
    
    Returns:
        The row as a list of k+1 integers starting at 0, or None if reconstruction fails.
    """
    expected_count = k * (k + 1) // 2
    if len(diffs) != expected_count:
        return None
    
    # Sort differences
    sorted_diffs = sorted(diffs)
    
    # Try to reconstruct by greedily assigning differences to positions
    # A row [0, a1, a2, ..., ak] has differences:
    # a1-0, a2-0, ..., ak-0, a2-a1, a3-a1, ..., ak-a1, ..., ak-a(k-1)
    # 
    # Strategy: We know the largest difference in the row is ak - 0 = ak (the max entry).
    # The second largest could be ak-1 or ak-a1, etc.
    # 
    # A simpler approach: use the fact that if we sort the differences,
    # the largest one must be the last element (ak).
    # Then the next largest must be either a(k-1) or ak-a1.
    # This is complex. Let's use a different approach:
    # 
    # Actually, for a given multiset of differences, there may be multiple valid rows.
    # We can try to construct one by working backwards from the largest difference.
    
    # The largest difference in the row is always the max entry (ak).
    # So ak = max(sorted_diffs)
    
    # Let's try a greedy reconstruction from the end:
    # row[k] = max_diff
    # For j from k-1 down to 1:
    #   row[j] must be such that all differences row[k]-row[j] are in the remaining set,
    #   and row[j]-row[i] for i<j are also in the remaining set.
    # This is still complex.
    
    # Alternative: Use the structure of difference triangles.
    # For k=5, we have 6 entries: 0, a, b, c, d, e
    # Differences: a, b, c, d, e, b-a, c-a, d-a, e-a, c-b, d-b, e-b, d-c, e-c, e-d
    
    # Let's try a simple heuristic: 
    # 1. The largest difference is e (the last element).
    # 2. Remove e from the set.
    # 3. The next largest difference must be either d or e-a. 
    #    If we assume it's d, then d is the second-to-last element.
    #    If it's e-a, then a = e - (second_largest).
    
    # This is getting complicated. Let's use a simpler approach:
    # We'll try to build the row by choosing each element one by one,
    # ensuring that the differences to previous elements are in the multiset.
    
    # Actually, let's just try all possible assignments for small k.
    # For k=5, this might be feasible with pruning.
    
    # Simpler approach: 
    # We know that the differences include all pairwise differences.
    # Let's sort the differences and try to assign them.
    
    # For now, let's use a simple greedy: 
    # Start with row = [0]
    # The next element must be such that its difference from 0 is in the set.
    # Pick the smallest available difference as the next element? Not necessarily.
    
    # Let's try a different strategy: 
    # The sum of all differences in a row has a specific structure.
    # For row [0, a1, ..., ak], the sum of all differences is:
    # sum_{i<j} (a_j - a_i) = sum_j a_j * j - sum_i a_i * (k-i)
    # This doesn't directly help.
    
    # Let's just try to reconstruct by trying each possible value for each position.
    # For k=5, the search space is manageable with pruning.
    
    # Actually, the simplest correct approach: 
    # A row is valid if and only if its difference multiset matches the input.
    # We can try to find a row by backtracking.
    
    # For efficiency, let's use a heuristic: 
    # The largest element is the max difference.
    # The second largest element is either the second max difference, or max - min.
    
    # Let's implement a simple backtracking search for the row reconstruction.
    
    def backtrack(pos, current_row, remaining_diffs):
        """Try to build the row from position pos onwards."""
        if pos == k + 1:
            # Check if all differences are used
            if not remaining_diffs:
                return current_row[:]
            return None
        
        if pos == 0:
            # First element is 0
            return backtrack(1, [0], remaining_diffs)
        
        # Try each possible value for current_row[pos]
        # The value must be greater than current_row[pos-1]
        # And the differences to all previous elements must be in remaining_diffs
        
        # To limit the search, we can try values that are differences from previous elements
        # Specifically, current_row[pos] - current_row[i] for i < pos must be in remaining_diffs
        
        # Let's try each difference in remaining_diffs as a candidate for current_row[pos] - current_row[pos-1]
        # Then check if all other differences are also in remaining_diffs
        
        for d in list(remaining_diffs.keys()):
            candidate = current_row[pos-1] + d
            # Check if candidate > current_row[pos-1] (already true since d > 0)
            # Check if all differences to previous elements are in remaining_diffs
            valid = True
            needed_diffs = []
            for i in range(pos):
                diff = candidate - current_row[i]
                if diff not in remaining_diffs:
                    valid = False
                    break
                needed_diffs.append(diff)
            
            if not valid:
                continue
            
            # Check if we have enough copies of each needed difference
            temp_remaining = dict(remaining_diffs)
            for diff in needed_diffs:
                temp_remaining[diff] -= 1
                if temp_remaining[diff] < 0:
                    valid = False
                    break
                if temp_remaining[diff] == 0:
                    del temp_remaining[diff]
            
            if not valid:
                continue
            
            # Recurse
            result = backtrack(pos + 1, current_row + [candidate], temp_remaining)
            if result is not None:
                return result
        
        return None
    
    # Convert diffs to a count dictionary
    diff_counts = {}
    for d in sorted_diffs:
        diff_counts[d] = diff_counts.get(d, 0) + 1
    
    result = backtrack(0, [], diff_counts)
    return result

def _greedy_row(used, k, rng, max_val=None):
    """Build one row greedily, choosing the smallest available next mark.
    used: set of forbidden differences.
    max_val: optional upper bound for entries.
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if max_val is not None and x > max_val:
            break
        # Occasionally jump ahead to create diversity
        if rng.random() < 0.1:
            x += rng.randint(1, 3)
            if max_val is not None and x > max_val:
                break
        
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _greedy_full(n_rows, k, rng, max_val=None):
    """Build all rows greedily with randomization."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, max_val)
        if len(row) < k + 1:
            return None
        rows.append(row)
    return rows

def _reconstruct_row_from_diffs(diffs, k, rng):
    """Reconstruct a strictly increasing row from a multiset of differences.
    
    Args:
        diffs: list of positive differences (should have exactly k*(k+1)//2 elements)
        k: number of intervals (row length is k+1)
        rng: random number generator for tie-breaking
    
    Returns:
        The row as a list of k+1 integers starting at 0, or None if reconstruction fails.
    """
    expected_count = k * (k + 1) // 2
    if len(diffs) != expected_count:
        return None
    
    # Count occurrences of each difference
    diff_counts = {}
    for d in diffs:
        diff_counts[d] = diff_counts.get(d, 0) + 1
    
    # The largest difference in the row is always the max entry (last element).
    # We'll try to reconstruct the row by choosing each element one by one.
    
    def backtrack(pos, current_row, remaining_counts):
        """Try to build the row from position pos onwards."""
        if pos == k + 1:
            # Check if all differences are used
            for cnt in remaining_counts.values():
                if cnt != 0:
                    return None
            return current_row[:]
        
        if pos == 0:
            # First element is 0
            return backtrack(1, [0], remaining_counts)
        
        # Try each possible value for current_row[pos]
        # The value must be greater than current_row[pos-1]
        # And the differences to all previous elements must be in remaining_counts
        
        # To limit the search, we can try values that are differences from previous elements
        # Specifically, current_row[pos] - current_row[i] for i < pos must be in remaining_counts
        
        # Let's try each difference in remaining_counts as a candidate for current_row[pos] - current_row[pos-1]
        candidates = list(remaining_counts.keys())
        rng.shuffle(candidates)  # Randomize for diversity
        
        for d in candidates:
            if remaining_counts[d] <= 0:
                continue
            candidate = current_row[pos-1] + d
            # Check if all differences to previous elements are in remaining_counts
            valid = True
            needed_diffs = []
            for i in range(pos):
                diff = candidate - current_row[i]
                if diff not in remaining_counts or remaining_counts[diff] <= 0:
                    valid = False
                    break
                needed_diffs.append(diff)
            
            if not valid:
                continue
            
            # Check if we have enough copies of each needed difference
            temp_remaining = dict(remaining_counts)
            for diff in needed_diffs:
                temp_remaining[diff] -= 1
                if temp_remaining[diff] < 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Recurse
            result = backtrack(pos + 1, current_row + [candidate], temp_remaining)
            if result is not None:
                return result
        
        return None
    
    result = backtrack(0, [], diff_counts)
    return result

def _calculate_conflicts(rows):
    """Count the number of duplicate differences across all rows."""
    seen = set()
    conflicts = 0
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    conflicts += 1
                else:
                    seen.add(d)
    return conflicts

def _sa_optimize(rows, k, rng, time_budget):
    """Simulated Annealing to minimize scope and conflicts."""
    current_rows = [list(r) for r in rows]
    current_conflicts = _calculate_conflicts(current_rows)
    current_scope = _max_entry(current_rows)
    current_cost = current_scope + 100 * current_conflicts
    
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    best_conflicts = current_conflicts
    best_cost = current_cost
    
    start_time = time.time()
    temperature = 10.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_budget and temperature > 0.1:
        if _elapsed() > _DEADLINE - 5:
            break
            
        # Generate a neighbor
        new_rows = [list(r) for r in current_rows]
        row_idx = rng.randrange(len(new_rows))
        row = new_rows[row_idx]
        
        # Strategy: Shift a random element up or down, or swap two elements
        move_type = rng.random()
        if move_type < 0.7:
            # Shift an element
            if len(row) > 1:
                pos = rng.randrange(1, len(row))
                delta = rng.randint(-5, 5)
                new_val = row[pos] + delta
                # Ensure strictly increasing
                prev_val = row[pos-1]
                next_val = row[pos+1] if pos < len(row)-1 else float('inf')
                
                if new_val > prev_val and new_val < next_val:
                    row[pos] = new_val
                    # Update differences for this row and check global conflicts
                    # For efficiency, we just recalculate cost for now, 
                    # but we can optimize by tracking diffs. 
                    # Given small k, full recalc is okay for moderate iterations.
                else:
                    # Invalid move, revert
                    row[pos] = row[pos] # No change
                    continue
        else:
            # Swap two elements (if it keeps order? No, swap breaks order usually. 
            # Better: Pick two distinct positions and try to adjust them to swap roles or just randomize one)
            # Let's stick to shifting as it's safer for maintaining increasing order.
            # Or: Pick a random row, pick a random position, set to a random valid value in range
            if len(row) > 1:
                pos = rng.randrange(1, len(row))
                prev_val = row[pos-1]
                next_val = row[pos+1] if pos < len(row)-1 else current_scope + 10
                if next_val - prev_val > 1:
                    row[pos] = rng.randint(prev_val + 1, next_val - 1)
                else:
                    continue

        # Calculate new cost
        new_conflicts = _calculate_conflicts(new_rows)
        new_scope = _max_entry(new_rows)
        new_cost = new_scope + 100 * new_conflicts
        
        # Acceptance probability
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or rng.random() < math.exp(-delta_cost / temperature):
            current_rows = new_rows
            current_conflicts = new_conflicts
            current_scope = new_scope
            current_cost = new_cost
            
            if current_conflicts == 0 and current_scope < best_scope:
                best_rows = [list(r) for r in current_rows]
                best_scope = current_scope
                best_conflicts = 0
                best_cost = current_cost
        
        temperature *= cooling_rate
    
    return best_rows

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Multiple randomized greedy restarts to find a valid start
    best_rows = None
    best_score = float('inf')
    
    n_restarts = 200
    for i in range(n_restarts):
        if _elapsed() > _DEADLINE - 30:
            break
        rng2 = random.Random(12345 + i * 7919)
        rows = _greedy_full(n_rows, k, rng2)
        if rows is None:
            continue
        if not _is_valid(rows, k):
            continue
        score = _max_entry(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: pure greedy without randomization jumps
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
        best_score = _max_entry(best_rows)
    
    # Phase 2: Simulated Annealing
    if _elapsed() < _DEADLINE - 10:
        time_left = _DEADLINE - 10 - _elapsed()
        # Limit SA time to ensure we return in time
        sa_time = min(60, time_left)
        improved_rows = _sa_optimize(best_rows, k, rng, time_budget=sa_time)
        
        if improved_rows and _is_valid(improved_rows, k):
            new_score = _max_entry(improved_rows)
            if new_score < best_score:
                best_rows = improved_rows
                best_score = new_score
    
    # Final validation
    if not _is_valid(best_rows, k):
        # Shouldn't happen, but fallback to a known valid greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
    
    return best_rows
