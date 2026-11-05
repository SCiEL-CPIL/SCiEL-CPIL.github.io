"""Difference triangle set (4,7) solver.

Strategy:
- Compute a tight lower bound on the scope (largest entry) using a counting argument.
- Use randomized backtracking search to pack n_rows rows of (k+1) increasing integers
  starting at 0, minimizing the maximum entry.
- Use a deadline guard to return the best valid result found before the time limit.

The key insight: we want to minimize the largest entry (scope). We search for a valid
configuration where all positive differences within rows are distinct across all rows.
"""
import time
import random
import sys

def _lower_bound(n_rows, k):
    """Counting lower bound: total distinct differences needed vs available in [1, M].
    Each row of k+1 elements has C(k+1, 2) = k*(k+1)//2 differences.
    Total differences needed: n_rows * k*(k+1)//2
    Available differences in [1, M]: M
    So M >= n_rows * k*(k+1)//2
    """
    total_needed = n_rows * k * (k + 1) // 2
    return total_needed

def _try_construct(n_rows, k, scope, rng, time_limit):
    """Try to construct n_rows rows with largest entry <= scope.
    Uses randomized backtracking.
    Returns list of rows or None if failed.
    """
    start_time = time.time()
    
    # We need n_rows rows, each with k+1 elements starting at 0, strictly increasing, max <= scope
    # All positive differences (a_j - a_i for i < j) must be distinct across all rows.
    
    # Total number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Available differences are 1..scope
    if total_diffs > scope:
        return None
    
    # We'll build rows one at a time. For each row, we choose k elements from 1..scope
    # (since 0 is fixed as the first element).
    
    # Strategy: for each row, we need to pick k values from 1..scope such that
    # all differences with 0 and among themselves are unused.
    
    used = set()
    rows = []
    
    # Pre-compute: for a given set of chosen values, what differences are generated?
    # For a row [0, a1, a2, ..., ak], differences are:
    # a1, a2, ..., ak (differences with 0)
    # a2-a1, a3-a1, ..., ak-a1
    # a3-a2, a4-a2, ..., ak-a2
    # ...
    
    # Randomized backtracking: build row by row, within each row pick elements in order.
    
    def backtrack(row_idx, pos, current_row, used_set):
        """Build row_idx-th row. pos is how many elements we've placed (0 is fixed).
        current_row is the list of chosen elements so far (excluding 0).
        used_set is the set of all differences used so far."""
        if time.time() - start_time > time_limit:
            return None
        
        if row_idx == n_rows:
            return [r for r in rows] if rows else None
        
        if pos == k:
            # Row is complete
            rows.append([0] + current_row)
            result = backtrack(row_idx + 1, 0, [], used_set)
            if result is not None:
                return result
            rows.pop()
            return None
        
        # Determine which differences are already used by this row
        # current_row has pos elements: c[0], c[1], ..., c[pos-1]
        # New element x must satisfy: x - c[i] not in used_set for all i, and x - 0 = x not in used_set
        # Also x > current_row[-1] (or x > 0 if pos == 0)
        
        min_val = current_row[-1] + 1 if pos > 0 else 1
        
        # Collect candidate values
        candidates = []
        for x in range(min_val, scope + 1):
            # Check if x itself is used (difference with 0)
            if x in used_set:
                continue
            # Check differences with existing elements in this row
            valid = True
            new_diffs = []
            for c in current_row:
                d = x - c
                if d in used_set:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                candidates.append((x, new_diffs))
        
        if not candidates:
            return None
        
        # Shuffle for randomization
        rng.shuffle(candidates)
        
        # Try candidates with some randomization
        for x, new_diffs in candidates:
            # Also add the difference x (with 0) to new_diffs
            all_new = new_diffs + [x]
            
            # Check for conflicts within new_diffs themselves? No, they're all different by construction
            # since x - c[i] are all distinct for distinct c[i], and x is not equal to any x - c[i] (since c[i] > 0)
            
            # Apply
            used_set.update(all_new)
            current_row.append(x)
            
            result = backtrack(row_idx, pos + 1, current_row, used_set)
            if result is not None:
                return result
            
            # Undo
            current_row.pop()
            used_set.difference_update(all_new)
        
        return None
    
    # Start backtracking
    result = backtrack(0, 0, [], used)
    return result


def _greedy_construct(n_rows, k):
    """Simple greedy: each row takes smallest available marks."""
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


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope)."""
    
    wall_start = time.time()
    deadline = wall_start + 140  # Safety margin before 150s kill
    
    # Get the greedy solution as a baseline
    greedy_rows = _greedy_construct(n_rows, k)
    greedy_scope = max(max(r) for r in greedy_rows)
    
    best_rows = greedy_rows
    best_scope = greedy_scope
    
    # Lower bound
    lb = _lower_bound(n_rows, k)
    
    # Try to improve using randomized backtracking
    # Search from lb upward, but also try values between lb and greedy_scope
    
    rng = random.Random(42)
    
    # First, try to find a solution at the lower bound
    # Then try intermediate values
    
    # We'll try several random seeds and different scope values
    # Strategy: binary search or linear search for minimum feasible scope
    
    # Try from lb up to greedy_scope - 1, but with time constraints
    # Use randomized restarts
    
    # Determine which scopes to try
    # If greedy_scope - lb is small, try all
    # Otherwise, use a heuristic
    
    search_scopes = []
    if greedy_scope - lb <= 20:
        search_scopes = list(range(lb, greedy_scope))
    else:
        # Try lb, then some quantiles, then values close to greedy_scope
        search_scopes = [lb]
        # Add some intermediate values
        step = (greedy_scope - lb) // 10
        for i in range(1, 10):
            v = lb + i * step
            if v < greedy_scope:
                search_scopes.append(v)
        # Add values near greedy_scope
        for v in range(greedy_scope - 3, greedy_scope):
            if v >= lb and v not in search_scopes:
                search_scopes.append(v)
    
    # Shuffle the search order to explore different regions
    # But keep lb first (it's the most important)
    if len(search_scopes) > 1:
        rest = search_scopes[1:]
        rng.shuffle(rest)
        search_scopes = [search_scopes[0]] + rest
    
    for scope in search_scopes:
        if time.time() > deadline:
            break
        if scope >= best_scope:
            continue
        
        # Try multiple random restarts for this scope
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            time_per_attempt = min(20, (deadline - time.time()) / max(1, max_restarts - restart))
            rng_attempt = random.Random(rng.randint(0, 2**31))
            
            result = _try_construct(n_rows, k, scope, rng_attempt, time_per_attempt)
            if result is not None:
                # Verify
                all_diffs = set()
                valid = True
                for row in result:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                        if not valid:
                            break
                    if not valid:
                        break
                if valid and len(result) == n_rows and all(len(r) == k + 1 for r in result):
                    actual_scope = max(max(r) for r in result)
                    if actual_scope < best_scope:
                        best_scope = actual_scope
                        best_rows = result
                    break  # Found a solution at this scope, move to next
    
    return best_rows
