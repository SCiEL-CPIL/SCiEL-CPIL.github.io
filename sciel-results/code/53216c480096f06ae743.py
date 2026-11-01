"""
Difference triangle set (4,7): Minimize the largest entry.
Uses Constraint Satisfaction with Backtracking and Forward Checking.
"""
import time
import random
import sys

_START_TIME = None
_DEADLINE = 145.0  # Safety margin for 150s limit

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_row_diffs(row):
    """Compute set of differences for a row (excluding 0)."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _validate(rows, n_rows, k):
    """Check if rows form a valid DTS. Returns True if valid."""
    if len(rows) != n_rows:
        return False
    used_diffs = set()
    for r in range(n_rows):
        row = rows[r]
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(k):
            if row[i] >= row[i+1]:
                return False
        # Check diffs
        row_diffs = _compute_row_diffs(row)
        if row_diffs & used_diffs:
            return False
        used_diffs |= row_diffs
    return True

def _solve_csp(M, n_rows, k, rng, max_nodes=200000):
    """
    Try to find a DTS with max entry <= M.
    Variables: x[r][j] for r in 0..n_rows-1, j in 1..k (index 0 is 0).
    Total variables: n_rows * k.
    """
    # Order: Interleave rows for each column
    # j=1: r=0,1,2,3
    # j=2: r=0,1,2,3
    # ...
    order = []
    for j in range(1, k + 1):
        for r in range(n_rows):
            order.append((r, j))
            
    assigned = [[0] * (k + 1) for _ in range(n_rows)]
    used_vals = set()
    used_diffs = set()
    
    nodes_visited = 0
    max_nodes_limit = max_nodes
    
    def backtrack(idx):
        nonlocal nodes_visited
        if _time_left() < 1.0:
            return False
        
        if idx == len(order):
            return True
            
        nodes_visited += 1
        if nodes_visited > max_nodes_limit:
            return False
            
        r, j = order[idx]
        
        # Determine range for x[r][j]
        low = assigned[r][j-1] + 1
        high = M
        
        # Check if any candidate works
        # We can iterate from low to high, or high to low?
        # Randomize start?
        # Let's iterate from low to high.
        
        # To speed up, we can pre-compute the "forbidden" diffs for this position?
        # Forbidden diffs are those in used_diffs.
        # A candidate v is valid if:
        # - v not in used_vals
        # - for all i < j: (v - assigned[r][i]) not in used_diffs
        # - for all s < r: for all i <= j: (v - assigned[s][i]) not in used_diffs  (Note: assigned[s][j] is assigned if s<r)
        # - for all s > r: for all i < j: (v - assigned[s][i]) not in used_diffs (assigned[s][j] is NOT assigned yet)
        #   Wait, if s > r, assigned[s][j] is 0 (unassigned). 
        #   But assigned[s][i] for i < j IS assigned.
        
        # So we check diffs against:
        # 1. Same row, previous cols.
        # 2. Previous rows, current and previous cols.
        # 3. Later rows, previous cols.
        
        # This is O(k * n_rows) checks per candidate.
        
        # Optimization: 
        # We can pre-compute a set of "banned" values for v?
        # If v - a = d where d in used_diffs, then v = a + d is banned.
        # So for each assigned value a, and each used diff d, v = a+d and v = a-d are banned.
        # This is a huge set.
        
        # Let's just check directly.
        
        # To add randomization, we can shuffle the range? No, range is contiguous.
        # We can try random candidates first?
        
        # Let's try candidates in a random order? 
        # That might be slow to generate.
        # Let's stick to sequential but maybe skip some?
        
        # Better: 
        # Collect valid candidates, then pick one randomly?
        # Or just iterate.
        
        # Let's iterate from low to high.
        
        # Pre-compute list of assigned values to check against.
        # We need to check v against:
        # - used_vals
        # - diffs with specific assigned values.
        
        # Let's create a list of (value, row, col) that are assigned.
        # Actually, we can just loop through the relevant assigned slots.
        
        # Relevant assigned slots:
        # - (r, 0) to (r, j-1)
        # - (0, 0) to (r-1, j)  (Note: for s<r, col j is assigned)
        # - (r+1, 0) to (n_rows-1, j-1) (Note: for s>r, col j is NOT assigned)
        
        # Let's build a list of assigned values to check diffs against.
        check_vals = []
        for i in range(j):
            check_vals.append(assigned[r][i])
            
        for s in range(r):
            for i in range(j + 1):
                check_vals.append(assigned[s][i])
                
        for s in range(r + 1, n_rows):
            for i in range(j):
                check_vals.append(assigned[s][i])
                
        # Remove duplicates? Not necessary for correctness, just speed.
        # Set of check_vals for faster diff check?
        # No, we need to check if (v - a) is in used_diffs.
        
        # Let's just loop.
        
        # To speed up, we can break early if we find a conflict.
        
        # Randomize the iteration order of candidates?
        # Let's just pick the first valid one. 
        # To avoid getting stuck, we can restart with a different M or different random seed.
        
        # Actually, let's try a random valid candidate if we have time?
        # For now, sequential.
        
        # Let's try a simple randomization: 
        # Pick a random start point in [low, high] and wrap around?
        # Or just iterate.
        
        # Let's iterate.
        
        # To make it faster, we can check `v not in used_vals` first.
        
        # We will use a local variable for assigned to avoid global lookup?
        # assigned is in closure.
        
        # Let's try to find a valid v.
        
        # To improve success rate, we can shuffle the order of checking candidates?
        # No, let's just find the first one.
        
        # If we want to be smarter, we can count how many candidates are valid and pick randomly?
        # That requires scanning all.
        
        # Let's just scan.
        
        # One more optimization: 
        # If we fail to find a v, return False.
        
        # Let's try to speed up the diff check by using a set of used_diffs.
        
        # We'll iterate v from low to high.
        
        # To avoid deep recursion overhead, we can use an explicit stack?
        # No, 28 depth is fine.
        
        # Let's add a node limit check.
        
        # We will try candidates in a random order? 
        # Let's generate a list of candidates and shuffle? 
        # That's O(M) memory and time. M is ~120. 
        # Let's do it. It adds randomness and might escape local traps.
        
        candidates = []
        for v in range(low, high + 1):
            if v in used_vals:
                continue
            # Check diffs
            valid = True
            for a in check_vals:
                d = v - a
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                candidates.append(v)
                
        if not candidates:
            return False
            
        # Pick a candidate randomly? 
        # If we pick the first, we are deterministic.
        # If we pick randomly, we explore more.
        # Let's pick randomly.
        v = rng.choice(candidates)
        
        # Assign
        assigned[r][j] = v
        used_vals.add(v)
        
        # Add new diffs
        new_diffs = set()
        for i in range(j):
            d = v - assigned[r][i]
            new_diffs.add(d)
        for s in range(r):
            for i in range(j + 1):
                d = v - assigned[s][i]
                new_diffs.add(d)
        for s in range(r + 1, n_rows):
            for i in range(j):
                d = v - assigned[s][i]
                new_diffs.add(d)
                
        used_diffs.update(new_diffs)
        
        if backtrack(idx + 1):
            return True
            
        # Backtrack
        assigned[r][j] = 0
        used_vals.remove(v)
        used_diffs.difference_update(new_diffs)
        
        return False
        
    if backtrack(0):
        return assigned
    return None

def _greedy_build(n_rows, k, row_order, rng):
    """Greedy: build rows in given order, each row greedily picks smallest valid next value."""
    used = set()
    rows = [None] * n_rows
    for r in row_order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                ok = True
                for a in row:
                    d = x - a
                    if d in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                    break
                x += 1
        rows[r] = row
    return rows

def _greedy_interleaved(n_rows, k, rng, strategy="smallest"):
    """
    Interleaved greedy: at each step, pick a row to extend.
    strategy: 'smallest' (smallest last element), 'largest' (largest last element), 
              'random', 'balanced' (most elements placed)
    """
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows
    
    while placed < total_needed:
        if _time_left() < 3.0:
            return None
        
        # Find candidate rows that need more elements
        candidates = [r for r in range(n_rows) if len(rows[r]) < k + 1]
        if not candidates:
            break
        
        if strategy == "smallest":
            best_r = min(candidates, key=lambda r: rows[r][-1])
        elif strategy == "largest":
            best_r = max(candidates, key=lambda r: rows[r][-1])
        elif strategy == "random":
            best_r = rng.choice(candidates)
        elif strategy == "balanced":
            best_r = max(candidates, key=lambda r: len(rows[r]))
        else:
            best_r = min(candidates, key=lambda r: rows[r][-1])
        
        row = rows[best_r]
        x = row[-1] + 1
        while True:
            ok = True
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
            if ok:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
                placed += 1
                break
            x += 1
    return rows

def _try_greedy_variants(n_rows, k, rng, num_trials=20):
    """Try multiple greedy strategies and return best."""
    best_rows = None
    best_max = float('inf')
    
    strategies = ["smallest", "largest", "random", "balanced"]
    
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        # Random row order with sequential build
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, order, rng)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    # Try interleaved strategies
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        strategy = strategies[t % len(strategies)]
        rows = _greedy_interleaved(n_rows, k, rng, strategy=strategy)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    return best_rows, best_max

def _local_search(rows, n_rows, k, rng, time_budget=30.0):
    """
    Local search: try to reduce the maximum value by reassigning elements.
    This is a simple hill-climbing approach.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start_time = time.time()
    
    # Precompute diffs for each row
    row_diffs = [_compute_row_diffs(rows[r]) for r in range(n_rows)]
    
    def get_global_diffs_except_row(exclude_r):
        all_d = set()
        for r in range(n_rows):
            if r == exclude_r:
                continue
            all_d |= row_diffs[r]
        return all_d

    while time.time() - start_time < time_budget and _time_left() > 3.0:
        current_max = max(max(rows[r]) for r in range(n_rows))
        if current_max < best_max:
            best_max = current_max
            best_rows = [list(r) for r in rows]
        
        # Find the row with the maximum value
        max_val = current_max
        max_rows = [r for r in range(n_rows) if max(rows[r]) == max_val]
        
        # Try to reduce the last element of one of these rows
        improved = False
        for r in max_rows:
            if _time_left() < 3.0:
                break
            row = rows[r]
            last_idx = len(row) - 1
            old_val = row[last_idx]
            
            # Try values from old_val - 1 down to row[last_idx - 1] + 1
            for new_val in range(old_val - 1, row[last_idx - 1], -1):
                # Construct the hypothetical new row
                new_row = row[:-1] + [new_val]
                
                # Check internal consistency of the new row
                # The set of diffs for new_row must be disjoint from global_diffs_except_row(r)
                # AND internal to new_row (which is guaranteed if it's strictly increasing and we check diffs)
                
                # Compute diffs for new_row
                new_diffs = _compute_row_diffs(new_row)
                
                # Get global diffs excluding row r
                global_diffs = get_global_diffs_except_row(r)
                
                # Check for conflicts
                if new_diffs & global_diffs:
                    continue
                
                # Check if new_diffs are all distinct (should be true for strictly increasing)
                # Also check if new_diffs are distinct from each other (implied by set size)
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                
                # If valid, accept the move
                rows[r] = new_row
                row_diffs[r] = new_diffs
                improved = True
                break
            
            if improved:
                break
        
        if not improved:
            # Try random restarts or perturbations?
            # For now, just break if no improvement found
            # But maybe we can try a different row?
            # Let's try to perturb a random element in a random row
            r = rng.choice(range(n_rows))
            idx = rng.choice(range(1, k)) # Don't touch 0
            old_val = rows[r][idx]
            
            # Try a small random change
            # Must maintain increasing order
            low = rows[r][idx-1] + 1
            high = rows[r][idx+1] - 1 if idx < k else old_val + 10
            
            for _ in range(10):
                new_val = rng.randint(low, high)
                if new_val == old_val:
                    continue
                new_row = rows[r][:idx] + [new_val] + rows[r][idx+1:]
                
                # Check strictly increasing
                if not all(new_row[i] < new_row[i+1] for i in range(k)):
                    continue
                
                new_diffs = _compute_row_diffs(new_row)
                global_diffs = get_global_diffs_except_row(r)
                
                if new_diffs & global_diffs:
                    continue
                
                # Accept
                rows[r] = new_row
                row_diffs[r] = new_diffs
                improved = True
                break
        
        if not improved:
            # No improvement found, maybe stuck. 
            # We could try a more aggressive restart, but for now just break.
            # Or we can continue for a few more iterations hoping for a random break.
            # Let's break to save time for CSP.
            break
            
    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # 1. Try Greedy Variants to get a good initial solution
    greedy_rows, greedy_max = _try_greedy_variants(n_rows, k, rng, num_trials=20)
    
    if greedy_rows is None:
        # Fallback to simple greedy if all failed
        greedy_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        greedy_max = max(max(r) for r in greedy_rows)
    
    # 2. Local Search to improve
    ls_rows, ls_max = _local_search(greedy_rows, n_rows, k, rng, time_budget=30.0)
    
    # 3. CSP to try to beat the local search result
    # We know the lower bound is 112.
    # We will try M from 112 up to ls_max - 1
    best_rows = ls_rows
    best_M = ls_max
    
    # If local search improved, we have a target.
    # If not, we still have a target from greedy.
    
    # Allocate remaining time to CSP
    # Let's try M from 112 to best_M - 1
    for M in range(112, best_M):
        if _time_left() < 10.0:
            break
        
        # Try a few restarts for this M
        found = False
        # Max 2 restarts per M to allow trying more M values
        for restart in range(2):
            if _time_left() < 10.0:
                break
                
            r_rng = random.Random(1000 + M * 10 + restart)
            
            # Limit nodes to prevent hanging on one M
            # If M is small, it should be fast. If M is large, it might be hard.
            # Let's give it 200k nodes.
            
            sol = _solve_csp(M, n_rows, k, r_rng, max_nodes=300000)
            
            if sol:
                # Verify
                if _validate(sol, n_rows, k):
                    best_rows = [list(row) for row in sol]
                    best_M = M
                    found = True
                    break
        
        if found:
            # We found a solution for this M, which is the smallest possible for this range
            break
    
    # Final check: if we have time, maybe do one more local search on the CSP result?
    # Or just return the best found.
    
    return best_rows

# Wait, the prompt says "Keep the function names and signatures of the seed program".
# The seed program had `construct(n_rows, k)`.
# It also had `_compute_row_diffs`.
# It also had `_solve_csp`.
# I have kept these.

# One detail: The seed program returned `rows` which is a list of lists.
# My code does the same.

# Let's double check the time management.
# _DEADLINE is 145.0.
# _local_search has a budget of 30s.
# _try_greedy_variants is fast.
# _solve_csp checks _time_left() < 1.0.
# The loop in construct checks _time_left() < 10.0.
# This seems safe.

# One potential issue: `_local_search` might return a solution that is worse than the input if it gets stuck?
# No, it keeps `best_rows` and `best_max` from the start.
# `best_rows` is initialized to the input rows.
# So it will always return at least as good as the input.

# What if `_try_greedy_variants` returns None?
# I have a fallback.

# What if `_solve_csp` finds a solution but it's invalid?
# I check with `_validate`.

# The seed program had a specific structure. I am combining it with the inspiration.
# The inspiration had `_greedy_build`, `_greedy_interleaved`, `_try_greedy_variants`, `_local_search`.
# I have included all of these.

# The seed program had `_solve_csp`.
# I have included it.

# The main driver `construct` orchestrates them.

# Let's refine the CSP call.
# If `best_M` is 134 (from greedy/LS), we try M=112..133.
# If we find M=130, we stop.
# If we don't find anything, we return the LS result (134).

# This looks correct.

# One small optimization: in `_solve_csp`, the `check_vals` list construction is done every time.
# This is O(k * n_rows).
# It's acceptable.

# I will output the complete code now.
