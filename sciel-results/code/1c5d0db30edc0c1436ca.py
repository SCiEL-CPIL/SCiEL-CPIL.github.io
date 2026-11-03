import time
import random
import sys

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _solve_csp(n_rows, k, M, time_limit=1.0):
    """
    Attempt to find a solution with scope M using backtracking.
    n_rows: number of rows
    k: number of intervals (so k+1 elements)
    M: maximum value allowed
    time_limit: time in seconds to spend on this attempt
    """
    start_time = time.time()
    n_elements = k + 1  # 6
    num_diffs_per_row = n_elements * (n_elements - 1) // 2  # 15
    
    used_diffs = set()
    rows = []
    
    # We will use a recursive backtracking search.
    # To handle the large search space, we use randomized restarts and ordering.
    
    def backtrack(row_idx):
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            return True
        
        # Generate candidates for the next row
        # A row is defined by x_1, ..., x_k such that 0 < x_1 < ... < x_k <= M
        # We iterate through possible x_1, then x_2, etc.
        # To improve efficiency, we can randomize the order or use heuristics.
        
        # Heuristic: Try to use small differences first? Or random?
        # Let's try to iterate x_1 from 1 to M, x_2 from x_1+1 to M, etc.
        # This is too slow if done naively for all combinations.
        
        # Better approach:
        # We need to pick 15 distinct differences from {1, ..., M} that form a valid difference triangle.
        # Let's iterate through possible "shapes" of the row.
        # A shape is determined by the gaps g_1, ..., g_k.
        # x_i = sum(g_1...g_i).
        # The differences are sums of consecutive gaps.
        
        # Let's just iterate x_1, x_2, ..., x_k.
        # To speed up, we can prune early if any difference is already used.
        
        # Randomize the start of the search to avoid local minima
        # We can shuffle the range of x_1?
        
        # Let's use a list of candidates and shuffle it.
        # Generating all candidates is expensive.
        
        # Alternative: Pick a random subset of available differences?
        
        # Let's stick to iterating x_1..x_k but with pruning.
        # We can also sort rows by their max element to break symmetry?
        
        # For now, let's try a simple randomized backtracking.
        # We will try to fill the row by choosing x_1, then x_2, etc.
        
        # To make it faster, we can pre-check if enough differences are left.
        # Remaining differences needed: (n_rows - row_idx) * num_diffs_per_row
        # Available differences: M - len(used_diffs)
        if M - len(used_diffs) < (n_rows - row_idx) * num_diffs_per_row:
            return False
        
        # Iterate x_1
        for x1 in range(1, M - k + 2):  # x_k <= M, so x1 <= M - k
            if x1 in used_diffs:
                continue
            
            # Iterate x_2
            for x2 in range(x1 + 1, M - k + 3):
                d12 = x2 - x1
                if d12 in used_diffs or d12 == x1: # d12 != x1 unless x1=0 which is not
                    # Check if d12 is already used in this row (x1 is diff 1-0)
                    # x1 is diff(1,0). d12 is diff(2,1).
                    # They are distinct indices, so they are distinct differences.
                    # But we must check against used_diffs.
                    if d12 in used_diffs:
                        continue
                
                # ... This nested loop is too deep (5 levels).
                # We need a more efficient search.
                
                pass
        
        # Let's use a different strategy: Randomized Local Search or Simulated Annealing
        # to minimize the scope.
        # Or, use a SAT-like approach with bitsets.
        
        # Given the complexity, let's implement a simpler greedy + local search.
        return False

    # The backtracking above is incomplete. Let's implement a proper one.
    # But 5 nested loops with pruning might be okay if we cut off early.
    # However, Python recursion and loops are slow.
    
    # Let's try a different approach:
    # 1. Generate all valid rows for scope M?
    #    Number of rows is C(M, 5). For M=105, this is huge.
    
    # 2. Use a constraint solver library? Not available.
    
    # 3. Use a randomized restart with a very fast check.
    #    We will pick 5 random numbers x_1..x_5, check validity.
    #    If valid, add to rows.
    #    If we get stuck, backtrack.
    
    # Let's implement a randomized backtracking search with a time limit.
    
    def _randomized_backtrack(row_idx, depth_limit=100000):
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            return True
        
        if depth_limit <= 0:
            return False
            
        # Heuristic: Try to pick a row with small max element?
        # Or just random?
        
        # We will iterate through possible x_1, x_2, ..., x_5
        # To avoid iterating all C(M,5), we will sample.
        
        # Actually, let's just do a full search but with aggressive pruning.
        # The number of valid rows is small compared to C(M,5).
        
        # Let's try to fix the first row to a "canonical" form to break symmetry?
        # e.g., smallest possible x_1, then x_2, etc.
        
        # For now, let's just do a DFS with randomization of the order of x_1.
        
        # Generate a list of possible x_1 values
        x1_candidates = list(range(1, M - k + 2))
        random.shuffle(x1_candidates)
        
        for x1 in x1_candidates:
            if x1 in used_diffs:
                continue
            
            # Recursively find x_2..x_k
            # We need a helper to find the rest of the row given x_1
            def find_rest(pos, current_max, current_row):
                if time.time() - start_time > time_limit:
                    return False
                
                if pos == k:
                    # Check all differences
                    diffs = set()
                    for i in range(len(current_row)):
                        for j in range(i+1, len(current_row)):
                            d = current_row[j] - current_row[i]
                            if d in used_diffs or d in diffs:
                                return False
                            diffs.add(d)
                    
                    # Valid row
                    used_diffs.update(diffs)
                    rows.append(current_row)
                    
                    if _randomized_backtrack(row_idx + 1, depth_limit - 1):
                        return True
                    
                    # Backtrack
                    used_diffs.difference_update(diffs)
                    rows.pop()
                    return False
                
                # Find next element x_{pos+1}
                # It must be > current_max
                # And we need to leave room for k - pos - 1 more elements
                # So x_{pos+1} <= M - (k - pos - 1)
                
                max_next = M - (k - pos - 1)
                if current_max >= max_next:
                    return False
                
                # Candidates for next value
                candidates = list(range(current_max + 1, max_next + 1))
                random.shuffle(candidates)
                
                for v in candidates:
                    # Check differences with existing elements in current_row
                    ok = True
                    new_diffs = set()
                    for i in range(len(current_row)):
                        d = v - current_row[i]
                        if d in used_diffs or d in new_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    
                    if not ok:
                        continue
                    
                    # Also check if d is 0 (should not happen)
                    # And if d > M (should not happen)
                    
                    # Tentatively add v
                    current_row.append(v)
                    
                    # We don't add diffs to used_diffs yet, only check for conflicts within row
                    # But we must check against used_diffs (other rows)
                    
                    if find_rest(pos + 1, v, current_row):
                        return True
                    
                    current_row.pop()
                
                return False
            
            # Start with row [0, x1]
            if find_rest(1, x1, [0, x1]):
                return True
        
        return False

    # Run multiple restarts
    attempts = 0
    while time.time() - start_time < time_limit:
        used_diffs.clear()
        rows.clear()
        if _randomized_backtrack(0):
            return rows
        attempts += 1
        # Small sleep to allow other threads? No, single thread.
        
    return None

def _greedy_init(n_rows, k):
    """Greedy initialization."""
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

def construct(n_rows, k):
    deadline = time.time() + 140.0
    
    # Try to find optimal or near-optimal solution using CSP
    # Lower bound is 105
    for M in range(105, 113):
        if time.time() > deadline - 10:
            break
        # Try to solve for scope M
        # Use a time limit for each M to ensure we try multiple M values
        time_per_m = min(10.0, (deadline - time.time()) / (113 - M))
        sol = _solve_csp(n_rows, k, M, time_limit=time_per_m)
        if sol is not None:
            if _is_valid(sol):
                return sol
    
    # Fallback to greedy if CSP fails
    rows = _greedy_init(n_rows, k)
    return rows
