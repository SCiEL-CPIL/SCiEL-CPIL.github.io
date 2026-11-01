"""Rewritten program using constraint satisfaction and simulated annealing."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within each row are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    deadline = time.time() + 105.0  # Safety margin
    
    # Parameters
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Initial scope estimate: greedy gives a rough upper bound
    # We want to start near the theoretical minimum or slightly above
    # The simple counting bound is roughly sqrt(2 * total_diffs) for one ruler,
    # but for multiple rows, the scope M must satisfy M(M+1)/2 >= total_diffs
    # So M >= sqrt(2 * total_diffs)
    min_possible_scope = int((2 * total_diffs) ** 0.5)
    
    # Start with a scope that is feasible but tight
    # Greedy usually gives a much larger scope, so we start tighter and expand if needed
    # Let's start at min_possible_scope + small buffer
    initial_scope = min_possible_scope + 5
    
    # Try to build a valid solution with this scope using backtracking/SA
    best_rows = None
    best_scope = float('inf')
    
    # 1. Try to find a solution with a tight scope using Simulated Annealing
    # We represent the state as a list of rows.
    # To make SA work, we need a way to measure "conflict" or "invalidity".
    
    # Strategy:
    # Start with a random assignment of values within [0, scope].
    # Objective: Minimize the number of duplicate differences.
    # If duplicates == 0, we have a valid solution.
    
    # However, ensuring strictly increasing and starting at 0 is a hard constraint.
    # We can enforce this by only sampling valid values for each position.
    
    # Let's use a different approach:
    # 1. Generate a valid set of rows with a generous scope using a fast heuristic.
    # 2. Use Local Search / SA to reduce the scope.
    
    # Step 1: Fast Heuristic Initialization
    # We will use a randomized greedy that tries to keep values small.
    rows = _initialize(n_rows, k, initial_scope, deadline)
    
    if rows is None:
        # If initialization fails (too tight), expand scope and retry
        scope = initial_scope
        while time.time() < deadline:
            scope += 10
            rows = _initialize(n_rows, k, scope, deadline)
            if rows:
                break
        if not rows:
            # Fallback to simple greedy without scope limit
            rows = _greedy_construct(n_rows, k)
            if not rows:
                return [[0] * (k + 1) for _ in range(n_rows)] # Invalid fallback, should not happen
            
    current_scope = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    # Step 2: Optimize Scope using Simulated Annealing
    # We will try to reduce the maximum value.
    
    # Define a cost function:
    # Cost = (max_value - target_scope) * W1 + num_conflicts * W2
    # We want to drive num_conflicts to 0 and max_value down.
    
    # A better objective for SA:
    # We fix the scope M. We try to find rows within [0, M] that are valid.
    # If we find one, we try M-1, etc.
    # This is a binary search / linear search on M.
    
    # Let's implement a function that tries to solve for a specific scope M.
    
    # Since SA can be slow, we will use a time-boxed approach.
    # We will try scopes from best_scope down to min_possible_scope.
    
    # To speed up, we can use the current best_rows as a starting point for SA
    # and allow values to go up to M.
    
    # Let's refine the current solution to reduce the max value.
    
    # We will use a "cooling" schedule for SA.
    
    # State: list of rows
    # Move: Pick a random row, pick a random index, change value to a random valid value in [prev+1, next-1] (or [0, M] for first/last).
    # Cost: Number of duplicate differences.
    
    # We want to minimize Cost.
    
    # To reduce scope, we can add a penalty for large values.
    # Cost = num_conflicts * 100 + sum(max(0, val - M) for all vals)
    # But we want to strictly stay within M.
    
    # Alternative:
    # For a given M, run SA to find a valid configuration.
    # If found, set best_scope = M and continue with M-1.
    
    # This might be too slow if we restart SA from scratch for each M.
    # Instead, we will run SA with a dynamic M that decreases over time.
    
    # Let's try a simpler local search first: "Reduce Max"
    # While possible:
    #   Find the element with the max value.
    #   Try to decrease it to the largest possible value that doesn't create conflicts.
    #   If stuck, try to move other elements to free up space.
    
    # The current code does something similar but is inefficient.
    
    # Let's implement a more robust SA that minimizes conflicts + scope.
    
    # Objective function:
    # conflicts = number of pairs of differences that are equal (across all rows)
    # scope_penalty = sum(max(0, v - current_target) for all v)
    # We will decrease current_target over time.
    
    # Actually, let's just minimize conflicts, and separately track the min scope achieved.
    # But if we don't penalize scope, SA might find a valid solution with large scope.
    # We need to bias towards small values.
    
    # Let's use this cost:
    # cost = num_conflicts * 1000 + sum(v for v in all_values)
    # This encourages both validity and small values.
    
    # Let's implement this SA.
    
    rows = _simulated_annealing(rows, k, deadline, best_scope)
    
    if rows:
        current_scope = max(max(r) for r in rows)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in rows]
    
    return best_rows

def _greedy_construct(n_rows, k):
    """
    Simple greedy construction without scope limit.
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

def _initialize(n_rows, k, scope, deadline):
    """
    Try to initialize rows within [0, scope].
    Returns None if fails.
    """
    # Randomized greedy
    for _ in range(50):  # Try a few times
        if time.time() > deadline:
            return None
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try to pick a value > x and <= scope
                # Prefer smaller values
                candidates = []
                # Sample a few random candidates and the smallest available
                # To speed up, just try smallest first, then random
                # But we need to avoid conflicts.
                
                # Let's try a random walk
                # Pick a random value in [x+1, scope]
                # If conflict, try next
                
                # For efficiency, let's just use greedy with randomization
                # Try x+1, x+2, ... up to scope
                # But to allow backtracking, we need a more flexible search.
                
                # For initialization, let's just use a simple randomized greedy
                # that tries values in a biased random order.
                
                found = False
                # Try a few random candidates
                for _ in range(20):
                    cand = random.randint(x + 1, scope)
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
                
                if not found:
                    # Try linear scan
                    for cand in range(x + 1, scope + 1):
                        new = {cand - a for a in row}
                        if not new & used:
                            row.append(cand)
                            used |= new
                            x = cand
                            found = True
                            break
                
                if not found:
                    success = False
                    break
            rows.append(row)
        
        if success:
            return rows
            
    return None

def _simulated_annealing(rows, k, deadline, initial_scope):
    """
    Use Simulated Annealing to minimize conflicts and scope.
    """
    n_rows = len(rows)
    k_plus_1 = k + 1
    
    # Initial state
    current_rows = [r[:] for r in rows]
    
    # Calculate initial cost
    def get_conflicts(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        # Count duplicates
        seen = set()
        conflicts = 0
        for d in diffs:
            if d in seen:
                conflicts += 1
            else:
                seen.add(d)
        return conflicts
    
    def get_sum(rows):
        return sum(sum(row) for row in rows)
    
    def get_cost(rows):
        return get_conflicts(rows) * 1000 + get_sum(rows)
    
    current_cost = get_cost(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_cost = current_cost
    
    # Temperature schedule
    # Start high, decrease exponentially
    T = 100.0
    cooling = 0.99
    
    # Max time per step
    step_time = 0.01
    
    while time.time() < deadline:
        # Pick a random row and index
        i = random.randint(0, n_rows - 1)
        j = random.randint(1, k_plus_1)  # Don't change index 0 (always 0)
        
        old_val = current_rows[i][j]
        
        # Determine valid range for new value
        min_val = current_rows[i][j - 1] + 1
        max_val = current_rows[i][j + 1] - 1 if j + 1 < k_plus_1 else 1000  # No upper bound initially
        
        if min_val > max_val:
            continue
        
        # Pick a new value
        # Bias towards smaller values to reduce scope
        # Sample from a distribution that favors smaller values
        # e.g., uniform in [min_val, max_val] but with a penalty for large values
        
        # Let's just pick a random value in the range
        new_val = random.randint(min_val, max_val)
        
        if new_val == old_val:
            continue
        
        # Apply move
        current_rows[i][j] = new_val
        
        # Calculate new cost
        # To speed up, we can incrementally update conflicts, but for simplicity, let's recalculate
        # This might be slow, so let's optimize get_conflicts
        
        # Actually, recalculating is O(n_rows * k^2). For k=6, n_rows=7, it's 7*36=252 ops.
        # That's fast enough.
        
        new_cost = get_cost(current_rows)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [r[:] for r in current_rows]
        else:
            # Reject
            current_rows[i][j] = old_val
        
        # Cool down
        T *= cooling
        if T < 0.1:
            T = 0.1  # Don't go too low
    
    return best_rows

# EVOLVE-BLOCK-END
