import random
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    diffs_per_row = k * (k + 1) // 2
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        return False
                    used_diffs.add(d)
        return True

    # Helper to compute diffs for a single row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Phase 1: Initial Greedy Construction
    # This establishes a baseline. We construct rows one by one.
    # For each row, we start with 0 and greedily pick the smallest next number
    # that doesn't conflict with existing diffs in the row or previously placed rows.
    
    def greedy_construct():
        used_global = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_diffs = set()
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & current_diffs) and not (new_diffs & used_global):
                        row.append(x)
                        current_diffs |= new_diffs
                        used_global |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_max = get_max_entry(best_rows)
    
    # Phase 2: Local Search / Hill Climbing
    # We try to reduce the maximum entry by modifying rows.
    # Strategy: 
    # 1. Identify the row with the largest entry (or random row).
    # 2. Try to rebuild that row with a smaller scope, ensuring it doesn't conflict with other rows.
    # 3. If successful and the new max is lower, accept.
    
    # To make this efficient, we pre-compute the set of differences used by all OTHER rows
    # for a given row index. This allows us to check conflicts against the "environment" quickly.
    
    def get_other_diffs(rows, idx):
        others = set()
        for i, row in enumerate(rows):
            if i == idx:
                continue
            others |= row_diffs(row)
        return others

    # Backtracking search to find a valid row within a scope limit
    # that avoids a specific set of forbidden differences.
    def find_row_with_limit(scope_limit, forbidden_diffs, k_val, rng):
        # Returns a list of k_val+1 integers starting at 0, strictly increasing, max <= scope_limit
        # such that all internal diffs are distinct and do not intersect forbidden_diffs.
        
        # We use backtracking.
        # State: current row, current set of diffs
        # We want to fill k_val more elements.
        
        # To speed up, we can sort candidates? No, we must pick increasing integers.
        # We can randomize the order of trying candidates to escape local traps?
        # Actually, for a fixed scope limit, greedy smallest-first is often good for existence,
        # but randomization helps if greedy fails due to poor early choices.
        
        # Let's use a recursive backtracker with a time limit.
        
        def backtrack(row, current_diffs, remaining):
            if remaining == 0:
                return row
            
            last = row[-1]
            # Next number must be > last and <= scope_limit
            # Also, the new diffs {x - a for a in row} must be disjoint from current_diffs and forbidden_diffs
            
            # To add randomness, we can shuffle the range if it's not too large?
            # Or just try sequentially. Sequential is deterministic. 
            # Let's try a mix: mostly sequential, but if we fail, we might have made a bad choice earlier.
            # For now, simple backtracking.
            
            # Optimization: The smallest possible next number is last + 1.
            # The largest possible is scope_limit.
            
            # Heuristic: Try smaller numbers first to keep the row compact.
            for x in range(last + 1, scope_limit + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & current_diffs:
                    continue
                if new_diffs & forbidden_diffs:
                    continue
                
                # Check if we can complete the rest of the row
                # Simple check: are there enough numbers left?
                # If x is too large, we might not have room for 'remaining' more numbers.
                # Max gap is 1. Min space needed is 'remaining'.
                # So x + remaining <= scope_limit is a necessary condition.
                if x + remaining > scope_limit:
                    break
                
                # Recursive step
                result = backtrack(row + [x], current_diffs | new_diffs, remaining - 1)
                if result:
                    return result
            
            return None

        # Initial call
        # Row starts with [0]. Diffs is empty. Remaining is k_val.
        # Note: 0 is in the row, so diffs with 0 are just the values themselves.
        return backtrack([0], set(), k_val)

    # Main loop for local search
    # We will iterate, trying to improve the worst row.
    
    # To make it more robust, we can also try swapping rows?
    # Or just rebuilding one row at a time.
    
    # Let's define a function to attempt to improve the set by rebuilding one row.
    def try_improve(rows, max_limit):
        # max_limit is the current best_max - 1 (we want to go below best_max)
        # We pick a random row to rebuild.
        # We calculate the forbidden diffs (all diffs from other rows).
        # We try to find a new row with scope <= max_limit that fits.
        
        # To be efficient, we should target the row with the largest current scope,
        # or a random row if we want to explore.
        # Let's target the row with the largest scope first, as it's the bottleneck.
        
        # Find index of row with max scope
        max_scope_idx = max(range(len(rows)), key=lambda i: max(rows[i]))
        
        # But maybe that row is "easy" to fit and another is "hard"?
        # Let's try rebuilding the worst row.
        
        forbidden = get_other_diffs(rows, max_scope_idx)
        
        # Try to find a row with scope <= max_limit
        new_row = find_row_with_limit(max_limit, forbidden, k, random)
        
        if new_row:
            # Check if it actually reduces the max
            old_max = max(rows[max_scope_idx])
            new_max = max(new_row)
            if new_max < old_max:
                # Update rows
                new_rows = [row[:] for row in rows]
                new_rows[max_scope_idx] = new_row
                return new_rows
        return None

    # Iterative improvement
    # We start with the greedy solution.
    # We try to reduce the max scope step by step.
    
    current_rows = best_rows
    current_max = best_max
    
    # We will try to lower the limit.
    # The limit is the maximum allowed entry.
    # We try limits from current_max - 1 down to the theoretical lower bound.
    
    # Lower bound: n_rows * k * (k+1) / 2 ? No, that's total diffs.
    # The max entry M must be at least the number of distinct differences.
    # Total diffs = n_rows * k * (k+1) / 2.
    # So M >= total_diffs.
    
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # We will iterate limits downwards.
    # But backtracking can be slow.
    # Let's use a time-sliced approach.
    
    # Sort rows by scope descending? No, we keep them as is.
    
    # Let's try a loop:
    # For limit in range(current_max - 1, lower_bound - 1, -1):
    #    if time > deadline: break
    #    try to find a configuration with max <= limit
    
    # However, "try to find a configuration" is hard if we just rebuild one row.
    # Rebuilding one row might not be enough if the conflict structure is tight.
    
    # Alternative: Simulated Annealing on the set of rows?
    # State: list of rows.
    # Move: Pick a random row, replace it with a random valid Golomb ruler (scope < current max).
    # Cost: Max scope.
    
    # This is a different family: SA / Metropolis.
    
    # Let's implement a simple SA loop.
    
    def random_golomb_row(scope_limit, rng):
        """Generate a random Golomb ruler with max entry <= scope_limit."""
        # Greedy with randomization
        row = [0]
        current_diffs = set()
        
        # We need to pick k more numbers.
        # To ensure it's a Golomb ruler, we check diffs.
        # We pick the next number randomly from [last+1, scope_limit].
        
        # To make it more likely to succeed, we can use a "lookahead" or just retry.
        # For simplicity, let's use a backtracker that picks randomly.
        
        def backtrack(row, current_diffs, remaining):
            if remaining == 0:
                return row
            
            last = row[-1]
            candidates = list(range(last + 1, scope_limit + 1))
            rng.shuffle(candidates)
            
            for x in candidates:
                if x + remaining > scope_limit:
                    continue # Not enough room
                
                new_diffs = {x - a for a in row}
                if new_diffs & current_diffs:
                    continue
                
                result = backtrack(row + [x], current_diffs | new_diffs, remaining - 1)
                if result:
                    return result
            return None
        
        return backtrack([0], set(), k)

    # SA Parameters
    T_start = 100.0
    T_end = 1.0
    alpha = 0.995 # Cooling rate
    
    # We need to track the global diffs set for quick conflict checking?
    # In SA, we replace one row at a time.
    # So we maintain `all_diffs` and `row_diffs_list`.
    
    # Initialize SA state
    sa_rows = [row[:] for row in current_rows]
    sa_row_diffs = [row_diffs(row) for row in sa_rows]
    all_diffs = set()
    for d in sa_row_diffs:
        all_diffs |= d
    
    sa_max = max(max(row) for row in sa_rows)
    best_sa_rows = [row[:] for row in sa_rows]
    best_sa_max = sa_max
    
    T = T_start
    step = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 5.0:
        # Pick a random row to perturb
        i = random.randint(0, n_rows - 1)
        old_row = sa_rows[i]
        old_diffs = sa_row_diffs[i]
        old_max = max(old_row)
        
        # Generate a new row
        # We want to try to reduce the max.
        # If old_max is the global max, we should try a smaller scope.
        # Otherwise, we can try similar scope.
        
        # Target scope: slightly less than current global max, or same.
        # Let's try to generate a row with scope <= sa_max.
        # To encourage improvement, if old_max == sa_max, try scope < sa_max.
        
        target_scope = sa_max
        if old_max == sa_max and sa_max > lower_bound:
            # Try to shrink
            target_scope = sa_max - random.randint(1, 5)
            if target_scope < k + 1:
                target_scope = k + 1
        
        new_row = random_golomb_row(target_scope, random)
        
        if new_row is None:
            # Failed to generate, try again with larger scope? Or skip.
            # Let's skip this step.
            continue
        
        new_diffs = row_diffs(new_row)
        new_max = max(new_row)
        
        # Check conflict with other rows
        # all_diffs contains diffs of ALL rows.
        # We need to check if new_diffs intersects with (all_diffs - old_diffs).
        other_diffs = all_diffs - old_diffs
        
        if new_diffs & other_diffs:
            # Conflict
            continue
        
        # Calculate new global max
        # If we replace row i, the new global max is max(max of other rows, new_max)
        # We can precompute max of other rows?
        # For n_rows=4, it's cheap to just compute.
        
        new_sa_max = max(max(row) for j, row in enumerate(sa_rows) if j != i)
        if new_max > new_sa_max:
            new_sa_max = new_max
        
        delta = new_sa_max - sa_max
        
        # Acceptance probability
        if delta < 0:
            accept = True
        else:
            if T > 0:
                prob = math.exp(-delta / T)
                accept = random.random() < prob
            else:
                accept = False
        
        if accept:
            # Update state
            # Remove old diffs from all_diffs
            all_diffs -= old_diffs
            # Add new diffs
            all_diffs |= new_diffs
            
            sa_rows[i] = new_row
            sa_row_diffs[i] = new_diffs
            sa_max = new_sa_max
            
            if sa_max < best_sa_max:
                best_sa_max = sa_max
                best_sa_rows = [row[:] for row in sa_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        else:
            no_improve_count += 1
            
        # Cooling
        T *= alpha
        if T < T_end:
            T = T_end
            
        step += 1
        if no_improve_count > max_no_improve:
            # Restart or break?
            # Let's break if we've been stuck for a long time and T is low.
            if T <= T_end:
                break
            # Otherwise, keep going, maybe random restart?
            # Simple: just keep going, SA might escape.
            pass

    # Compare SA best with initial greedy best
    if best_sa_max < best_max:
        best_max = best_sa_max
        best_rows = best_sa_rows

    return best_rows
# EVOLVE-BLOCK-END
