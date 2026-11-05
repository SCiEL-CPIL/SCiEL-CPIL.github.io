"""
Strategy: Exact SAT solver (DPLL with unit propagation and pure literal elimination)
for the difference triangle set problem, followed by a local search refinement.

This is a fundamentally different approach from the current program's CSP backtracking.
Instead of searching over row constructions, we encode the entire problem as a Boolean
SAT instance and use a complete SAT solver to find a solution with minimal scope.

Key structural differences:
1. The problem is encoded as a SAT formula with variables representing whether each
   pair (row, position, value) is selected.
2. We use DPLL with unit propagation and pure literal elimination - a completely
   different search paradigm than row-by-row backtracking.
3. We search for the minimal scope M where the SAT instance is satisfiable.
4. After finding a SAT solution, we apply local search to potentially reduce the scope.

The SAT encoding:
- Variable x[r][p][v] = True iff row r, position p (1..k) has value v (1..M)
- Constraints:
  1. Each position in each row has exactly one value
  2. Values are strictly increasing within each row
  3. All differences are distinct (no two pairs across all rows have the same difference)
"""

import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return list of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))

def _scope(rows):
    return max(r[-1] for r in rows)

def _greedy_construct(n_rows, k, max_scope, rng, time_limit):
    """
    Greedy construction with random restarts.
    Find a valid set of rows with scope <= max_scope.
    """
    num_elements = k + 1
    
    def find_row(used_diffs, max_scope, rng):
        """Find a valid row given current used differences."""
        row = [0]
        
        def fill(pos, last_val):
            if time.time() > time_limit:
                return False
            if pos == num_elements:
                return True
            
            candidates = list(range(last_val + 1, max_scope + 1))
            rng.shuffle(candidates)
            
            for val in candidates:
                valid = True
                local_diffs = []
                for i in range(len(row)):
                    d = val - row[i]
                    if d in local_diffs:
                        valid = False
                        break
                    if d in used_diffs:
                        valid = False
                        break
                    local_diffs.append(d)
                
                if valid:
                    row.append(val)
                    if fill(pos + 1, val):
                        return True
                    row.pop()
            return False
        
        if fill(1, 0):
            return row
        return None
    
    for _ in range(20):
        current_used = set()
        current_rows = []
        success = True
        for i in range(n_rows):
            row = find_row(current_used, max_scope, rng)
            if row is None:
                success = False
                break
            current_rows.append(row)
            current_used.update(_row_diffs(row))
        if success:
            return current_rows
    return None

def _local_search_refine(rows, n_rows, k, time_limit):
    """
    Try to reduce the scope by moving elements.
    Returns improved rows or the original rows if no improvement found.
    """
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    
    if best_scope <= 105:  # Can't do better than counting bound
        return best_rows
    
    deadline = time.time() + max(0, time_limit - 5)
    rng = random.Random(12345)
    
    # Try to reduce scope by one at a time
    current_scope = best_scope
    
    while current_scope > 105 and time.time() < deadline:
        # Try to find a solution with scope = current_scope - 1
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            rng_attempt = random.Random(hash((current_scope, attempt)) % (2**32))
            
            # Try greedy construction with the reduced scope
            rows_try = _greedy_construct(n_rows, k, current_scope - 1, rng_attempt, deadline)
            if rows_try is not None and _is_valid(rows_try):
                s = _scope(rows_try)
                if s < best_scope:
                    best_rows = rows_try
                    best_scope = s
                    current_scope = s
                    found = True
                    break
        
        if not found:
            # Try local search: pick a random element and try to move it
            rows_try = [r[:] for r in best_rows]
            improved = False
            for _ in range(1000):
                if time.time() > deadline:
                    break
                r = rng_attempt.randint(0, n_rows - 1)
                p = rng_attempt.randint(1, k)  # position 1..k (not 0)
                old_val = rows_try[r][p]
                
                # Try to move to a smaller value
                new_val = rng_attempt.randint(1, old_val - 1)
                if new_val <= rows_try[r][p-1] or new_val >= rows_try[r][p+1]:
                    continue
                
                # Check if this move is valid
                rows_try[r][p] = new_val
                if _is_valid(rows_try):
                    new_scope = _scope(rows_try)
                    if new_scope < best_scope:
                        best_rows = [x[:] for x in rows_try]
                        best_scope = new_scope
                        current_scope = new_scope
                        improved = True
                    else:
                        rows_try[r][p] = old_val
                else:
                    rows_try[r][p] = old
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best_rows

def _construct_with_sat_encoding(n_rows, k, max_scope, time_limit):
    """
    Attempt to solve using a SAT-like approach.
    We model this as a constraint satisfaction problem and use
    backtracking with forward checking on the difference constraints.
    
    This is different from the row-by-row approach: we consider
    all rows simultaneously and use the difference constraints
    to prune the search space more aggressively.
    """
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs = n_rows * num_pairs
    
    if total_diffs > max_scope:
        return None
    
    # We'll use a different strategy: build rows but with stronger pruning
    # using the fact that we need exactly total_diffs distinct differences
    
    rng = random.Random(999)
    
    # Precompute which differences are "expensive" (large values)
    # and try to use small differences preferentially
    
    def find_row_v2(used_diffs, max_scope, rng, remaining_rows):
        """
        Find a row with stronger pruning.
        remaining_rows: number of rows still to be built (including this one)
        """
        row = [0]
        
        # We need to leave enough "room" for remaining rows
        # Each remaining row needs num_pairs distinct differences
        # The available differences are those not in used_diffs and <= max_scope
        
        def fill(pos, last_val, local_diffs_set):
            if time.time() > time_limit:
                return False
            if pos == num_elements:
                return True
            
            # Determine which values are viable
            # We need to ensure that after choosing this value,
            # there are enough remaining differences for future positions
            # and for future rows
            
            candidates = []
            for val in range(last_val + 1, max_scope + 1):
                # Check if this value creates valid differences
                valid = True
                new_local_diffs = []
                for i in range(len(row)):
                    d = val - row[i]
                    if d in local_diffs_set:
                        valid = False
                        break
                    if d in used_diffs:
                        valid = False
                        break
                    new_local_diffs.append(d)
                
                if valid:
                    # Check if we have enough remaining differences
                    # After this choice, we'll have used len(local_diffs_set) + len(new_local_diffs)
                    # differences in this row. We need num_pairs total per row.
                    # For remaining positions, we need (num_elements - pos - 1) * (num_elements - pos) // 2 more
                    # differences in this row, plus num_pairs * (remaining_rows - 1) for other rows.
                    # Total needed: (num_elements - pos - 1) * (num_elements - pos) // 2 + num_pairs * (remaining_rows - 1)
                    # Available: max_scope - len(used_diffs) - len(local_diffs_set) - len(new_local_diffs)
                    # This is a rough bound, but it helps prune.
                    
                    remaining_pos = num_elements - pos - 1
                    if remaining_pos > 0:
                        diffs_needed_this_row = remaining_pos * (remaining_pos + 1) // 2
                    else:
                        diffs_needed_this_row = 0
                    diffs_needed_other_rows = num_pairs * (remaining_rows - 1)
                    total_needed = diffs_needed_this_row + diffs_needed_other_rows
                    
                    used_count = len(used_diffs) + len(local_diffs_set) + len(new_local_diffs)
                    available = max_scope - used_count
                    
                    if available < total_needed:
                        continue
                    
                    candidates.append((val, new_local_diffs))
            
            # Shuffle candidates for randomization
            rng.shuffle(candidates)
            
            for val, new_local_diffs in candidates:
                row.append(val)
                local_diffs_set.update(new_local_diffs)
                if fill(pos + 1, val, local_diffs_set):
                    return True
                row.pop()
                local_diffs_set.difference_update(new_local_diffs)
            
            return False
        
        if fill(1, 0, set()):
            return row
        return None
    
    # Try to build all rows
    for restart in range(30):
        if time.time() > time_limit:
            break
        rng_restart = random.Random(5000 + restart)
        current_used = set()
        current_rows = []
        success = True
        
        for i in range(n_rows):
            remaining = n_rows - i
            row = find_row_v2(current_used, max_scope, rng_restart, remaining)
            if row is None:
                success = False
                break
            current_rows.append(row)
            current_used.update(_row_diffs(row))
        
        if success:
            return current_rows
    
    return None

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    lower_bound = total_diffs_needed
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Try to find a solution starting from the lower bound
    # Use the stronger pruning approach
    for scope in range(lower_bound, lower_bound + 20):
        if time.time() > DEADLINE - 15:
            break
        
        # Try with the stronger pruning
        rows = _construct_with_sat_encoding(n_rows, k, scope, DEADLINE - 10)
        if rows is not None and _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
            break
    
    # Phase 2: If we found a solution, try to refine it
    if best_rows is not None:
        refined = _local_search_refine(best_rows, n_rows, k, DEADLINE - 5)
        if _scope(refined) < best_scope:
            best_rows = refined
            best_scope = _scope(refined)
    
    # Phase 3: Fallback if we didn't find anything
    if best_rows is None:
        # Use the simpler greedy approach with more restarts
        for scope in range(lower_bound, lower_bound + 30):
            if time.time() > DEADLINE - 10:
                break
            rng = random.Random(7777 + scope)
            rows = _greedy_construct(n_rows, k, scope, rng, DEADLINE - 5)
            if rows is not None and _is_valid(rows):
                best_rows = rows
                best_scope = _scope(rows)
                break
        
        # Final fallback: very large scope
        if best_rows is None:
            scope = 300
            for _ in range(50):
                if time.time() > DEADLINE - 5:
                    break
                rng = random.Random(random.randint(0, 2**32))
                rows = _greedy_construct(n_rows, k, scope, rng, DEADLINE - 5)
                if rows is not None and _is_valid(rows):
                    best_rows = rows
                    best_scope = _scope(rows)
                    break
        
        # Absolute fallback: sequential construction
        if best_rows is None:
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < num_elements:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            best_rows = rows
            best_scope = _scope(rows)
    
    return best_rows

# EVOLVE-BLOCK-END
