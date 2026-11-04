import time
import random
import sys
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s kill
    
    # Total number of differences needed
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    
    # Lower bound on scope: we need at least total_diffs_needed distinct positive differences
    # The maximum difference in scope S is S, and there are S positive differences (1..S)
    # So we need S >= total_diffs_needed
    lower_bound = total_diffs_needed
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Try to find solutions with scope close to lower bound using backtracking
    # with constraint propagation and degree-based heuristic
    
    # We'll try scopes from lower_bound upward, but limit the range
    max_scope_to_try = lower_bound + 20
    
    for scope in range(lower_bound, max_scope_to_try + 1):
        if time.time() > deadline - 5:
            break
            
        result = _backtracking_with_propagation(n_rows, k, scope, start_time, deadline)
        if result is not None:
            actual_scope = max(max(row) for row in result)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = result
            if best_scope == lower_bound:
                break
        elif time.time() > deadline - 5:
            break
    
    # Phase 2: If backtracking didn't find good solutions, use randomized local search
    if best_rows is None or best_scope > lower_bound + 5:
        seed = 0
        while time.time() < deadline - 2:
            rows = _randomized_greedy_with_local_search(n_rows, k, seed, start_time, deadline)
            if rows is not None:
                scope = max(max(row) for row in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows
                    if best_scope == lower_bound:
                        break
            seed += 1
    
    # Phase 3: Fallback to simple greedy if nothing else worked
    if best_rows is None:
        best_rows = _simple_greedy(n_rows, k)
    
    return best_rows


def _backtracking_with_propagation(n_rows, k, max_scope, start_time, deadline):
    """
    Backtracking search with constraint propagation.
    Uses MRV (Minimum Remaining Values) and degree heuristics.
    """
    used_diffs = set()
    rows = []
    
    # Precompute: for each possible value v and each row position, 
    # what differences would be created
    # This is dynamic so we compute on the fly
    
    def backtrack(row_idx, col_idx, current_row):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        if col_idx == k + 1:
            # Row complete
            rows.append(list(current_row))
            success = backtrack(row_idx + 1, 1, [0])
            if success:
                return True
            rows.pop()
            return False
        
        # Find valid candidates for current position
        start_val = current_row[col_idx - 1] + 1
        end_val = min(max_scope, current_row[col_idx - 1] + max_scope)
        
        # Collect valid candidates with their "cost" (how many future constraints they block)
        valid_candidates = []
        for val in range(start_val, end_val + 1):
            new_diffs = set()
            conflict = False
            for i in range(col_idx):
                d = val - current_row[i]
                if d <= 0:
                    continue
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if not conflict and new_diffs:
                # Degree: count how many of these differences are "small" (more constrained)
                # Smaller differences are more likely to be needed by other rows
                degree = sum(1 for d in new_diffs if d <= max_scope // 2)
                valid_candidates.append((val, new_diffs, degree))
        
        if not valid_candidates:
            return False
        
        # Sort by degree (prefer lower degree = less constraining)
        valid_candidates.sort(key=lambda x: (x[2], x[0]))
        
        for val, new_diffs, _ in valid_candidates:
            used_diffs.update(new_diffs)
            current_row.append(val)
            
            if backtrack(row_idx, col_idx + 1, current_row):
                return True
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
            if time.time() > deadline:
                return False
        
        return False
    
    if backtrack(0, 1, [0]):
        return list(rows)
    return None


def _randomized_greedy_with_local_search(n_rows, k, seed, start_time, deadline):
    """
    Randomized greedy construction followed by local search to reduce scope.
    """
    rng = random.Random(seed)
    used = set()
    rows = []
    
    # Phase A: Build rows greedily with randomization
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            # Try a range of candidates, pick randomly from valid ones
            valid_xs = []
            x = last_x + 1
            # Limit search to reasonable range
            max_x = last_x + 50
            while x <= max_x and len(valid_xs) < 10:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                # Extend search
                x = last_x + 51
                while len(valid_xs) < 10:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
                if not valid_xs:
                    return None  # Failed
            
            # Pick randomly from valid candidates
            chosen_x = rng.choice(valid_xs)
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    # Phase B: Local search to reduce scope
    # Try to reduce the maximum element by moving elements to smaller positions
    scope = max(max(row) for row in rows)
    
    for iteration in range(50):  # Limited iterations
        if time.time() > deadline - 3:
            break
        
        improved = False
        for r in range(n_rows):
            for c in range(len(rows[r]) - 1, 0, -1):  # From last element backwards
                if rows[r][c] > scope:
                    continue  # Already at or below target
                
                # Try to move this element to a smaller position
                old_val = rows[r][c]
                for new_val in range(rows[r][c-1] + 1, old_val):
                    # Check if moving to new_val is valid
                    if _is_move_valid(rows, r, c, new_val, used):
                        # Perform the move
                        _apply_move(rows, r, c, old_val, new_val, used)
                        scope = max(max(row) for row in rows)
                        improved = True
                        break
                
                if not improved and time.time() > deadline - 3:
                    break
            if time.time() > deadline - 3:
                break
        
        if not improved:
            break
    
    return rows


def _is_move_valid(rows, row_idx, col_idx, new_val, used_diffs):
    """Check if moving rows[row_idx][col_idx] to new_val is valid."""
    row = rows[row_idx]
    
    # Check that new_val is greater than previous element
    if col_idx > 0 and new_val <= row[col_idx - 1]:
        return False
    
    # Check that new_val is less than next element (if exists)
    if col_idx < len(row) - 1 and new_val >= row[col_idx + 1]:
        return False
    
    # Compute differences that would change
    # Remove old differences involving col_idx
    old_val = row[col_idx]
    
    # Check all differences in the row involving position col_idx
    for i in range(len(row)):
        if i == col_idx:
            continue
        if i < col_idx:
            old_d = old_val - row[i]
            new_d = new_val - row[i]
            if new_d <= 0:
                return False
            # new_d must not be in used_diffs (excluding old_d which we're removing)
            if new_d != old_d and new_d in used_diffs:
                return False
        else:
            old_d = row[i] - old_val
            new_d = row[i] - new_val
            if new_d <= 0:
                return False
            # new_d must not be in used_diffs (excluding old_d which we're removing)
            if new_d != old_d and new_d in used_diffs:
                return False
    
    return True


def _apply_move(rows, row_idx, col_idx, old_val, new_val, used_diffs):
    """Apply a move: change rows[row_idx][col_idx] from old_val to new_val."""
    row = rows[row_idx]
    
    # Update used_diffs
    for i in range(len(row)):
        if i == col_idx:
            continue
        if i < col_idx:
            old_d = old_val - row[i]
            new_d = new_val - row[i]
            if old_d != new_d:
                used_diffs.discard(old_d)
                used_diffs.add(new_d)
        else:
            old_d = row[i] - old_val
            new_d = row[i] - new_val
            if old_d != new_d:
                used_diffs.discard(old_d)
                used_diffs.add(new_d)
    
    row[col_idx] = new_val


def _simple_greedy(n_rows, k):
    """Simple greedy: always pick the smallest valid value."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            new_diffs = {x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows
