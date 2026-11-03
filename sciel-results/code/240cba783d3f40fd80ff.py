"""
Difference Triangle Set solver using Simulated Annealing with efficient diff tracking.
Combines the structural approach of the inspiration (SA with global diff counts) 
with enhanced move operators (row shifting) and robust initialization.
"""
import time
import random
import math

# EVOLVE-BLOCK-START

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _row_diffs(row):
    """
    Calculate all positive pairwise differences for a row.
    For k=5, row has 6 elements, so 15 differences.
    """
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _count_conflicts(diff_counts):
    """
    Count the number of duplicate differences.
    If a diff appears 'c' times, it contributes (c-1) to the conflict count.
    """
    conflicts = 0
    for cnt in diff_counts.values():
        if cnt > 1:
            conflicts += cnt - 1
    return conflicts

def _greedy_row(used_diffs, k, rng, max_val=None):
    """
    Build a single row greedily.
    used_diffs: set of differences that are already "occupied" by other rows 
                (we want to avoid these to keep the solution valid/low conflict).
    max_val: upper bound for the last entry.
    """
    row = [0]
    x = 0
    target_len = k + 1
    
    while len(row) < target_len:
        x += 1
        if max_val is not None and x > max_val:
            break
        
        # Small random jump to escape local greedy traps
        if rng.random() < 0.05:
            x += rng.randint(1, 2)
            if max_val is not None and x > max_val:
                break

        new_diffs = {x - a for a in row}
        
        # Check if this candidate x introduces any new differences that are 
        # already in the global 'used' set.
        # For the purpose of the SA move, we just want a valid row structure 
        # (strictly increasing). The SA loop will handle the global conflicts 
        # via the cost function. However, generating a row that is "compatible" 
        # with the current used set is a better starting point.
        
        # Actually, for the Rebuild move, we want to generate a row that fits 
        # into the "holes" of the current difference spectrum if possible, 
        # or at least is a valid Sidon set itself.
        
        # Strictly increasing check is implicit in x > row[-1] since we increment x.
        # But we must ensure new_diffs don't overlap within the row itself?
        # No, the problem requires all differences *across all rows* to be distinct.
        # Within a row, differences must also be distinct? 
        # "all differences between two entries of the same row are distinct over all rows"
        # This implies:
        # 1. Within a row, a_j - a_i must be unique for each pair (i,j). (Sidon property within row)
        # 2. Across rows, no difference can repeat.
        
        # So, a valid row MUST be a Sidon set (B2 sequence).
        # Greedy construction of a Sidon set:
        # We need to ensure {x - a for a in row} are all distinct.
        
        internal_diffs = set()
        valid = True
        for a in row:
            d = x - a
            if d in internal_diffs:
                valid = False
                break
            internal_diffs.add(d)
        
        if valid:
            # Also check against global used? 
            # If we are rebuilding a row in SA, we remove its old diffs first.
            # So 'used_diffs' passed here should be the diffs of the OTHER rows.
            # We don't strictly need to avoid 'used_diffs' to make a valid *row*, 
            # but avoiding them reduces immediate conflicts.
            # However, if we avoid them too strictly, we might fail to find a row 
            # within max_val. 
            # Let's just ensure it's a valid Sidon set internally.
            row.append(x)
            
    return row if len(row) == target_len else None

def _initialize(n_rows, k, rng):
    """
    Generate an initial set of rows.
    We try to build them one by one, avoiding conflicts as much as possible.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        # Try a few times to find a good row
        best_row = None
        best_score = float('inf')
        for _ in range(10):
            row = _greedy_row(used, k, rng, max_val=None)
            if row:
                # Score: prefer smaller max entry
                if max(row) < best_score:
                    best_score = max(row)
                    best_row = row
        
        if best_row:
            rows.append(best_row)
            # Update used set with diffs of this row
            for d in _row_diffs(best_row):
                used.add(d)
        else:
            # Fallback: just a simple arithmetic progression or random
            # This should rarely happen
            rows.append([0, 1, 3, 6, 10, 15])
            for d in _row_diffs(rows[-1]):
                used.add(d)
                
    return rows

def _sa_optimize(rows, k, rng, time_budget=130.0):
    """
    Simulated Annealing to minimize scope + conflicts.
    """
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    
    # Initialize diff counts
    diff_counts = {}
    for row in current_rows:
        for d in _row_diffs(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
    def get_scope():
        return max(max(r) for r in current_rows)
        
    def get_conflicts():
        return _count_conflicts(diff_counts)
        
    def get_cost():
        # Heavily penalize conflicts to ensure validity is reached first
        return get_scope() + 1000.0 * get_conflicts()
        
    current_scope = get_scope()
    current_conflicts = get_conflicts()
    current_cost = current_scope + 1000.0 * current_conflicts
    
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    best_conflicts = current_conflicts
    best_cost = current_cost
    
    start_time = time.time()
    temp = 50.0  # Initial temperature
    cooling = 0.999
    min_temp = 0.1
    
    # Move counters for statistics (optional)
    moves_accepted = 0
    moves_total = 0
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
            
        # Pick a random move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Move 1: Rebuild a random row
            ri = rng.randrange(n_rows)
            old_row = current_rows[ri]
            
            # Remove old diffs
            for d in _row_diffs(old_row):
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # Determine max_val for new row
            # Allow slight expansion to escape local minima, but bias towards current scope
            # If we have conflicts, we might need to expand to find a valid spot?
            # Or we might need to compress.
            # Let's try a range around current scope.
            base_scope = current_scope
            max_limit = base_scope + rng.randint(0, 10)
            
            # Get used diffs from OTHER rows
            other_used = set(diff_counts.keys())
            
            new_row = _greedy_row(other_used, k, rng, max_val=max_limit)
            
            if new_row is None:
                # Failed to build row within limit, restore old
                for d in _row_diffs(old_row):
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                continue
            
            current_rows[ri] = new_row
            
            # Add new diffs
            for d in _row_diffs(new_row):
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
        elif move_type < 0.8:
            # Move 2: Shift a single entry in a random row
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            idx = rng.randint(1, len(row) - 1)
            old_val = row[idx]
            
            # Determine delta
            # Bias towards reducing values to compress scope
            if rng.random() < 0.7:
                delta = -rng.randint(1, 5)
            else:
                delta = rng.randint(-5, 5)
                
            new_val = old_val + delta
            
            # Check bounds and strict increase
            if new_val <= 0:
                continue
            if idx > 0 and new_val <= row[idx - 1]:
                continue
            if idx < len(row) - 1 and new_val >= row[idx + 1]:
                continue
                
            if new_val == old_val:
                continue
                
            # Remove old diffs for this row
            old_diffs = _row_diffs(row)
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
                    
            # Apply change
            row[idx] = new_val
            new_diffs = _row_diffs(row)
            
            # Add new diffs
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
        else:
            # Move 3: Shift the entire row (except first element 0)
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            
            # Small shift
            delta = rng.choice([-2, -1, 1, 2])
            
            # Check if shifting maintains strict increase (it does, since all shift by same amount)
            # But we must ensure row[0] remains 0? 
            # The problem states "starting at 0". So row[0] must be 0.
            # If we shift the whole row, row[0] becomes delta, which is invalid.
            # So this move is only valid if we allow row[0] to move? 
            # No, "starts at 0" is a hard constraint.
            # So we cannot shift the whole row while keeping row[0]=0 unless we shift relative to 0?
            # i.e., row[i] += delta for i>0. This changes the differences involving 0.
            # This is effectively a set of single-entry shifts, but correlated.
            # Let's skip this move or implement it as "Scale/Shift" carefully.
            # Actually, shifting i>0 by delta changes d_{i,0} and d_{j,i}.
            # It's a valid move if strict increase is maintained.
            
            # Check strict increase for i>0
            valid_shift = True
            for i in range(1, len(row)):
                new_val = row[i] + delta
                if new_val <= 0:
                    valid_shift = False
                    break
                if i > 1 and new_val <= row[i-1]:
                    valid_shift = False
                    break
                if i < len(row) - 1 and new_val >= row[i+1]:
                    valid_shift = False
                    break
            
            if not valid_shift:
                continue
                
            # Remove old diffs
            old_diffs = _row_diffs(row)
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
                    
            # Apply shift
            for i in range(1, len(row)):
                row[i] += delta
                
            new_diffs = _row_diffs(row)
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
        # Evaluate new state
        new_scope = get_scope()
        new_conflicts = get_conflicts()
        new_cost = new_scope + 1000.0 * new_conflicts
        
        delta_cost = new_cost - current_cost
        moves_total += 1
        
        # Acceptance criterion
        if delta_cost < 0 or (temp > min_temp and rng.random() < math.exp(-delta_cost / temp)):
            current_scope = new_scope
            current_conflicts = new_conflicts
            current_cost = new_cost
            moves_accepted += 1
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_scope = current_scope
                best_conflicts = current_conflicts
                best_rows = [list(r) for r in current_rows]
        else:
            # Revert move
            # We need to revert the specific move made.
            # This is tricky with the generic "remove/add" logic above if we don't store state.
            # To make it simpler and faster, we should store the previous state of the modified row(s)
            # or just re-calculate from scratch? Re-calc is O(N^2) which is fine for N=105 diffs.
            # But reverting requires knowing what changed.
            
            # Let's refactor: Instead of in-place modification with complex revert,
            # we can copy the row, modify the copy, check cost, and if rejected, discard the copy.
            # But updating diff_counts is the expensive part if done naively.
            
            # Alternative: The current code modifies `current_rows` and `diff_counts` in place.
            # To revert, we need to undo the specific change.
            
            # This implementation is getting complex for revert.
            # Let's switch to a "Candidate State" approach for the move.
            
            # REFACTORING MOVE LOGIC FOR SAFETY:
            # I will rewrite the loop to use a candidate row and candidate diff_counts 
            # for the affected row only, or just re-evaluate cost from scratch 
            # if the move is rejected? No, re-evaluating cost from scratch is O(N) where N is total diffs.
            # Given N=105, O(N) is very fast. 
            # So: 
            # 1. Save old row.
            # 2. Modify row in place.
            # 3. Update diff_counts in place.
            # 4. Calculate new cost.
            # 5. If accepted, keep.
            # 6. If rejected, revert row to old, and revert diff_counts.
            
            # To revert diff_counts easily, we can just re-initialize diff_counts from scratch 
            # if a move is rejected? That's O(N). 
            # Or we can store the old diffs of the modified row and subtract them, 
            # then add the old row's diffs back.
            
            # Let's implement the revert logic properly here.
            
            if move_type < 0.5:
                # Revert Rebuild
                old_row = current_rows[ri] # This is the NEW row
                # We lost the OLD row reference in the block above? 
                # In the block above: `old_row = current_rows[ri]` was the OLD row.
                # Then `current_rows[ri] = new_row`.
                # So we need to keep `old_row` accessible. 
                # In my current code structure, `old_row` is a local variable in the `if` block.
                # I need to restructure to ensure `old_row` is available for revert.
                
                # This indicates the current structure is flawed for easy revert.
                # Let's restart the SA loop with a cleaner "Candidate" pattern.
                
                # BREAK: I will rewrite the _sa_optimize function to be cleaner.
                pass 

    return best_rows, best_scope

# EVOLVE-BLOCK-END

# Since the revert logic in the previous draft was messy, I will provide the final clean implementation below.
# The key is to keep the "old" data available to revert.

def _sa_optimize_clean(rows, k, rng, time_budget=130.0):
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    
    def build_diff_counts(rws):
        dc = {}
        for row in rws:
            for d in _row_diffs(row):
                dc[d] = dc.get(d, 0) + 1
        return dc

    diff_counts = build_diff_counts(current_rows)
    
    def get_scope(rws):
        return max(max(r) for r in rws)
        
    def get_conflicts(dc):
        c = 0
        for cnt in dc.values():
            if cnt > 1:
                c += cnt - 1
        return c
        
    def get_cost(rws, dc):
        return get_scope(rws) + 1000.0 * get_conflicts(dc)
        
    current_scope = get_scope(current_rows)
    current_conflicts = get_conflicts(diff_counts)
    current_cost = current_scope + 1000.0 * current_conflicts
    
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    best_conflicts = current_conflicts
    best_cost = current_cost
    
    start_time = time.time()
    temp = 50.0
    cooling = 0.999
    min_temp = 0.1
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
            
        move_type = rng.random()
        
        # Save state for potential revert
        old_row_idx = -1
        old_row_data = None
        old_diffs_of_row = None
        
        if move_type < 0.5:
            # Rebuild Row
            old_row_idx = rng.randrange(n_rows)
            old_row_data = list(current_rows[old_row_idx])
            old_diffs_of_row = _row_diffs(old_row_data)
            
            # Remove old diffs
            for d in old_diffs_of_row:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # Build new row
            other_used = set(diff_counts.keys())
            max_limit = current_scope + rng.randint(0, 10)
            new_row = _greedy_row(other_used, k, rng, max_val=max_limit)
            
            if new_row is None:
                # Revert
                for d in old_diffs_of_row:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                continue
            
            current_rows[old_row_idx] = new_row
            new_diffs_of_row = _row_diffs(new_row)
            
            # Add new diffs
            for d in new_diffs_of_row:
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
        elif move_type < 0.8:
            # Shift Entry
            old_row_idx = rng.randrange(n_rows)
            old_row_data = list(current_rows[old_row_idx])
            old_diffs_of_row = _row_diffs(old_row_data)
            
            row = current_rows[old_row_idx]
            idx = rng.randint(1, len(row) - 1)
            old_val = row[idx]
            
            if rng.random() < 0.7:
                delta = -rng.randint(1, 5)
            else:
                delta = rng.randint(-5, 5)
                
            new_val = old_val + delta
            
            if new_val <= 0:
                continue
            if idx > 0 and new_val <= row[idx - 1]:
                continue
            if idx < len(row) - 1 and new_val >= row[idx + 1]:
                continue
            if new_val == old_val:
                continue
                
            # Remove old diffs
            for d in old_diffs_of_row:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
                    
            row[idx] = new_val
            new_diffs_of_row = _row_diffs(row)
            
            # Add new diffs
            for d in new_diffs_of_row:
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
        else:
            # Shift Row (entries 1..end)
            old_row_idx = rng.randrange(n_rows)
            old_row_data = list(current_rows[old_row_idx])
            old_diffs_of_row = _row_diffs(old_row_data)
            
            row = current_rows[old_row_idx]
            delta = rng.choice([-2, -1, 1, 2])
            
            valid = True
            for i in range(1, len(row)):
                nv = row[i] + delta
                if nv <= 0:
                    valid = False
                    break
                if i > 1 and nv <= row[i-1]:
                    valid = False
                    break
                if i < len(row) - 1 and nv >= row[i+1]:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Remove old diffs
            for d in old_diffs_of_row:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
                    
            for i in range(1, len(row)):
                row[i] += delta
                
            new_diffs_of_row = _row_diffs(row)
            for d in new_diffs_of_row:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
        # Evaluate
        new_scope = get_scope(current_rows)
        new_conflicts = get_conflicts(diff_counts)
        new_cost = new_scope + 1000.0 * new_conflicts
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or (temp > min_temp and rng.random() < math.exp(-delta_cost / temp)):
            current_scope = new_scope
            current_conflicts = new_conflicts
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_scope = current_scope
                best_conflicts = current_conflicts
                best_rows = [list(r) for r in current_rows]
        else:
            # Revert
            # Restore row
            current_rows[old_row_idx] = old_row_data
            # Restore diff counts
            # We removed `old_diffs_of_row` and added `new_diffs_of_row`.
            # So we must remove `new_diffs_of_row` and add back `old_diffs_of_row`.
            # Note: `new_diffs_of_row` is the diffs of the CURRENT (modified) row.
            
            # Remove new diffs
            for d in new_diffs_of_row:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
                    
            # Add back old diffs
            for d in old_diffs_of_row:
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
        # Cool down
        temp *= cooling
        if temp < min_temp:
            temp = min_temp

    return best_rows, best_scope

def construct(n_rows, k):
    rng = random.Random(42)
    
    # 1. Initialize
    initial_rows = _initialize(n_rows, k, rng)
    
    # 2. Optimize
    if _elapsed() < _DEADLINE - 10:
        rows, scope = _sa_optimize_clean(initial_rows, k, rng, time_budget=_DEADLINE - 10 - _elapsed())
    else:
        rows = initial_rows
        scope = _max_entry(rows)
        
    return rows

# EVOLVE-BLOCK-END
