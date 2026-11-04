"""Improved program using Simulated Annealing with bitset optimization."""
import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    # Theoretical lower bound for M: roughly sqrt(2 * total_diffs) is a loose bound, 
    # but M must be at least k. A tighter bound is derived from counting.
    # For n=4, k=7, total_diffs = 4 * 28 = 112. 
    # If max diff is M, there are M possible diffs. So M >= 112.
    # Actually, the diffs are distinct. So we need at least 112 distinct diffs.
    # The largest diff is M. The available diffs are 1..M. So M >= 112.
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)

    def count_conflicts(rows):
        """Count the number of duplicate differences across all rows."""
        used = 0
        conflicts = 0
        for row in rows:
            row_diffs = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if (used >> d) & 1:
                        conflicts += 1
                    used |= (1 << d)
                    row_diffs |= (1 << d)
            # Check internal conflicts (though rows are built to be valid, 
            # during moves we might temporarily break them if we aren't careful, 
            # but we enforce validity in moves. 
            # Wait, if we enforce validity in moves, internal conflicts are 0.
            # So we only need to check global conflicts.
        return conflicts

    # 1. Initial Solution: Greedy with randomization
    def greedy_solution(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                if not conflict:
                    row.append(x)
                    used |= new_diffs
            if randomize and len(row) > 2:
                # Perturb slightly
                if random.random() < 0.5:
                    i1, i2 = random.sample(range(1, len(row)), 2)
                    row[i1], row[i2] = row[i2], row[i1]
                    row = [0] + sorted(row[1:])
            rows.append(row)
        return rows

    # Initial best
    best_rows = greedy_solution(randomize=True, seed=42)
    best_max = get_max_entry(best_rows)
    best_conflicts = count_conflicts(best_rows)
    
    # If greedy didn't find a valid solution (conflicts > 0), we need to be more careful.
    # But greedy with bitsets usually finds valid ones quickly for small n, k.
    # Let's verify validity.
    def is_valid(rows):
        used = 0
        for row in rows:
            if len(row) != k + 1: return False
            if row[0] != 0: return False
            for i in range(len(row)-1):
                if row[i] >= row[i+1]: return False
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if (used >> d) & 1:
                        return False
                    used |= (1 << d)
        return True

    if not is_valid(best_rows):
         # Fallback to a more robust greedy or simple backtracking if greedy failed
         # For now, assume greedy works. If not, SA will try to fix it.
         pass

    # 2. Simulated Annealing
    # State: rows
    # Cost: M + lambda * conflicts
    # Moves: 
    #   A. Rebuild a row: Remove diffs, rebuild greedily to minimize span or fit M.
    #   B. Shift an element: Pick (r, i), try new_val.
    
    def get_cost(rows):
        m = get_max_entry(rows)
        c = count_conflicts(rows)
        # We want to minimize M primarily, but conflicts must be 0.
        # If conflicts > 0, it's invalid. 
        # We can use a large penalty for conflicts.
        return m + 1000 * c

    def rebuild_row(rows, row_idx, max_val_limit):
        """
        Rebuild row at row_idx to be valid and have max entry <= max_val_limit.
        Returns new row or None if impossible.
        """
        # Remove current row's diffs from global usage? 
        # In SA, we usually compute cost from scratch or maintain a global state.
        # Maintaining global state is faster. Let's maintain `used_diffs` bitset.
        pass

    # To maintain speed, we will implement SA with a global `used_diffs` bitset.
    # This requires careful undo/redo or recomputation.
    # Given the small size (4 rows, 8 elements), recomputation is fast enough for ~10k-50k iters.
    # But bitset ops are so fast, let's try to maintain state.
    
    # Let's stick to a simpler SA that recomputes cost but uses bitsets for the cost calculation.
    # Actually, maintaining `used_diffs` is better.
    
    current_rows = [list(r) for r in best_rows]
    current_max = best_max
    current_conflicts = best_conflicts
    current_cost = get_cost(current_rows)
    
    # Precompute row diffs for fast removal?
    # Let's just use the bitset approach in the move generation.
    
    def generate_move(rows):
        """
        Generates a new state (rows) by applying a move.
        Returns (new_rows, new_cost, move_info)
        """
        # Move 1: Rebuild a random row
        if random.random() < 0.5:
            r_idx = random.randint(0, n_rows - 1)
            # Remove old row
            old_row = rows[r_idx]
            # We need to know which diffs were used by this row to remove them from `used`?
            # Since we don't maintain `used` globally in this function, we'll just compute new cost.
            # To be efficient, we can try to rebuild the row to be "tight".
            
            new_row = [0]
            # Try to fill the row with small numbers
            # Greedy fill
            # We need to check against OTHER rows.
            # Let's extract diffs of other rows.
            other_diffs = 0
            for i in range(n_rows):
                if i == r_idx: continue
                row = rows[i]
                for a in range(len(row)):
                    for b in range(a+1, len(row)):
                        other_diffs |= (1 << (row[b] - row[a]))
            
            # Now build new_row such that its diffs don't conflict with other_diffs
            # and are internally distinct.
            # We want to minimize the max element.
            # Greedy: pick smallest x > last such that x - prev elements are not in other_diffs and not in new_row_diffs.
            
            new_row_diffs = 0
            x = 0
            while len(new_row) < k + 1:
                x += 1
                conflict = False
                for a in new_row:
                    d = x - a
                    if (other_diffs >> d) & 1:
                        conflict = True
                        break
                    if (new_row_diffs >> d) & 1:
                        conflict = True
                        break
                    # Also check if d is too large? No, we want to minimize max.
                if not conflict:
                    new_row.append(x)
                    for a in new_row[:-1]:
                        new_row_diffs |= (1 << (x - a))
                    # Optimization: if x is already larger than current max, we might stop?
                    # No, we must complete the row.
            
            # Check if this row is valid
            # It is by construction.
            # Does it have fewer conflicts? 
            # It has 0 conflicts with other rows.
            # But what about its own internal conflicts? Checked.
            # So this move always results in a row with 0 conflicts with others.
            # However, the *other* rows might have conflicts with each other.
            # The total conflict count will be the conflicts among the other rows.
            
            new_rows = [list(r) for r in rows]
            new_rows[r_idx] = new_row
            return new_rows, None # Cost will be computed later
            
        else:
            # Move 2: Shift an element in a row
            r_idx = random.randint(0, n_rows - 1)
            row = list(rows[r_idx])
            if len(row) <= 1:
                return rows, None
            i_idx = random.randint(1, len(row) - 1) # Don't move 0
            old_val = row[i_idx]
            
            # Try to find a new value for row[i_idx]
            # Constraints:
            # 1. row[i_idx-1] < new_val < row[i_idx+1] (if i_idx < len-1)
            # 2. new_val > row[i_idx-1]
            # 3. If i_idx == len-1, new_val > row[i_idx-1]
            
            min_val = row[i_idx-1] + 1
            max_val = row[i_idx+1] - 1 if i_idx < len(row)-1 else 1000 # Large bound
            
            # Try values near old_val
            search_range = 5
            candidates = []
            for v in range(max(min_val, old_val - search_range), min(max_val, old_val + search_range) + 1):
                if v == old_val: continue
                candidates.append(v)
            random.shuffle(candidates)
            
            if not candidates:
                # Try a random valid value
                if min_val < max_val:
                    candidates = [random.randint(min_val, max_val)]
                else:
                    return rows, None
                    
            for new_val in candidates:
                # Check validity of the row with new_val
                test_row = row[:]
                test_row[i_idx] = new_val
                # Check increasing
                if not all(test_row[j] < test_row[j+1] for j in range(len(test_row)-1)):
                    continue
                # Check internal diffs
                internal_diffs = set()
                valid = True
                for a in range(len(test_row)):
                    for b in range(a+1, len(test_row)):
                        d = test_row[b] - test_row[a]
                        if d in internal_diffs:
                            valid = False
                            break
                        internal_diffs.add(d)
                if not valid:
                    continue
                
                # If valid, this is a candidate move
                new_rows = [list(r) for r in rows]
                new_rows[r_idx] = test_row
                return new_rows, None

    # SA Loop
    T = 100.0
    cooling = 0.999
    min_T = 0.1
    iter_count = 0
    
    # We need to compute cost efficiently.
    # Let's precompute the bitset of diffs for each row to speed up removal/addition?
    # Or just recompute global cost.
    
    while time.time() < deadline:
        iter_count += 1
        if T < min_T:
            break
            
        new_rows, _ = generate_move(current_rows)
        
        # Compute new cost
        new_max = get_max_entry(new_rows)
        new_conflicts = count_conflicts(new_rows)
        new_cost = new_max + 1000 * new_conflicts
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_rows = new_rows
            current_max = new_max
            current_conflicts = new_conflicts
            current_cost = new_cost
            
            # Update best if valid
            if new_conflicts == 0:
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in current_rows]
        
        T *= cooling
        if T < min_T:
             T = min_T
             
        # Check time
        if time.time() > deadline - 5:
            break

    # Final check
    if not is_valid(best_rows):
        # If SA failed to find a valid solution, fall back to greedy
        best_rows = greedy_solution()
        best_max = get_max_entry(best_rows)
        
    return best_rows

# EVOLVE-BLOCK-END
