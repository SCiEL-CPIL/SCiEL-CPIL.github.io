import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit
    
    n = k + 1
    # n_rows=7, k=5 => n=6
    # Differences per row: 6*5/2 = 15
    # Total differences needed: 7 * 15 = 105
    
    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def is_valid_row(row):
        if len(row) != n:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diffs = row_diffs(row)
        expected = len(row) * (len(row) - 1) // 2
        return len(diffs) == expected

    def greedy_construct(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < n:
                found = False
                # Try values starting from last+1 up to a reasonable limit
                # The limit can be dynamic, but 200 is safe for start
                for x in range(row[-1] + 1, 200):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    # Get an initial valid solution
    initial_rows = None
    for s in range(100):
        initial_rows = greedy_construct(s)
        if initial_rows:
            break
    
    if not initial_rows:
        # Fallback: very slow but guaranteed if scope is large enough
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < n:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update(new_diffs)
            rows.append(row)
        initial_rows = rows

    current_rows = [list(r) for r in initial_rows]
    current_scope = max(max(r) for r in current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope

    # Precompute row diffs for current solution
    row_diff_sets = [row_diffs(r) for r in current_rows]
    used_diffs = set()
    for dset in row_diff_sets:
        used_diffs.update(dset)

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        all_diffs = set()
        for r in rows:
            if not is_valid_row(r):
                return False
            dset = row_diffs(r)
            if all_diffs & dset:
                return False
            all_diffs.update(dset)
        return True

    # Simulated Annealing
    # State: rows
    # Move: Pick a row, pick an index, change value to a nearby value
    # Cost: Scope (max element). We want to minimize this.
    # However, minimizing scope directly is hard because we must maintain validity.
    # We can use a penalty-based cost or just try to reduce the max element of the row containing the global max.
    
    # A better approach for DTS: 
    # 1. Pick a row.
    # 2. Pick an element (index > 0).
    # 3. Try to decrease it or move it to a new position that doesn't conflict.
    # If we just decrease the max element, we might create conflicts.
    
    # Let's use a standard SA loop:
    # 1. Generate a neighbor by changing one element in one row.
    # 2. Check if the new row is valid and doesn't conflict with other rows.
    # 3. Calculate new scope.
    # 4. Accept if better or with probability exp(-delta/T).
    
    # To speed up neighbor generation, we can focus on the row with the current maximum.
    
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    
    # We will iterate until time runs out
    while time.time() - start_time < time_limit and T > min_T:
        # Pick a row to modify
        # Bias towards rows with larger elements
        row_indices = list(range(n_rows))
        # Weight by the max element of the row
        weights = [max(current_rows[i]) + 1 for i in row_indices]
        total_w = sum(weights)
        r = random.uniform(0, total_w)
        acc = 0
        chosen_row_idx = 0
        for i, w in zip(row_indices, weights):
            acc += w
            if r <= acc:
                chosen_row_idx = i
                break
        
        row = current_rows[chosen_row_idx]
        
        # Pick an index to change. 
        # Changing the last element is most likely to reduce scope.
        # Changing intermediate elements might allow reducing the last element later.
        # Let's pick a random index from 1 to n-1.
        idx = random.randint(1, n - 1)
        
        old_val = row[idx]
        
        # Try to find a new value
        # We want to try values close to old_val, but also potentially smaller to reduce scope.
        # If idx is the last one, we definitely want to try smaller values.
        
        # Determine range for new value
        lo = row[idx - 1] + 1
        # hi can be old_val + some delta, or if we are trying to reduce, just old_val - 1?
        # Let's try a window around old_val, and also specifically try to reduce if it's the max.
        
        # If this element is part of the global max, prioritize reduction.
        # For simplicity, let's try a random new value in [lo, old_val + 10]
        # But we must ensure the row remains strictly increasing.
        
        # Actually, a good move is to try to "slide" the element to a free spot.
        # Let's try a few candidates.
        
        candidates = []
        # Try decreasing
        for delta in range(1, 11):
            nv = old_val - delta
            if nv >= lo:
                candidates.append(nv)
            else:
                break
        # Try increasing slightly (to allow future decreases or resolve conflicts)
        for delta in range(1, 11):
            nv = old_val + delta
            candidates.append(nv)
            
        # Shuffle candidates
        random.shuffle(candidates)
        
        move_made = False
        for nv in candidates:
            # Check if nv is valid for the row position
            if nv <= row[idx - 1]:
                continue
            if idx < n - 1 and nv >= row[idx + 1]:
                continue
            
            # Construct new row
            new_row = list(row)
            new_row[idx] = nv
            
            # Check internal validity (distinct diffs)
            new_diffs = row_diffs(new_row)
            if len(new_diffs) != n * (n - 1) // 2:
                continue
            
            # Check conflict with other rows
            # We need to check if any new_diffs are in (used_diffs - old_row_diffs)
            other_diffs = used_diffs - row_diff_sets[chosen_row_idx]
            if new_diffs & other_diffs:
                continue
            
            # Move is valid!
            # Calculate new scope
            new_rows = [list(r) for r in current_rows]
            new_rows[chosen_row_idx] = new_row
            new_scope = get_scope(new_rows)
            
            delta_scope = new_scope - current_scope
            
            # Acceptance criterion
            if delta_scope < 0 or random.random() < pow(2.71828, -delta_scope / T):
                # Update state
                # Remove old diffs from used_diffs
                used_diffs.difference_update(row_diff_sets[chosen_row_idx])
                # Add new diffs
                used_diffs.update(new_diffs)
                
                # Update current rows and diffs
                current_rows[chosen_row_idx] = new_row
                row_diff_sets[chosen_row_idx] = new_diffs
                current_scope = new_scope
                
                # Update best
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
                    # Optional: break early if we hit the bound?
                    # The counting bound is 105. If we hit 105, we can stop.
                    if best_scope <= 105:
                        break
                
                move_made = True
                break
        
        if best_scope <= 105:
            break
            
        # Cool down
        T *= cooling

    # Final validation
    if not validate(best_rows):
        # This should not happen if our update logic is correct, but just in case
        # Fall back to initial_rows
        best_rows = [list(r) for r in initial_rows]

    return best_rows

# For testing/debugging if run directly
if __name__ == "__main__":
    # The evaluator will call construct(7, 5)
    # We can test with smaller parameters first if needed, but the contract is fixed.
    # Let's just ensure it runs without error for the target case.
    # Note: The evaluator passes n_rows and k.
    pass
