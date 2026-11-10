import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    def get_diffs(row):
        """Get all positive differences in a row as a set."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def compute_scope(rows):
        """Compute the maximum entry across all rows."""
        return max(max(row) for row in rows)

    def greedy_construct(seed):
        """Greedy construction to get an initial valid solution."""
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                # Check for duplicates within the new set and against used
                if len(new_diffs) == len(row) and not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        
        return rows, used

    # Get initial solution with greedy
    initial_rows, initial_used_diffs = greedy_construct(0)
    best_rows = [row[:] for row in initial_rows]
    best_scope = compute_scope(best_rows)
    
    # Simulated annealing to improve
    rng = random.Random(42)
    current_rows = [row[:] for row in initial_rows]
    current_used_diffs = set(initial_used_diffs)
    current_scope = compute_scope(current_rows)
    
    T = 10.0  # Initial temperature
    cooling = 0.999
    
    while time.time() < deadline and T > 0.1:
        # Pick a random row to modify
        row_idx = rng.randint(0, n_rows - 1)
        old_row = current_rows[row_idx][:]
        old_diffs = get_diffs(old_row)
        
        # Generate a candidate new row
        new_row = None
        if rng.random() < 0.7:
            # Try to reduce the last (largest) element
            last_val = old_row[-1]
            # Try a smaller value
            reduction = rng.randint(1, max(1, last_val // 10))
            new_val = last_val - reduction
            new_val = max(new_val, old_row[-2] + 1)  # Must be > previous
            
            if new_val != old_row[-1]:
                new_row = old_row[:-1] + [new_val]
        else:
            # Try to rearrange a middle element
            if len(old_row) > 2:
                pos = rng.randint(1, len(old_row) - 2)  # Not first or last
                old_val = old_row[pos]
                # Try a different value
                lower_bound = old_row[pos - 1] + 1
                upper_bound = old_row[pos + 1] - 1
                if upper_bound > lower_bound:
                    new_val = rng.randint(lower_bound, upper_bound)
                    if new_val != old_val:
                        new_row = old_row[:]
                        new_row[pos] = new_val
        
        if new_row is None or new_row == old_row:
            T *= cooling
            continue
            
        # Check validity of new_row
        new_diffs = get_diffs(new_row)
        
        # 1. Check internal consistency: all differences in new_row must be distinct
        # Since get_diffs returns a set, if len(new_diffs) < expected count, there's a duplicate.
        # Expected number of differences is k*(k+1)/2
        expected_diff_count = (k + 1) * k // 2
        if len(new_diffs) != expected_diff_count:
            T *= cooling
            continue
            
        # 2. Check against other rows
        # Remove old diffs from global set
        current_used_diffs -= old_diffs
        
        # Check if new diffs collide with remaining global diffs
        if new_diffs & current_used_diffs:
            # Collision, revert global set
            current_used_diffs |= old_diffs
            T *= cooling
            continue
        
        # Valid move. Add new diffs to global set
        current_used_diffs |= new_diffs
        current_rows[row_idx] = new_row
        new_scope = compute_scope(current_rows)
        
        if new_scope < current_scope:
            current_scope = new_scope
            if new_scope < best_scope:
                best_rows = [row[:] for row in current_rows]
                best_scope = new_scope
        else:
            # Accept with probability based on temperature
            delta = new_scope - current_scope
            if delta > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_scope = new_scope
                else:
                    # Reject move: revert row and global set
                    current_rows[row_idx] = old_row
                    current_used_diffs -= new_diffs
                    current_used_diffs |= old_diffs
            else:
                # Should not happen if new_scope >= current_scope, but just in case
                current_scope = new_scope
                
        T *= cooling
    
    # Final verification
    # Re-check best_rows validity to be absolutely sure
    final_used = set()
    valid = True
    for row in best_rows:
        diffs = get_diffs(row)
        if len(diffs) != (k + 1) * k // 2:
            valid = False
            break
        if diffs & final_used:
            valid = False
            break
        final_used |= diffs
        
    if not valid:
        # Fallback to initial greedy solution
        best_rows = [row[:] for row in initial_rows]
    
    return best_rows
