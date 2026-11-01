import time
import random
import math

def construct(n_rows, k):
    """
    Constructs a difference triangle set using Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for 120s limit
    
    # 1. Initial Solution: Greedy
    # We use a slightly randomized greedy to get a valid start
    def greedy_init():
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

    current_rows = greedy_init()
    
    # Precompute differences for current rows
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Build global diff counts
    diff_counts = {}
    for row in current_rows:
        for d in get_row_diffs(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
    current_conflicts = sum(1 for c in diff_counts.values() if c > 1)
    current_max = max(max(row) for row in current_rows)
    
    # Objective: Minimize MaxEntry + K * Conflicts
    K_CONFLICT = 50  # Penalty weight for conflicts
    
    def get_cost(max_val, conflicts):
        return max_val + K_CONFLICT * conflicts

    current_cost = get_cost(current_max, current_conflicts)
    
    best_rows = [r[:] for r in current_rows]
    best_max = current_max
    best_conflicts = current_conflicts
    
    # Simulated Annealing Parameters
    T = 50.0  # Initial Temperature
    cooling_rate = 0.999
    min_T = 0.1
    
    # Move Generation: Suffix Shift
    # Pick a row, pick a start index i, shift row[i:] by delta
    while time.time() < deadline and T > min_T:
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, k)  # Start from index 1 to 0 to preserve 0? 
        # Actually, if we shift index 0, it's no longer 0. 
        # The problem states "starting at 0". So row[0] must remain 0.
        # So we can only shift indices 1..k.
        
        # Pick a shift amount
        # Larger shifts earlier in the annealing?
        max_shift = int(10 * (T / 50.0)) + 1
        delta = random.randint(-max_shift, max_shift)
        
        if delta == 0:
            T *= cooling_rate
            continue
            
        # Try the move
        old_row = current_rows[r_idx]
        new_row = old_row[:]
        
        # Apply shift
        valid_shift = True
        for j in range(i_idx, k + 1):
            new_val = old_row[j] + delta
            if j > 0 and new_val <= new_row[j-1]:
                valid_shift = False
                break
            new_row[j] = new_val
            
        if not valid_shift:
            T *= cooling_rate
            continue
            
        # Calculate new diffs for the modified row
        new_diffs = get_row_diffs(new_row)
        old_diffs = get_row_diffs(old_row)
        
        # Update diff_counts
        # Remove old
        for d in old_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
                
        # Add new
        for d in new_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
        # Calculate new conflicts and max
        new_conflicts = sum(1 for c in diff_counts.values() if c > 1)
        new_max = max(max(row) for row in current_rows if i != r_idx) # Optimization: only recompute max for other rows? 
        # Actually, max of current_rows[r_idx] is new_row[-1].
        # We can maintain a max heap or just recompute. n_rows=7, k=6, so 7 rows. Recomputing max is cheap.
        new_max = max(max(row) for row in current_rows if i != r_idx) if n_rows > 1 else 0
        # Wait, current_rows[r_idx] is still old_row. We need to consider new_row.
        other_maxes = [max(row) for i, row in enumerate(current_rows) if i != r_idx]
        new_max = max([new_row[-1]] + other_maxes)
        
        new_cost = get_cost(new_max, new_conflicts)
        
        # Acceptance Criteria
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            # Accept
            current_rows[r_idx] = new_row
            current_conflicts = new_conflicts
            current_max = new_max
            current_cost = new_cost
            
            # Update best if valid
            if new_conflicts == 0 and new_max < best_max:
                best_max = new_max
                best_conflicts = 0
                best_rows = [r[:] for r in current_rows]
        else:
            # Reject: Revert diff_counts
            for d in new_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
                
            for d in old_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
        T *= cooling_rate
        
    # Return the best valid solution found
    # If best_conflicts > 0, we should try to repair or just return the best valid one found during the process.
    # Since we started with a valid one, best_rows is always valid.
    return best_rows
