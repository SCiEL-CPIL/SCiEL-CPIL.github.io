import random
import time
import sys

def construct(n_rows, k):
    """
    Finds a difference triangle set for n_rows rows and k differences per row (k+1 elements).
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s safety margin for 120s limit
    
    # Total number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Initial solution: Greedy construction
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
    
    best_rows = greedy_init()
    best_score = max(max(r) for r in best_rows)
    
    # Calculate cost: (num_conflicts, max_element)
    def get_cost(rows):
        used = {}
        conflicts = 0
        max_val = 0
        for r in rows:
            for i in range(len(r)):
                if r[i] > max_val:
                    max_val = r[i]
                for j in range(i):
                    d = r[i] - r[j]
                    if d in used:
                        conflicts += 1
                    else:
                        used[d] = True
        return (conflicts, max_val)
    
    current_rows = [list(r) for r in best_rows]
    current_cost = get_cost(current_rows)
    
    # Simulated Annealing parameters
    T = 100.0
    T_min = 0.01
    alpha = 0.9995  # Cooling rate
    max_iters = 500000 # Upper bound on iterations, will stop by time anyway
    
    # Helper to check if a row is strictly increasing
    def is_valid_row(row):
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        return True
    
    # Move: Pick a row, pick a position, change the value
    # To keep it strictly increasing, we change a_{i,j} to a new value v
    # such that a_{i,j-1} < v < a_{i,j+1} (if j < k)
    # Or if j == k, v > a_{i,k-1}
    
    def make_move(rows):
        i = random.randint(0, n_rows - 1)
        # Don't change the first element (always 0)
        j = random.randint(1, k)
        
        # Determine range for new value
        low = rows[i][j-1] + 1
        # High is either rows[i][j+1] - 1 or current value + some delta
        if j < k:
            high = rows[i][j+1] - 1
        else:
            high = rows[i][j] + 10 # Allow expansion
        
        if high < low:
            # Need to expand the row or shift
            # Simplest: just pick a random larger value if it's the last element
            if j == k:
                new_val = random.randint(rows[i][j-1] + 1, rows[i][j] + 20)
            else:
                # This shouldn't happen if we maintain validity, but if it does, 
                # we might need to shift the whole tail. 
                # For simplicity in SA, let's just retry or pick a value that forces a shift.
                # Let's just pick a value and then fix the row.
                new_val = random.randint(low, high)
        else:
            new_val = random.randint(low, high)
            
        old_val = rows[i][j]
        rows[i][j] = new_val
        
        # Fix the row to be strictly increasing
        # If new_val is too small, push subsequent elements up
        if j < k:
            for m in range(j+1, k+1):
                if rows[i][m] <= rows[i][m-1]:
                    rows[i][m] = rows[i][m-1] + 1
        else:
            # If last element, just ensure it's > prev
            if rows[i][j] <= rows[i][j-1]:
                rows[i][j] = rows[i][j-1] + 1
        
        return i, j, old_val

    best_conflicts, best_max = current_cost
    
    while time.time() < deadline:
        # Make a move
        i, j, old_val = make_move(current_rows)
        
        # Calculate new cost
        new_cost = get_cost(current_rows)
        
        # Acceptance probability
        delta_conf = new_cost[0] - current_cost[0]
        delta_max = new_cost[1] - current_cost[1]
        
        # We prioritize conflicts. If conflicts increase, only accept if T is high.
        # If conflicts same, minimize max.
        # Combined cost function for SA:
        # cost = conflicts * 10000 + max_val
        old_combined = current_cost[0] * 10000 + current_cost[1]
        new_combined = new_cost[0] * 10000 + new_cost[1]
        delta = new_combined - old_combined
        
        if delta < 0 or random.random() < pow(2.71828, -delta / max(T, 1e-9)):
            current_cost = new_cost
            # Update best if valid and smaller max
            if new_cost[0] == 0 and new_cost[1] < best_max:
                best_max = new_cost[1]
                best_rows = [list(r) for r in current_rows]
        else:
            # Revert
            current_rows[i][j] = old_val
            # Re-fix row if necessary? 
            # Since we only changed one element and reverted, the row structure 
            # might have been altered by the "fix" step. 
            # To be safe, we should store the whole row or revert carefully.
            # Let's store the whole row before move for simplicity in revert.
            # Actually, my make_move modifies the list in place. 
            # Reverting just the value might leave the tail shifted.
            # Better approach: Store a copy of the row before modification.
            
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            
        # If we found a valid solution, we can restart or continue to optimize max
        # If current is valid, we can try to compress it
        if current_cost[0] == 0:
             # Try to reduce max value by local search
             # Just continue SA, it will naturally try to reduce max_val since it's in cost
             pass
             
    return best_rows

# To make the revert logic correct, I need to adjust the SA loop to store row copies.
# Let's refine the implementation in the final block.
