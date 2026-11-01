import time
import random
import sys

def construct(n_rows, k):
    """
    Constructs n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct over all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # For n=4, k=7, we have 4 rows of 8 elements.
    # Total elements = 32.
    # Total differences = 4 * (7*8/2) = 112.
    # Max possible difference is bounded by the largest entry.
    # We assume max entry won't exceed 500 for safety in array size.
    MAX_VAL_BOUND = 500
    
    # Helper to get all differences for a specific row
    def get_row_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    # Helper to check validity of the entire set
    def is_valid(rows):
        all_diffs = []
        for row in rows:
            all_diffs.extend(get_row_diffs(row))
        return len(all_diffs) == len(set(all_diffs))

    # 1. Greedy Initialization
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                # Try to find the smallest x > row[-1]
                x = row[-1] + 1
                while True:
                    # Check conflicts with existing row elements
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        break
                    x += 1
                row.append(x)
                used_diffs.update(new_diffs)
            rows.append(row)
        return rows

    # 2. Simulated Annealing Optimizer
    def optimize_sa(rows, time_limit):
        # State
        current_rows = [r[:] for r in rows]
        current_max = max(max(r) for r in current_rows)
        
        # Pre-compute used diffs set
        current_used = set()
        for r in current_rows:
            current_used.update(get_row_diffs(r))
        
        # We track the best solution found
        best_rows = [r[:] for r in current_rows]
        best_max = current_max
        
        # SA Parameters
        T = 50.0  # Initial temperature
        T_min = 0.1
        alpha = 0.995  # Cooling rate
        
        # Helper to calculate cost (max entry)
        def get_cost(rows):
            return max(max(r) for r in rows)
        
        # Helper to check if a specific move is valid and get the delta in used_diffs
        # Returns (is_valid, new_used_set, cost_change)
        # Note: We don't want to copy the whole set every time if possible, 
        # but for k=7, sets are small (28 diffs per row). 
        # Total 112 diffs. Set operations are fast.
        
        steps = 0
        while time.time() < time_limit:
            # Select a move
            # Bias towards reducing the max
            if random.random() < 0.6:
                # Find the index of the global maximum
                max_val = -1
                max_idx = (-1, -1)
                for i in range(n_rows):
                    for j in range(1, len(current_rows[i])): # Skip 0
                        if current_rows[i][j] > max_val:
                            max_val = current_rows[i][j]
                            max_idx = (i, j)
                i, j = max_idx
            else:
                i = random.randint(0, n_rows - 1)
                j = random.randint(1, len(current_rows[i]) - 1)
            
            row = current_rows[i]
            old_val = row[j]
            
            # Determine valid range for new value
            # Must be > row[j-1] and < row[j+1] (if exists)
            lower_bound = row[j-1] + 1
            upper_bound = old_val - 1 # We want to try reducing, but SA allows increases
            if j < len(row) - 1:
                upper_bound = min(upper_bound, row[j+1] - 1)
            
            # If we want to allow increases to escape local minima:
            # Let's sample from a range around old_val
            # Range: [lower_bound, upper_bound] if we want to decrease
            # Or [lower_bound, old_val + T] if we want to allow increases?
            # Actually, let's just pick a random valid integer in [lower_bound, upper_bound]
            # If lower_bound > upper_bound, no move possible in this direction.
            # To allow "bad" moves (increasing max), we might need to pick a value > old_val?
            # No, if we pick a value > old_val, it must be < row[j+1].
            # Let's define the search space as [lower_bound, min(upper_bound, old_val + 10)]
            # If we strictly want to reduce max, we only look below old_val.
            # But SA needs to accept worse solutions.
            # Let's define the candidate value range.
            
            max_possible = row[j+1] - 1 if j < len(row) - 1 else old_val + 10
            min_possible = row[j-1] + 1
            
            if min_possible > max_possible:
                T *= alpha
                continue
                
            # Sample new value
            # Bias towards smaller values if we are trying to minimize
            # But to explore, sample uniformly in [min_possible, max_possible]
            # However, if max_possible is much larger than old_val, we might jump too far.
            # Let's limit the jump.
            search_max = min(max_possible, old_val + 5)
            if search_max < min_possible:
                T *= alpha
                continue
                
            new_val = random.randint(min_possible, search_max)
            
            if new_val == old_val:
                continue
                
            # Calculate old diffs involving index j
            old_diffs = set()
            for t in range(len(row)):
                if t != j:
                    if row[j] > row[t]:
                        old_diffs.add(row[j] - row[t])
                    else:
                        old_diffs.add(row[t] - row[j])
            
            # Calculate new diffs involving index j
            new_diffs = set()
            for t in range(len(row)):
                if t != j:
                    if new_val > row[t]:
                        new_diffs.add(new_val - row[t])
                    else:
                        new_diffs.add(row[t] - new_val)
            
            # Check conflicts
            # The new diffs must not conflict with (current_used - old_diffs)
            # i.e., new_diffs & (current_used - old_diffs) must be empty
            # Equivalently: (new_diffs & current_used) must be subset of old_diffs?
            # No. new_diffs are the *new* set of diffs for this row.
            # The global set becomes (current_used - old_diffs) | new_diffs.
            # This is valid if (new_diffs & (current_used - old_diffs)) is empty.
            
            # Optimization: 
            # Check if any d in new_diffs is in current_used AND not in old_diffs.
            conflict = False
            for d in new_diffs:
                if d in current_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                T *= alpha
                continue
                
            # Valid move!
            # Calculate new cost
            # The only change in max is potentially at this row.
            # New row max: max(row[:j] + [new_val] + row[j+1:])
            # Old row max: max(row)
            # Global max: max(max(other rows), new_row_max)
            
            # Update state
            row[j] = new_val
            current_used = (current_used - old_diffs) | new_diffs
            
            new_max = max(max(r) for r in current_rows)
            
            # Accept/Reject
            old_cost = current_max
            new_cost = new_max
            
            if new_cost < old_cost:
                # Always accept
                current_max = new_cost
                if new_cost < best_max:
                    best_max = new_cost
                    best_rows = [r[:] for r in current_rows]
            else:
                # Accept with probability exp(-(new_cost - old_cost) / T)
                delta = new_cost - old_cost
                if delta == 0:
                    # Accept neutral moves to explore
                    pass
                else:
                    prob = 2.718281828 ** (-delta / T)
                    if random.random() > prob:
                        # Reject: Undo move
                        row[j] = old_val
                        current_used = (current_used - new_diffs) | old_diffs
                        # current_max remains old_cost
                        T *= alpha
                        continue
            
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_rows, best_max

    # Main Logic
    # 1. Get initial solution
    init_rows = greedy_init()
    
    # 2. Optimize
    # We will run SA for the remaining time
    time_left = deadline - time.time()
    if time_left > 1:
        best_rows, best_score = optimize_sa(init_rows, deadline - 1.0)
    else:
        best_rows = init_rows
        best_score = max(max(r) for r in best_rows)
        
    # 3. Final Verification (Optional but good for safety)
    # If the solution is invalid, fall back to greedy
    # (SA should maintain validity, but just in case)
    if not is_valid(best_rows):
        best_rows = init_rows
        best_score = max(max(r) for r in best_rows)

    return best_rows
