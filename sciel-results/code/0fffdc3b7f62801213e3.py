import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Helper to check if a row is valid given a set of used differences
    # and to compute its differences
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Greedy construction of a single row given used diffs
    # Tries to minimize the max entry
    def greedy_row(used_diffs, k, max_limit=None):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if max_limit is not None and x > max_limit:
                return None
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                row.append(x)
                # We don't update used_diffs here because this is a trial row
                # We need to check against the *global* used diffs of other rows
        return row

    # Backtracking search for a row with max entry <= limit
    def backtrack_row(used_diffs, k, limit, node_limit=10000):
        nodes_visited = 0
        row = [0]
        
        def backtrack(last_val, current_row, current_diffs):
            nonlocal nodes_visited
            if len(current_row) == k + 1:
                return list(current_row)
            
            # Pruning: if we exceed node limit, return None
            nodes_visited += 1
            if nodes_visited > node_limit:
                return None
            
            # Try next values
            # Heuristic: start from last_val + 1
            for x in range(last_val + 1, limit + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                if new_diffs & current_diffs:
                    continue
                current_row.append(x)
                current_diffs |= new_diffs
                res = backtrack(x, current_row, current_diffs)
                if res:
                    return res
                current_row.pop()
                current_diffs -= new_diffs
            return None
        
        return backtrack(0, row, set())

    # Initial greedy construction
    used_diffs_global = set()
    rows = []
    for _ in range(n_rows):
        # Greedy construction for the whole set
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs_global):
                row.append(x)
                used_diffs_global |= new_diffs
        rows.append(row)
        
    current_best = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random row to modify
        i = random.randint(0, n_rows - 1)
        
        # Compute used diffs of other rows
        used_diffs_others = set()
        for j in range(n_rows):
            if j == i:
                continue
            used_diffs_others |= get_diffs(rows[j])
        
        old_row = rows[i]
        old_max = max(old_row)
        
        # Try to find a better row
        # First try to improve: max < old_max
        new_row = None
        # Use backtracking with a limit slightly smaller than old_max
        # If old_max is small, limit the search
        limit_improve = old_max - 1
        if limit_improve >= k: # Need at least k+1 numbers, so max must be >= k
            new_row = backtrack_row(used_diffs_others, k, limit_improve, node_limit=5000)
        
        # If no improvement found, try to find an alternative with same max (to shuffle)
        if new_row is None:
            # Sometimes we just want a valid row, even if max is same
            # Or try greedy
            new_row = greedy_row(used_diffs_others, k, old_max)
            if new_row is None:
                # If greedy fails, maybe try backtracking with same limit
                new_row = backtrack_row(used_diffs_others, k, old_max, node_limit=5000)
        
        if new_row is None:
            T *= cooling_rate
            continue
            
        new_max = max(new_row)
        
        # Calculate cost change
        # Cost is the global maximum
        # If we change row i, the global max changes only if:
        # 1. old_row had the global max, and new_max < old_max -> cost decreases
        # 2. old_row had the global max, and new_max > old_max -> cost increases
        # 3. old_row did NOT have the global max, and new_max > current_global_max -> cost increases
        # 4. Otherwise cost stays same
        
        current_global_max = max(max(r) for r in rows)
        
        # Determine new global max
        if i == 0:
            # Recalculate if it's the first row? No, just compare
            pass
            
        # Efficiently calculate new global max
        # If old_row was the unique max, and new_max < old_max, new global max is second max
        # This is expensive to calculate every time. 
        # Approximation: 
        # If new_max < old_max:
        #   If old_max was the global max, cost might decrease.
        #   Else cost stays same.
        # If new_max > old_max:
        #   Cost increases to at least new_max.
        
        # Let's just calculate the new global max properly for correctness, 
        # but optimize by only doing it if necessary.
        
        # For simplicity and correctness, calculate new global max
        # This is O(n_rows * k), which is fast.
        new_rows = rows[:]
        new_rows[i] = new_row
        new_global_max = max(max(r) for r in new_rows)
        
        delta_cost = new_global_max - current_global_max
        
        if delta_cost < 0:
            # Improvement
            rows = new_rows
            if new_global_max < current_best:
                current_best = new_global_max
                best_rows = [r[:] for r in rows]
        elif delta_cost == 0:
            # Same cost, accept with some probability to explore
            if random.random() < 0.5:
                rows = new_rows
        else:
            # Worse cost
            if random.random() < pow(2.718281828, -delta_cost / T):
                rows = new_rows
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_rows

# EVOLVE-BLOCK-END
