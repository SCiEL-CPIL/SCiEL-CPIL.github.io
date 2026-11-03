"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct across
    all rows. Minimize the largest entry.
    
    Uses Simulated Annealing with efficient difference set updates.
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on max entry
    # Each row has max value M. Max difference is M.
    # We need total_diffs distinct values from 1..M.
    # So M >= total_diffs.
    # Actually, the differences are not necessarily 1..M, but they must be distinct integers.
    # The maximum possible difference in a row with max M is M.
    # So all differences must be in [1, M].
    # Thus M >= total_diffs.
    lower_bound = total_diffs
    
    # Heuristic lower bound:
    # Each row contributes k(k+1)/2 diffs.
    # If we pack them tightly, the max diff in a row is at least k(k+1)/2.
    # But they share the global space.
    # A better bound: M >= total_diffs.
    
    # Start with a valid solution
    # Greedy construction to get a starting point
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k+1):
                # Try to find smallest x > row[-1] such that diffs are unique
                # This greedy might fail or produce large M.
                # We'll use a simple randomized greedy to get *a* valid solution first.
                found = False
                x = row[-1] + 1
                while not found:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                    else:
                        x += 1
            rows.append(row)
        return rows

    initial_rows = greedy_construct()
    
    # Precompute differences for initial state
    # To make SA fast, we maintain a global set of differences.
    # However, checking validity of a move requires knowing which diffs to remove/add.
    
    # Let's represent the state as a list of lists.
    # We'll maintain a global set `global_diffs` and a list of sets `row_diffs` for each row?
    # No, just global_diffs is enough if we carefully remove/add.
    
    # But we need to know which diffs belong to which row/position to remove them.
    # For a move on row r, position j:
    # Old diffs to remove: {old_val - row[i] for i in range(j)} U {row[i] - old_val for i in range(j+1, k+1)}
    # New diffs to add: {new_val - row[i] for i in range(j)} U {row[i] - new_val for i in range(j+1, k+1)}
    
    # We need to ensure the new row is strictly increasing.
    
    current_rows = [list(r) for r in initial_rows]
    current_max = max(max(r) for r in current_rows)
    
    # Build global diff set
    global_diffs = set()
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                global_diffs.add(row[j] - row[i])
                
    # Verify initial solution is valid (it should be by construction)
    if len(global_diffs) != total_diffs:
        # Fallback if greedy failed (unlikely for small k, but possible for large n_rows)
        # Just return the greedy result if SA fails or if we can't start
        pass
        
    best_rows = [list(r) for r in current_rows]
    best_max = current_max
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 115.0
    
    # Parameters
    T = 100.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    # Helper to get cost
    def get_cost(rows):
        return max(max(r) for r in rows)
    
    # Helper to check if a new value x at row r, pos j is valid
    # Returns True if valid, False otherwise
    # Also updates global_diffs if valid? No, let's do it in the main loop for clarity and speed.
    
    # Optimization: Precompute indices
    # For a move (r, j, new_val):
    # 1. Check bounds: row[r][j-1] < new_val < row[r][j+1] (if exists)
    # 2. Compute old_diffs_to_remove
    # 3. Compute new_diffs_to_add
    # 4. Check if any new_diff is in global_diffs - old_diffs_to_remove
    #    If yes, invalid.
    # 5. If valid, update global_diffs and row.
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Pick random move
        r = random.randint(0, n_rows - 1)
        j = random.randint(1, k) # Position 1..k (0 is fixed at 0)
        
        row = current_rows[r]
        old_val = row[j]
        
        # Determine range for new_val
        low_bound = row[j-1] + 1
        high_bound = row[j+1] - 1 if j < k else current_max + 100 # Allow expansion
        
        # Randomly pick a new value
        # Bias towards smaller values to minimize max
        # But we need to explore.
        # Let's pick from a range around old_val or from [low_bound, high_bound]
        
        # Strategy:
        # 1. Try to decrease max: if old_val is current_max, try smaller values.
        # 2. Otherwise, try random values in [low_bound, high_bound].
        
        # To make it efficient, let's sample from [low_bound, high_bound]
        # But high_bound can be large.
        # Let's cap the search space for new_val to be reasonable.
        # If we want to minimize max, we shouldn't go too high.
        # But sometimes we need to go high to escape a local minimum.
        
        # Let's try a few random candidates
        found_valid = False
        new_val = old_val
        
        # Number of candidates to try
        num_candidates = 10
        
        for _ in range(num_candidates):
            # Sample new_val
            # Mix of local and global moves
            if random.random() < 0.5:
                # Local move
                span = max(1, int(abs(old_val) * 0.1))
                new_val = old_val + random.randint(-span, span)
            else:
                # Global move within bounds
                if high_bound > low_bound:
                    new_val = random.randint(low_bound, high_bound)
                else:
                    continue
            
            if new_val <= low_bound or (j < k and new_val >= row[j+1]):
                continue
                
            # Check validity
            # Old diffs
            old_diffs = set()
            for i in range(j):
                old_diffs.add(old_val - row[i])
            for i in range(j+1, k+1):
                old_diffs.add(row[i] - old_val)
                
            # New diffs
            new_diffs = set()
            for i in range(j):
                new_diffs.add(new_val - row[i])
            for i in range(j+1, k+1):
                new_diffs.add(row[i] - new_val)
                
            # Check for collisions
            # A new diff is invalid if it is in global_diffs AND not in old_diffs
            # (because old_diffs are being removed)
            valid = True
            for d in new_diffs:
                if d in global_diffs and d not in old_diffs:
                    valid = False
                    break
                    
            if valid:
                # Apply move
                row[j] = new_val
                global_diffs.difference_update(old_diffs)
                global_diffs.update(new_diffs)
                
                new_max = max(max(r_row) for r_row in current_rows)
                
                # Accept?
                old_max = current_max
                delta_cost = new_max - old_max
                
                if delta_cost < 0 or random.random() < pow(2.718281828, -delta_cost / T):
                    # Accept
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r_row) for r_row in current_rows]
                    found_valid = True
                else:
                    # Reject: revert
                    row[j] = old_val
                    global_diffs.difference_update(new_diffs)
                    global_diffs.update(old_diffs)
                break # Only try one candidate per SA step? Or loop through candidates?
                # Let's try one candidate per temperature step to keep it simple and fast.
            
        T *= cooling_rate
        
    return best_rows

# EVOLVE-BLOCK-END
