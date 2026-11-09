"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _greedy_row(used, k, randomize=False, jump_prob=0.1, jump_max=5):
    """Build one row greedily given the set of used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if randomize and random.random() < jump_prob:
            x += random.randint(1, jump_max)
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _row_diffs(row):
    """Calculate the set of positive differences for a row."""
    return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0
    
    # 1. Initialize with deterministic greedy to get a valid baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, randomize=False)
        rows.append(row)
    
    current_rows = rows[:]
    current_max = max(max(r) for r in current_rows)
    best_rows = current_rows[:]
    best_max = current_max
    
    # 2. Simulated Annealing
    # State: list of rows
    # Cost: max entry
    # Move: Regenerate one random row
    
    T = 10.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 0.1
    
    # Pre-compute difference sets for current rows to speed up updates
    # But for simplicity in this iteration, we recalculate if needed or maintain a global set
    # Let's maintain a global used set and the diff sets of individual rows
    
    row_diffs = [_row_diffs(r) for r in current_rows]
    global_used = set().union(*row_diffs)
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = min_T
            
        # Select a random row to perturb
        idx = random.randint(0, n_rows - 1)
        
        # Remove current row's diffs from global set
        global_used -= row_diffs[idx]
        
        # Generate a new candidate row
        # We use a randomized greedy to find a valid row that fits in the remaining space
        # If it fails to find a row within a reasonable bound, we revert
        # But _greedy_row always finds a row, it might just be large.
        
        # To make the move interesting, we vary the randomization parameters
        jump_prob = random.uniform(0.0, 0.3)
        jump_max = random.randint(1, 10)
        
        new_row = _greedy_row(global_used, k, randomize=True, jump_prob=jump_prob, jump_max=jump_max)
        new_diffs = _row_diffs(new_row)
        
        # Check if new_row is valid (it is by construction, but let's be safe)
        if new_diffs & global_used:
             # This shouldn't happen with _greedy_row, but if it does, revert
             global_used |= row_diffs[idx]
             continue
             
        # Calculate new cost
        # The max entry of the new configuration is max(current_max_except_idx, max(new_row))
        # We need to know the max entry of all other rows.
        # Optimization: Keep track of max entry per row and global max.
        # For now, simple recalculation is O(N*K^2) which is small.
        
        new_rows = current_rows[:]
        new_rows[idx] = new_row
        new_max = max(max(r) for r in new_rows)
        
        # Acceptance probability
        delta = new_max - current_max
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            current_rows = new_rows
            current_max = new_max
            row_diffs[idx] = new_diffs
            global_used |= new_diffs
            
            if current_max < best_max:
                best_max = current_max
                best_rows = current_rows[:]
                
                # Early stop if we hit the theoretical lower bound (counting bound)
                # Counting bound for n=7, k=6:
                # Total diffs = n * (k*(k+1)/2) = 7 * 21 = 147.
                # Max diff is M. Number of possible diffs is M.
                # So M >= 147? No, diffs are 1..M.
                # Actually, the counting bound is usually derived from the number of distinct differences.
                # For (7,6), best known is 112.
                if best_max <= 112:
                     # We can try to go lower, but 112 is the target.
                     # The simple counting bound mentioned in prompt is 105.
                     if best_max <= 105:
                         break
        else:
            # Reject move, restore global_used
            global_used |= row_diffs[idx]
            
        T *= cooling_rate
        
    return best_rows
# EVOLVE-BLOCK-END
