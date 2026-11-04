"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    # Max scope we might need to consider. 
    # Counting bound is ~ sqrt(n_rows * num_pairs * 2). 
    # Best known for (7,5) is 112. Let's allow up to 200 for safety.
    MAX_SCOPE = 200
    
    def get_row_diffs(row):
        """Return list of all positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def compute_scope(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        """Check if all differences are distinct."""
        all_diffs = []
        for row in rows:
            all_diffs.extend(get_row_diffs(row))
        return len(all_diffs) == len(set(all_diffs))

    # --- Fast Greedy Construction using Bitmasks ---
    
    # Precompute difference bitmasks for all possible rows up to MAX_SCOPE
    # A row is valid if its bitmask has no overlap with the current global used bitmask.
    # Since MAX_SCOPE is 200, we can use an integer bitmask (Python ints are arbitrary precision).
    
    # We will generate candidate rows on the fly or precompute a list of good rows.
    # Precomputing ALL rows of length k+1 up to 200 is C(200, 6) which is ~ 10^10, too many.
    # Instead, we use a greedy approach that picks rows one by one.
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        used_diffs_mask = 0
        rows = []
        
        # We want to minimize the max element. 
        # Strategy: Try to build rows with small elements first.
        # However, simple greedy often gets stuck. 
        # Better: Generate a pool of candidate rows sorted by max element, then pick.
        
        # For efficiency, let's generate candidates up to a certain scope limit.
        # If we fail, we increase the limit.
        
        # Actually, a better greedy: 
        # 1. Generate many random valid rows (small scope).
        # 2. Select n_rows of them that are mutually disjoint in differences.
        
        # Let's try a different greedy: 
        # Iterate scope from lower bound to upper bound.
        # For each scope, generate all valid rows? No, too slow.
        
        # Let's stick to randomized greedy with restarts.
        # But the current program's SA was slow. 
        # Let's try: 
        # 1. Generate n_rows random rows that are internally valid.
        # 2. Check global validity.
        # 3. If invalid, identify conflicting rows and replace them with new random rows.
        
        # This is essentially a local search but with row-level moves.
        
        # Let's implement a "Greedy with Backtracking" or "Constraint Satisfaction" style.
        
        # Simpler: 
        # Generate a large pool of candidate rows (e.g., 1000) with max element <= 120.
        # Filter to keep only those that are internally valid (always true if constructed properly).
        # Then use a greedy set packing algorithm to pick n_rows disjoint rows.
        
        # To get a good pool, we need to generate rows efficiently.
        
        def generate_random_row(max_val, rng):
            # Pick k random distinct numbers from 1..max_val, sort them, prepend 0.
            # Ensure no duplicate differences? 
            # It's hard to ensure internal validity randomly without checking.
            # But for k=5 (6 elements), the chance of internal collision is low if max_val is large enough?
            # No, differences must be distinct. 
            # Let's just generate random increasing sequences and check internal validity.
            
            while True:
                # Sample k distinct values from 1 to max_val
                candidates = rng.sample(range(1, max_val + 1), k)
                candidates.sort()
                row = [0] + candidates
                # Check internal differences
                diffs = get_row_diffs(row)
                if len(diffs) == len(set(diffs)):
                    return row

        # Pool generation
        pool_size = 2000
        # Start with a moderate scope, e.g., 110
        current_scope_limit = 110
        
        while True:
            # Generate pool
            pool = []
            seen_rows = set()
            for _ in range(pool_size):
                r = generate_random_row(current_scope_limit, rng)
                # Use tuple for hashing
                t = tuple(r)
                if t not in seen_rows:
                    seen_rows.add(t)
                    pool.append(r)
            
            # Greedy selection from pool
            used_diffs_mask = 0
            selected_rows = []
            remaining_pool = pool[:]
            rng.shuffle(remaining_pool) # Randomize order for diversity
            
            for row in remaining_pool:
                if len(selected_rows) == n_rows:
                    break
                # Check if this row's differences conflict with used_diffs_mask
                row_diffs = get_row_diffs(row)
                row_mask = 0
                for d in row_diffs:
                    row_mask |= (1 << d)
                
                if (used_diffs_mask & row_mask) == 0:
                    selected_rows.append(row)
                    used_diffs_mask |= row_mask
            
            if len(selected_rows) == n_rows:
                return selected_rows
            
            # If failed, increase scope limit
            current_scope_limit += 5
            if current_scope_limit > 150:
                # Fallback to very large scope to ensure validity
                # Just generate sequentially valid rows
                rows = []
                used = set()
                for _ in range(n_rows):
                    row = [0]
                    x = 0
                    while len(row) < num_elements:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                    rows.append(row)
                return rows

    # --- Local Search to Improve Scope ---
    
    def local_search(rows, seed):
        rng = random.Random(seed)
        current_rows = [r[:] for r in rows]
        current_scope = compute_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        # Precompute masks for current rows
        row_masks = []
        for r in current_rows:
            m = 0
            for d in get_row_diffs(r):
                m |= (1 << d)
            row_masks.append(m)
        
        # Global mask
        global_mask = 0
        for m in row_masks:
            global_mask |= m
            
        # Ensure valid start
        if global_mask.bit_count() != total_diffs_needed:
             # Actually bit_count counts bits set. If all diffs are distinct, 
             # the number of set bits should equal total_diffs_needed.
             # But wait, if two rows have the same diff, the bit is set only once.
             # So if valid, bit_count == total_diffs_needed.
             pass

        # Generate candidate rows for replacement
        # We want to replace a row with a new one that has a smaller max element
        # or allows other rows to have smaller max elements.
        
        # Strategy:
        # 1. Identify the row with the largest max element.
        # 2. Generate new candidate rows with max element < current_max.
        # 3. Check if swapping reduces the global scope or maintains validity.
        
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
                
            # Find row to replace
            # Target the row with the largest scope
            max_idx = 0
            max_val = 0
            for i, r in enumerate(current_rows):
                if r[-1] > max_val:
                    max_val = r[-1]
                    max_idx = i
            
            # Generate a few new candidates for this row
            # Try to find a row with max element < max_val
            found_replacement = False
            for _ in range(20): # Try 20 random candidates
                # Target scope slightly smaller than current max
                target_scope = max_val - rng.randint(1, 5)
                if target_scope < num_elements: # Must fit k+1 elements
                    target_scope = num_elements
                
                new_row = generate_random_row(target_scope, rng)
                new_mask = 0
                for d in get_row_diffs(new_row):
                    new_mask |= (1 << d)
                
                # Check conflict with other rows
                # Remove current row's mask from global
                other_mask = global_mask & ~row_masks[max_idx]
                
                if (other_mask & new_mask) == 0:
                    # Valid swap
                    # Update state
                    global_mask = (global_mask & ~row_masks[max_idx]) | new_mask
                    row_masks[max_idx] = new_mask
                    current_rows[max_idx] = new_row
                    
                    new_scope = compute_scope(current_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                        
                    found_replacement = True
                    break
        
            if not found_replacement:
                # Try a random row
                idx = rng.randint(0, n_rows - 1)
                current_max_in_row = current_rows[idx][-1]
                # Try to improve this specific row
                for _ in range(5):
                    target_scope = current_max_in_row - rng.randint(1, 3)
                    if target_scope < num_elements:
                        continue
                    new_row = generate_random_row(target_scope, rng)
                    new_mask = 0
                    for d in get_row_diffs(new_row):
                        new_mask |= (1 << d)
                    
                    other_mask = global_mask & ~row_masks[idx]
                    if (other_mask & new_mask) == 0:
                        global_mask = (global_mask & ~row_masks[idx]) | new_mask
                        row_masks[idx] = new_mask
                        current_rows[idx] = new_row
                        
                        new_scope = compute_scope(current_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [r[:] for r in current_rows]
                        break
                        
        return best_rows

    # Main Logic
    # 1. Get an initial valid solution via greedy
    # Use multiple seeds to find a good initial solution
    initial_rows = None
    best_initial_scope = float('inf')
    
    for seed in range(10):
        if time.time() > DEADLINE - 60:
            break
        rows = greedy_construct(seed)
        scope = compute_scope(rows)
        if scope < best_initial_scope:
            best_initial_scope = scope
            initial_rows = rows
            
    if initial_rows is None:
        # Fallback
        initial_rows = greedy_construct(0)
        
    # 2. Improve with local search
    # Run local search for remaining time
    # We can run it multiple times with different seeds if time permits
    final_rows = initial_rows
    final_scope = best_initial_scope
    
    while time.time() < DEADLINE - 10:
        # Run local search on the current best
        improved = local_search(final_rows, random.randint(0, 10000))
        imp_scope = compute_scope(improved)
        if imp_scope < final_scope:
            final_scope = imp_scope
            final_rows = improved
        else:
            # If no improvement, break to save time? Or continue with different seed?
            # Let's continue a few more times to ensure we've explored.
            pass

    # Final validation
    if is_valid(final_rows):
        return final_rows
    else:
        # Fallback to greedy
        return greedy_construct(0)

# EVOLVE-BLOCK-END
