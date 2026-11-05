"""Improved program for difference triangle set (7,5)."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Strategy:
    1. Use a randomized greedy with backtracking (limited depth) to build rows.
    2. Try multiple random orderings and perturbations.
    3. Keep the best solution found within the time budget.
    """
    deadline = time.time() + 110  # 110s safety margin (limit is 120s)
    best_rows = None
    best_max = float('inf')
    
    # Number of differences per row
    num_diffs = k * (k + 1) // 2
    total_diffs_needed = n_rows * num_diffs
    
    # Upper bound on differences: if max entry is M, differences are in [1, M]
    # We need total_diffs_needed distinct differences, so M >= total_diffs_needed
    lower_bound = total_diffs_needed
    
    # Start with a reasonable upper bound
    # The simple counting bound is 105 for (7,5), best known is 112
    # Let's start from the lower bound and go up
    current_max = max(lower_bound, 80)
    
    # First, try a deterministic greedy to get a baseline
    def greedy_construct(n_rows, k, max_val, randomize=False, seed=None):
        """Greedy construction with optional randomization."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Generate candidate next values
                if randomize:
                    # Try values from current max + 1 up to max_val
                    candidates = list(range(row[-1] + 1, max_val + 1))
                    rng.shuffle(candidates)
                    # But we want smaller values first, so sort by value after shuffling?
                    # No, let's try a mix: some random, some greedy
                    # Let's just shuffle and try in order
                else:
                    candidates = list(range(row[-1] + 1, max_val + 1))
                
                placed = False
                for x in candidates:
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        if d > 0:
                            new_diffs.add(d)
                    
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        placed = True
                        break
                
                if not placed:
                    return None
        
        return rows
    
    # Try deterministic greedy first
    rows = greedy_construct(n_rows, k, current_max, randomize=False)
    if rows is not None:
        max_val = max(max(r) for r in rows)
        if max_val < best_max:
            best_max = max_val
            best_rows = [r[:] for r in rows]
    
    # Now try randomized approaches with increasing max_val
    # We'll try to find a solution with max_val < best_max
    
    # Start from a smaller max_val and increase
    # The lower bound is total_diffs_needed
    # Let's try values from lower_bound up to best_max - 1
    
    # For each candidate max_val, try multiple random attempts
    attempts_per_max = 20
    max_attempts_total = 500
    
    total_attempts = 0
    current_max = lower_bound
    
    while current_max < best_max and time.time() < deadline and total_attempts < max_attempts_total:
        # Try to construct with this max_val
        found = False
        for attempt in range(attempts_per_max):
            if time.time() > deadline or total_attempts >= max_attempts_total:
                break
            
            total_attempts += 1
            seed = random.randint(0, 10**9)
            rows = greedy_construct(n_rows, k, current_max, randomize=True, seed=seed)
            if rows is not None:
                max_val = max(max(r) for r in rows)
                if max_val < best_max:
                    best_max = max_val
                    best_rows = [r[:] for r in rows]
                    found = True
                    # If we found a solution, we can try to improve further
                    # But we're already using current_max, so this is good
                    break
        
        if found:
            # Try to reduce max_val further
            # We found a solution with max <= current_max, but maybe we can do better
            # Actually, the greedy might not use the full range, so let's check
            # and potentially reduce current_max
            pass
        
        current_max += 1
    
    # If we still have time, try to improve with more sophisticated search
    # Use a local search / hill climbing approach
    
    if best_rows is not None and time.time() < deadline - 5:
        # Try to improve the best solution by local search
        # We can try to reduce the max entry by rearranging rows
        
        # Compute all differences for the current best
        def get_all_diffs(rows):
            diffs = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        diffs.add(row[j] - row[i])
            return diffs
        
        # Try swapping rows or permuting
        # Actually, the order of rows doesn't matter, so we can try different row orderings
        
        # Let's try a simple local search: for each row, try to find a smaller max
        # by adjusting the row while keeping other rows fixed
        
        # This is complex, so let's try a different approach:
        # Use simulated annealing to minimize the max entry
        
        # Actually, let's just try more random constructions with the current best_max - 1
        # and see if we can find something better
        
        target_max = best_max - 1
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            seed = random.randint(0, 10**9)
            rows = greedy_construct(n_rows, k, target_max, randomize=True, seed=seed)
            if rows is not None:
                max_val = max(max(r) for r in rows)
                if max_val < best_max:
                    best_max = max_val
                    best_rows = [r[:] for r in rows]
                    target_max = best_max - 1
    
    # Return the best solution found
    if best_rows is not None:
        return best_rows
    else:
        # Fallback: return a valid but potentially suboptimal solution
        # Use the original greedy without max constraint
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
