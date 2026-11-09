"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

def _greedy_row(used, k):
    """Build one row greedily given the set of used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def construct(n_rows, k):
    # We want to minimize the largest entry across all rows.
    # The simple counting bound suggests ~105, best known 112.
    # Strategy: Use randomized restarts with different random seeds to try
    # to find a configuration with a smaller maximum entry.
    # We'll use a time limit to ensure we finish within 120s.
    
    import time
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer
    
    best_rows = None
    best_max = float('inf')
    
    # First, try the deterministic greedy as a baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    best_rows = rows
    best_max = max(max(r) for r in rows)
    
    # Now try randomized approaches
    # Idea: Randomly shuffle the order in which we pick differences, or
    # use a different greedy strategy with randomization.
    
    # Another idea: Use a different heuristic - instead of always picking the
    # smallest x, sometimes pick a slightly larger x to avoid conflicts.
    
    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(attempt * 7919 + 13)
        
        used = set()
        rows = []
        valid = True
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            # Randomized greedy: with some probability, skip ahead
            while len(row) < k + 1:
                x += 1
                # Occasionally jump ahead to avoid local minima
                if random.random() < 0.1:
                    x += random.randint(1, 5)
                
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            
            rows.append(row)
        
        current_max = max(max(r) for r in rows)
        if current_max < best_max:
            best_max = current_max
            best_rows = rows
        
        # Early stop if we hit the counting bound
        if best_max <= 105:
            break
    
    return best_rows
# EVOLVE-BLOCK-END
