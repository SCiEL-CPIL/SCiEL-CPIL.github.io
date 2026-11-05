"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Strategy: Use a greedy approach with backtracking and optimization.
    We use a more sophisticated greedy that tries to balance the usage of
    difference values and uses local search to improve the solution.
    """
    import random
    import time
    
    start_time = time.time()
    timeout = 115  # Leave some buffer before 120s
    
    # Total number of differences needed
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Simple counting bound: need at least total_diffs distinct differences
    # The differences range from 1 to max_val
    # So max_val must be at least total_diffs
    # But we also need to fit k+1 numbers in [0, max_val]
    
    # Start with a reasonable upper bound
    # The greedy ruler approach gives a baseline
    # Let's try to find a good solution using randomized greedy with restarts
    
    best_rows = None
    best_max = float('inf')
    
    def greedy_construct(seed):
        """Greedy construction with a given random seed for tie-breaking."""
        rng = random.Random(seed)
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
        
        return rows, max(max(row) for row in rows)
    
    # First, get a baseline with deterministic greedy
    baseline_rows, baseline_max = greedy_construct(0)
    best_rows = baseline_rows
    best_max = baseline_max
    
    # Try multiple random seeds to see if we can do better
    # The greedy is deterministic in structure but we can try different orderings
    # Actually, the standard greedy is quite tight. Let's try a different approach:
    # Use a more careful greedy that looks ahead
    
    def greedy_with_lookahead(seed, max_iterations=50000):
        """Greedy with limited lookahead to try to find better packings."""
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try the next few candidates and pick the one that leaves
                # the most room for future differences
                best_x = None
                best_score = -1
                
                # Look at next 20 candidates
                for candidate in range(x + 1, x + 21):
                    new_diffs = {candidate - a for a in row}
                    if new_diffs & used:
                        continue
                    # Score: prefer candidates that use smaller differences
                    # and don't create large gaps
                    score = -candidate  # Prefer smaller
                    # Also penalize if it creates a large unused gap
                    best_x = candidate
                    best_score = score
                    break  # First valid is best in pure greedy
                
                if best_x is None:
                    # Fallback: keep incrementing
                    x += 1
                    new_diffs = {x - a for a in row}
                    if new_diffs & used:
                        continue
                    best_x = x
                else:
                    x = best_x
                
                new_diffs = {x - a for a in row}
                row.append(x)
                used |= new_diffs
            
            rows.append(row)
        
        return rows, max(max(row) for row in rows)
    
    # Try a few random restarts with different strategies
    for seed in range(1, 20):
        if time.time() - start_time > timeout:
            break
        try:
            rows, max_val = greedy_with_lookahead(seed)
            if max_val < best_max:
                best_max = max_val
                best_rows = rows
        except Exception:
            continue
    
    # Try a more sophisticated approach: simulated annealing on the assignment
    # This is complex, so let's try a simple local search
    
    def local_search(rows, max_iters=10000):
        """Try to improve the solution by moving points."""
        if time.time() - start_time > timeout:
            return rows, max(max(row) for row in rows)
        
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        current_max = max(max(row) for row in rows)
        
        for _ in range(max_iters):
            if time.time() - start_time > timeout:
                break
            
            # Pick a random row and a random position to try to move
            row_idx = random.randint(0, len(rows) - 1)
            pos_idx = random.randint(1, len(rows[row_idx]) - 1)  # Don't move 0
            
            old_val = rows[row_idx][pos_idx]
            new_val = old_val + random.randint(-5, 5)
            
            if new_val <= 0 or new_val >= rows[row_idx][-1] + 1:
                continue
            
            # Check if the new value maintains strict increase
            if pos_idx > 0 and new_val <= rows[row_idx][pos_idx - 1]:
                continue
            if pos_idx < len(rows[row_idx]) - 1 and new_val >= rows[row_idx][pos_idx + 1]:
                continue
            
            # Compute new differences for this row
            old_diffs = set()
            for i in range(len(rows[row_idx])):
                for j in range(i + 1, len(rows[row_idx])):
                    old_diffs.add(rows[row_idx][j] - rows[row_idx][i])
            
            # Remove old diffs from used
            for d in old_diffs:
                used.discard(d)
            
            # Apply the change
            rows[row_idx][pos_idx] = new_val
            
            # Compute new differences
            new_diffs = set()
            for i in range(len(rows[row_idx])):
                for j in range(i + 1, len(rows[row_idx])):
                    new_diffs.add(rows[row_idx][j] - rows[row_idx][i])
            
            # Check for conflicts
            if new_diffs & used:
                # Revert
                rows[row_idx][pos_idx] = old_val
                for d in old_diffs:
                    used.add(d)
                continue
            
            # Accept the change
            for d in new_diffs:
                used.add(d)
            
            new_max = max(max(row) for row in rows)
            if new_max < current_max:
                current_max = new_max
        
        return rows, current_max
    
    # Apply local search to the best solution found so far
    if best_rows is not None:
        try:
            improved_rows, improved_max = local_search([row[:] for row in best_rows])
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows
        except Exception:
            pass
    
    return best_rows
# EVOLVE-BLOCK-END
