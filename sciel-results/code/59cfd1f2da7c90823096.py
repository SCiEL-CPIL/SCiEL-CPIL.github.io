import time
import random

def construct(n_rows, k):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Theoretical lower bound on the max element
    # Total distinct differences required: n_rows * (k+1 choose 2)
    # These differences must be in {1, ..., M}
    # So M >= n_rows * (k+1 choose 2)
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    best_scope = float('inf')
    best_rows = None
    
    # Parameters for randomized greedy
    # c: number of candidate next elements to consider
    # We can vary this or keep it fixed.
    c = 10
    
    seed = 0
    
    while time.time() < deadline:
        rows = _randomized_greedy(n_rows, k, seed, c)
        
        # Calculate scope
        scope = 0
        for row in rows:
            if row:
                scope = max(scope, row[-1])
        
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            
            # If we hit the lower bound, we can't do better
            if best_scope == lower_bound:
                break
        
        seed += 1
        
        # Optional: If we have been running for a while without improvement, 
        # we might increase randomness or change strategy.
        # For now, simple restarts are robust.
        
    # If no solution found (should not happen), return empty or greedy
    if best_rows is None:
        best_rows = _randomized_greedy(n_rows, k, 0, 1)
        
    return best_rows

def _randomized_greedy(n_rows, k, seed, c):
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            # Find the next c valid x's
            valid_xs = []
            last_x = row[-1]
            # We search up to a limit. 
            # The max element is likely < 2 * lower_bound.
            # Let's use a dynamic limit or a fixed high limit.
            # Fixed high limit is safer.
            limit = 500 
            
            x = last_x + 1
            while len(valid_xs) < c and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                # If we didn't find any (should be rare), expand limit
                x = last_x + 1
                while len(valid_xs) < c:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
            
            # Pick one
            chosen_x = rng.choice(valid_xs)
            
            # Update row and used
            row.append(chosen_x)
            # Calculate diffs with the previous elements
            # The new diffs are chosen_x - a for a in row[:-1]
            # Note: row[:-1] is the state before appending chosen_x
            # We can use the set difference logic
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            
        rows.append(row)
        
    return rows
