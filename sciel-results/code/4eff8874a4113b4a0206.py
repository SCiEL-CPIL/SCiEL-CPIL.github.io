import random
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max entry
    lower_bound = total_diffs
    
    # Helper to compute max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Helper to check validity of a set of rows
    def is_valid(rows):
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        return False
                    used_diffs.add(d)
        return True

    # Initial baseline using pure greedy
    def pure_greedy():
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

    baseline_rows = pure_greedy()
    best_rows = [row[:] for row in baseline_rows]
    best_max = get_max_entry(best_rows)
    
    # Strategy: 
    # 1. Generate a pool of candidate rows (Golomb rulers) with scope <= current_best_max - 1.
    # 2. Build a conflict graph where nodes are candidate rows, edges if they share a difference.
    # 3. Use randomized greedy to pick n_rows non-conflicting rows minimizing max scope.
    
    # To find a better solution, we need to search for rows with smaller max entries.
    # We will iterate with a decreasing scope limit.
    
    # Start from a slightly larger scope to ensure we have candidates, then tighten.
    # The greedy baseline is usually not optimal.
    # Let's try to find a solution with max < best_max.
    
    current_scope_limit = best_max - 1
    
    # We will try to construct a solution with max entry <= current_scope_limit.
    # If we fail, we might need to increase the limit or try different random seeds.
    
    # Since the space of Golomb rulers is large, we generate candidates on the fly or in batches.
    # For small k (like 8), the number of Golomb rulers with scope <= 134 is manageable?
    # Actually, the number of Golomb rulers of order n grows fast.
    # For k=8 (9 marks), scope 134.
    # Let's estimate: Number of Golomb rulers of order n with span m.
    # It's better to generate random valid rows and build the graph incrementally or use a heuristic.
    
    # Given the time constraint, we can't enumerate all.
    # We will use a "pool" of random valid rows.
    
    def generate_random_golomb_row(max_val):
        """Generate a random Golomb ruler (row) with max entry <= max_val."""
        # Start with 0
        row = [0]
        # We need k more elements.
        # To ensure it's a Golomb ruler, we pick elements one by one, checking differences.
        # This is a local search for a single row.
        
        # Simple random generation with backtracking is hard.
        # Let's use a greedy random construction:
        # Pick next element from available values that don't conflict with existing differences.
        
        # To make it diverse, we can shuffle the order of trying values?
        # Or just pick random valid values.
        
        for _ in range(k):
            # Find all valid next values
            # Current max is row[-1]. Next must be > row[-1] and <= max_val.
            valid_vals = []
            current_diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    current_diffs.add(row[j] - row[i])
            
            # We need to pick x > row[-1] such that {x - a for a in row} is disjoint from current_diffs.
            # And also disjoint from each other (which they are if x is fixed and row is fixed? No, x-a are distinct if row is distinct).
            # Wait, x - a_i and x - a_j are distinct if a_i != a_j.
            # So we just need {x - a for a in row} to be disjoint from current_diffs.
            
            # Range of x: row[-1] + 1 to max_val
            # Let's sample a few random x's to speed up?
            # Or iterate? If max_val is small (134), iteration is fast.
            
            found = False
            # Randomize the order of selection to explore different parts of the space
            # We can try random candidates first, then fallback to sequential if needed?
            # For now, let's try a random sample of candidates.
            
            candidates = list(range(row[-1] + 1, max_val + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in row}
                if not new_diffs & current_diffs:
                    row.append(x)
                    found = True
                    break
            
            if not found:
                return None # Failed to construct row with this max_val
            
        return row

    # Main loop: Try to find a valid set of n_rows rows with max entry < best_max
    # We will maintain a pool of candidate rows.
    
    # To improve efficiency, we can try to build the set row by row.
    # Row 1: Pick a random Golomb row with max <= limit.
    # Row 2: Pick a random Golomb row with max <= limit that doesn't conflict with Row 1.
    # ...
    
    # This is a constraint satisfaction problem.
    # Let's use a randomized backtracking search for the set of rows.
    
    def find_set_with_limit(limit, time_limit=5.0):
        """Try to find n_rows non-conflicting Golomb rows with max entry <= limit."""
        start = time.time()
        
        # We will use a recursive backtracking with randomization.
        # State: list of chosen rows, set of used differences.
        
        # To make it efficient, we generate candidates on the fly.
        
        def backtrack(chosen_rows, used_diffs, depth):
            if time.time() > start + time_limit:
                return None
            
            if depth == n_rows:
                return chosen_rows[:]
            
            # Generate a batch of candidate rows for this position
            # We want to try candidates that are "compact" (small max entry) to keep the overall max low?
            # Or just random ones within the limit.
            
            # Let's generate 20 random candidates
            candidates = []
            for _ in range(20):
                row = generate_random_golomb_row(limit)
                if row:
                    # Check if it conflicts with used_diffs
                    row_diffs = set()
                    valid = True
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            d = row[j] - row[i]
                            if d in used_diffs:
                                valid = False
                                break
                            row_diffs.add(d)
                        if not valid:
                            break
                    if valid:
                        candidates.append((row, row_diffs))
            
            # Randomize order
            random.shuffle(candidates)
            
            for row, row_diffs in candidates:
                # Try this row
                new_used = used_diffs | row_diffs
                result = backtrack(chosen_rows + [row], new_used, depth + 1)
                if result:
                    return result
            
            return None
        
        return backtrack([], set(), 0)

    # Try to find a solution with decreasing limits
    # Start from best_max - 1 and go down
    # If we find one, update best_max and best_rows
    
    # We can also try to improve the current best_rows by local search?
    # But the MWIS approach is structurally different.
    
    # Let's run the search for several seconds
    # We will try limits from best_max - 1 down to lower_bound
    
    # To avoid getting stuck, we can randomize the order of limits?
    # No, we want the smallest possible.
    
    # However, finding a solution with a very tight limit is hard.
    # It's often easier to find a valid solution with a looser limit and then tighten.
    # But our generate_random_golomb_row already restricts to the limit.
    
    # Let's try a few restarts with the current best_max - 1 limit first.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Try to find a solution with max < best_max
        # We use a time slice for each attempt
        time_slice = 5.0
        if time.time() + time_slice > deadline:
            time_slice = deadline - time.time()
            if time_slice < 0.1:
                break
        
        sol = find_set_with_limit(best_max - 1, time_slice)
        
        if sol:
            # Verify validity just in case
            if is_valid(sol):
                curr_max = get_max_entry(sol)
                if curr_max < best_max:
                    best_max = curr_max
                    best_rows = [row[:] for row in sol]
                    # If we found a better one, we can try to tighten further in next iterations
                    # The limit for next call will be best_max - 1 automatically
                else:
                    # This shouldn't happen if limit is best_max - 1
                    pass
            else:
                # Should not happen if backtrack logic is correct
                pass
        else:
            # If we couldn't find a solution with best_max - 1, 
            # maybe we should try a slightly larger limit to find *any* valid solution 
            # and then hope to improve? 
            # No, if we can't find one with best_max - 1, we are stuck at best_max 
            # unless we change the search strategy.
            # The current strategy relies on random generation of rows.
            # If the space is sparse, we might fail.
            
            # Alternative: Try to improve the existing best_rows using local search?
            # The prompt asks for a structurally different strategy.
            # The MWIS/Greedy approach is the main one.
            # If it fails, we return the best found so far.
            
            # Maybe try a different random seed or different generation heuristic?
            # For now, just keep trying.
            pass
        
        # If we have been trying for a while and not improving, 
        # we might want to try a looser limit to ensure we don't miss a better solution 
        # that requires a different structure?
        # No, if a solution with max M exists, it will be found with limit M.
        # If we fail to find it, it's a search failure.
        
        # To diversify, we can occasionally try to find a solution with limit = best_max 
        # (which should be easy) and then try to reduce it? 
        # But we already have a solution with best_max.
        
        # Let's add a small probability to try a limit slightly higher than best_max - 1 
        # to explore the neighborhood, in case the random generation is biased?
        # Actually, generate_random_golomb_row(limit) is constrained by limit.
        
        # One improvement: In find_set_with_limit, if we fail, we can try to 
        # reuse some rows from the previous best solution?
        # This is a form of hybrid.
        
        # For now, let's just keep running.
        
        # If we have spent a lot of time without improvement, we can break early?
        # No, keep going until deadline.
        
        # To prevent infinite loops if time_slice is very small, ensure progress.
        if time.time() > deadline:
            break

    return best_rows
# EVOLVE-BLOCK-END
