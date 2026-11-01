import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin
    
    # 1. Precompute valid rows (Sidon sets) up to a reasonable bound
    # The best known is 112. Greedy might be higher. Let's go up to 150.
    max_bound = 150
    valid_rows = []
    # Generate all subsets of {0..max_bound} of size k+1 starting with 0
    # This is too many if we just use combinations.
    # We need a backtracking generator for Sidon sets.
    
    def gen_sidon_sets():
        # Returns list of (row_tuple, max_val, diff_set)
        res = []
        def backtrack(start, current_row, used_diffs_in_row):
            if len(current_row) == k + 1:
                res.append((tuple(current_row), current_row[-1], frozenset(used_diffs_in_row)))
                return
            
            # Pruning: if we can't fit remaining numbers, return
            # Simple bound: remaining slots need at least (slots-1) more steps of 1
            # But Sidon constraint is stronger.
            
            for x in range(start, max_bound + 1):
                # Check if x can be added
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs_in_row:
                    continue
                
                current_row.append(x)
                used_diffs_in_row |= new_diffs
                backtrack(x + 1, current_row, used_diffs_in_row)
                current_row.pop()
                used_diffs_in_row -= new_diffs # Careful with set modification in place
                # Actually, passing sets by reference and modifying is tricky.
                # Better to pass copies or use a list and convert.
                # For k=6, depth is small.
        
        # Start with 0
        backtrack(1, [0], set())
        return res

    # This backtracking might be slow if max_bound is 150 and k=6.
    # Let's check complexity.
    # Branching factor is roughly max_bound. Depth 6.
    # 150^6 is 10^12. Too slow.
    # We need a tighter bound or a better generator.
    
    # Alternative: Don't precompute ALL. 
    # Use the greedy to get a solution.
    # Then use local search with on-the-fly candidate generation or a small precomputed set.
    
    # Let's try a different approach for candidates:
    # Generate a pool of "good" rows using randomized greedy with different seeds.
    # Pool size ~ 1000.
    
    pool = []
    for _ in range(2000):
        # Randomized greedy
        row = [0]
        used = set()
        for _ in range(k):
            # Try candidates in random order? Or just greedy?
            # Greedy picks smallest. Randomized picks from top few?
            # Let's stick to greedy but with a random shift? No, must start at 0.
            # We can randomize the "priority" of available numbers.
            
            # Simpler: Just use the standard greedy for the first few, 
            # but for the pool, we can generate random valid rows?
            # Generating a random valid Sidon set is non-trivial.
            
            # Let's just generate rows using a "biased" greedy.
            # Pick next x.
            # Standard greedy picks smallest x.
            # We can pick x from a range [min_valid, min_valid + 10] randomly?
            
            min_valid = row[-1] + 1
            valid_candidates = []
            while True:
                new = {min_valid - a for a in row}
                if not new & used:
                    valid_candidates.append(min_valid)
                    if len(valid_candidates) >= 10:
                        break
                min_valid += 1
                if min_valid > max_bound:
                    break
            
            if not valid_candidates:
                break
            
            # Pick one randomly
            x = random.choice(valid_candidates)
            row.append(x)
            used |= {x - a for a in row[:-1]}
        
        if len(row) == k + 1:
            # Store
            d = frozenset(row[j] - row[i] for i in range(k+1) for j in range(i+1, k+1))
            pool.append((tuple(row), row[-1], d))
    
    # Remove duplicates?
    # Sort by max
    pool.sort(key=lambda x: x[1])
    
    # 2. Initial Solution
    # Use a greedy construction to get a valid starting set.
    # We can use the pool to help? 
    # Simple greedy is fast.
    
    def greedy_construct():
        used_global = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            used_local = set()
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used_local and not new & used_global:
                    row.append(x)
                    used_local |= new
            rows.append(tuple(row))
            used_global |= used_local
        return rows, used_global

    best_rows, best_used = greedy_construct()
    best_max = max(r[-1] for r in best_rows)
    
    # 3. Local Search
    # We maintain a list of rows.
    # We try to improve.
    
    # To make local search effective, we need a good pool of replacement rows.
    # The pool generated above is random.
    # We should also include the rows from the greedy solution.
    
    # Let's add a function to check if a row can replace an existing one.
    
    def can_replace(row_idx, new_row, current_rows, current_used):
        # current_used is a set of all diffs
        # We need to check if new_row's diffs are disjoint from (current_used - diffs of current_rows[row_idx])
        old_row = current_rows[row_idx]
        old_diffs = {old_row[j] - old_row[i] for i in range(k+1) for j in range(i+1, k+1)}
        
        # If new_row is same as old_row, no change
        if new_row == old_row:
            return False
            
        new_diffs = {new_row[j] - new_row[i] for i in range(k+1) for j in range(i+1, k+1)}
        
        # Check intersection
        # Forbidden = current_used - old_diffs
        # We want new_diffs & Forbidden == empty
        # Equivalent to: new_diffs & (current_used | old_diffs) == new_diffs & old_diffs? No.
        # new_diffs must not intersect with current_used \ old_diffs.
        # So: new_diffs & current_used must be subset of old_diffs.
        # Actually, simpler:
        # Remove old_diffs from current_used -> forbidden
        # Check new_diffs & forbidden
        
        forbidden = current_used.difference(old_diffs)
        if new_diffs & forbidden:
            return False
        return True

    # Optimization: Precompute diffs for all rows in pool.
    # Pool entries: (row, max_val, diff_set)
    
    # Main loop
    while time.time() < deadline:
        # Identify the row with the maximum value
        max_val_in_sol = max(r[-1] for r in best_rows)
        # Find all rows with this max value
        candidates_to_replace = [i for i, r in enumerate(best_rows) if r[-1] == max_val_in_sol]
        
        if not candidates_to_replace:
            break
            
        # Pick one randomly
        target_idx = random.choice(candidates_to_replace)
        target_row = best_rows[target_idx]
        
        # Try to find a replacement from the pool
        # We want a row with max_val <= max_val_in_sol (preferably <)
        # And valid diffs.
        
        found = False
        # Iterate through pool. 
        # To speed up, we can filter pool by max_val < max_val_in_sol first.
        # If none, try max_val == max_val_in_sol.
        
        # Pre-filtering might be expensive if done every step.
        # But pool is ~2000. 2000 checks is fast.
        
        # Shuffle pool to avoid bias?
        # random.shuffle(pool) is expensive if pool is large, but 2000 is ok.
        # But we don't want to shuffle the sorted order if we want to try smaller ones first.
        # Let's try to find a row with smaller max first.
        
        best_replacement = None
        best_replacement_max = max_val_in_sol + 1
        
        for p_row, p_max, p_diffs in pool:
            if p_max >= best_replacement_max:
                continue # We want strictly smaller or equal, but we are looking for improvement
            # Actually, if we find one with p_max < max_val_in_sol, that's good.
            # If we find one with p_max == max_val_in_sol, it might help if it frees up diffs.
            
            # Check validity
            # We need to compute forbidden set.
            # This is the bottleneck.
            
            # Optimization: 
            # Instead of computing forbidden set from scratch every time,
            # we can pre-compute the union of diffs of all OTHER rows.
            
            # Let's restructure:
            # For a fixed target_idx, the forbidden set is constant for all candidates.
            # So compute it once outside the loop.
            
            pass # Placeholder

        # Let's implement the "compute forbidden once" logic.
        
        # 1. Compute forbidden set for target_idx
        # forbidden = union of diffs of all rows except target_idx
        forbidden = set()
        for i, r in enumerate(best_rows):
            if i == target_idx:
                continue
            # Compute diffs of r
            # We can precompute diffs for best_rows?
            # Let's maintain a list of diff sets for best_rows.
            
            # For now, compute on the fly.
            d = {r[j] - r[i] for i in range(k+1) for j in range(i+1, k+1)}
            forbidden |= d
            
        # 2. Iterate pool
        # We want to minimize the new max.
        # So sort pool by max? It is already sorted.
        # But we want to try smaller ones first.
        
        # We can break early if we find a valid row with max < current max.
        # But we might want the *smallest* valid row.
        
        # Let's iterate through pool.
        # Since pool is sorted by max, the first valid one we find will have the smallest max.
        
        for p_row, p_max, p_diffs in pool:
            if p_max > max_val_in_sol:
                break # No point in trying larger ones
            
            # Check if p_diffs intersects forbidden
            if not (p_diffs & forbidden):
                # Valid!
                # Update solution
                # Replace target_idx with p_row
                # Update best_rows
                # Update best_used?
                # We don't explicitly maintain best_used set, we recompute forbidden each time.
                # This is O(n_rows * k^2) per iteration.
                # n_rows=7, k=6. 7 * 36 = 256 ops.
                # Pool size 2000.
                # 2000 * 256 = 512,000 ops per step.
                # In Python, this might be slow. 500k ops is ~0.1s?
                # 120s / 0.1s = 1200 steps.
                # This is feasible.
                
                best_rows[target_idx] = p_row
                found = True
                break
        
        if not found:
            # Try to swap with a row of same max but different diffs?
            # The loop above included p_max == max_val_in_sol.
            # If no replacement found, we are stuck at this max.
            # We can try a random restart.
            
            # Restart logic
            if random.random() < 0.01: # 1% chance to restart
                best_rows, _ = greedy_construct()
                best_max = max(r[-1] for r in best_rows)
                
    return [list(r) for r in best_rows]
