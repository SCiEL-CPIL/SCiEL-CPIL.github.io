import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = 148.0  # Safety margin before 150s kill
    
    num_elements = k + 1  # each row has k+1 elements
    # Number of differences per row: C(k+1, 2) = (k+1)k/2
    diffs_per_row = num_elements * (k) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # If the problem is small enough, we could theoretically brute force, 
    # but for (7, 5) -> 7 rows, 6 elements, 15 diffs/row, 105 total diffs.
    # Scope is around 112.
    
    def get_diffs(row):
        """Return the set of differences for a single row."""
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}
    
    def is_valid(rows):
        """Check global validity."""
        all_diffs = set()
        for row in rows:
            diffs = get_diffs(row)
            if all_diffs & diffs:
                return False
            all_diffs |= diffs
        return True

    def get_scope(rows):
        return max(max(row) for row in rows)
    
    def generate_initial_solution():
        """
        Generate a valid initial solution using a simple greedy method.
        This ensures we start from a valid state for the annealing.
        """
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            current = 0
            # Greedy: pick smallest valid next number
            while len(row) < num_elements:
                current += 1
                new_diffs = {current - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(current)
                    used_diffs |= new_diffs
            rows.append(row)
        return rows

    # 1. Get a valid baseline
    best_rows = generate_initial_solution()
    best_scope = get_scope(best_rows)
    current_rows = [row[:] for row in best_rows]
    current_scope = best_scope
    
    rng = random.Random(42)
    
    # Simulated Annealing Parameters
    # We want to minimize scope. 
    # State: list of rows.
    # Move: 
    #   1. Pick a random row.
    #   2. Pick a random element (not the first, not the last maybe?) or just perturb the whole row?
    #   Perturbing a single element is hard because it breaks increasing order and many diffs.
    #   Better Move: "Re-greedy" a specific row with a slight bias towards smaller numbers, 
    #   or "Swap" two rows? No, rows are distinct.
    #   
    #   Let's use a "Repair" strategy:
    #   1. Pick a random row r.
    #   2. Remove row r from the set.
    #   3. Reconstruct row r using a randomized greedy that tries to keep it small, 
    #      but allows some "jumps" to avoid conflicts.
    #   4. If the new row is valid with the rest, accept if scope improves or with prob exp(-delta/T).
    #
    #   To improve scope, we need to reduce the max element.
    #   The "bottleneck" is usually the last row or rows with large gaps.
    
    def rebuild_row(used_diffs, target_max_hint=None, rng=None):
        """
        Rebuild a single row given the set of diffs already used by other rows.
        Uses a randomized greedy approach.
        """
        row = [0]
        current = 0
        # We want to keep the row small.
        # Strategy: try to increment by 1, but if conflict, jump.
        # To allow flexibility, we can randomly decide to "skip" a few numbers.
        
        while len(row) < num_elements:
            found = False
            # Try a few candidates starting from current + 1
            # Randomize the search range slightly to escape local traps
            max_trial = 10
            for _ in range(max_trial):
                # Randomly choose a step size between 1 and some small number?
                # Or just try current+1, current+2... but with a random cutoff?
                # Let's try a "biased random" walk:
                # candidate = current + rng.randint(1, 5)
                # But we must ensure increasing order.
                
                # Actually, simple randomized greedy:
                # Try candidates in random order within a window?
                # No, that's expensive.
                
                # Let's stick to sequential but with a random "restart" of the window?
                # Better: Just try current+1. If conflict, try current+2, etc.
                # To make it "randomized", we can occasionally accept a larger jump earlier.
                
                candidate = current + 1
                while True:
                    new_diffs = {candidate - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(candidate)
                        used_diffs |= new_diffs
                        current = candidate
                        found = True
                        break
                    candidate += 1
                    # Safety break if we go too far? 
                    # If we go too far, this row is bad.
                    if candidate > 200: 
                        break
                
                if found:
                    break
                else:
                    # Failed to find a small candidate, give up on this rebuild?
                    # Or return a very large row?
                    # If we return a large row, scope increases.
                    # Let's return None to indicate failure to build a "good" row.
                    return None
            if not found:
                return None
        return row

    def try_rebuild_row(rows, idx, used_diffs_others):
        """
        Attempt to rebuild row at index idx.
        Returns the new row or None if failed to find a valid one within reasonable bounds.
        """
        # We want to find a row that fits in the "gaps" of used_diffs_others.
        # We use a randomized greedy.
        # To encourage smaller scope, we prefer smaller numbers.
        # We can use a "temperature" to allow larger jumps?
        # For now, simple randomized greedy:
        
        # Let's try multiple random seeds for the greedy process?
        # The greedy process is deterministic if we always pick the smallest valid.
        # To randomize, we can shuffle the "preference" of candidates?
        # But candidates must be increasing.
        
        # Alternative: 
        # The set of valid differences is fixed. 
        # We are looking for a set of k+1 numbers.
        
        # Let's use a simple random restart of the greedy for the specific row.
        # Since the greedy is deterministic (smallest valid), it will always produce the same row.
        # So "rebuilding" a row with the same used_diffs_others will always yield the same row.
        # This means simple rebuild is useless.
        
        # We need to change the "context" or the "strategy".
        # Strategy: "Swap" two rows?
        # If we swap row A and row B, the used_diffs_others for A changes (it now sees B's diffs instead of A's? No, A's diffs are removed, B's added).
        # This doesn't change the set of used diffs globally, so the "greedy" result for any row would be the same if we just swapped them?
        # No, the greedy depends on the *order* in which rows were built? 
        # In my `generate_initial_solution`, I built them sequentially.
        # If I rebuild row i, I use the diffs of all OTHER rows.
        # If I swap row i and row j, the set of "other" diffs for the new row i (old j) is the same as for the new row j (old i)?
        # No. 
        # Old state: Rows R1..Rn. Used diffs = U.
        # Rebuild R1: Remove R1's diffs from U. Build new R1' based on U\R1.
        # If I swap R1 and R2, the set of diffs in the system is the same.
        # But the "greedy" reconstruction of a row depends on the specific set of available diffs.
        # Since the set of available diffs is the same (U \ R_i), the greedy reconstruction is deterministic and will always produce the same row for a given "hole" in the difference set.
        
        # Therefore, simple "rebuild" with deterministic greedy is a no-op (it will just reconstruct the same row).
        # We MUST introduce randomness into the row generation.
        
        # Randomized Row Generation:
        # Instead of always picking the smallest valid next number, 
        # pick a random valid next number from a window [current+1, current+K].
        
        def randomized_greedy_row(used_diffs, max_val_limit=200, rng=None):
            row = [0]
            current = 0
            while len(row) < num_elements:
                # Find all valid candidates in a window
                candidates = []
                # Window size increases as we go? Or fixed?
                # Let's try a window of size 10 initially, expanding if stuck.
                window_size = 10
                while len(candidates) == 0 and window_size < max_val_limit:
                    for c in range(current + 1, current + window_size + 1):
                        new_diffs = {c - a for a in row}
                        if not (new_diffs & used_diffs):
                            candidates.append(c)
                    window_size *= 2
                
                if not candidates:
                    return None
                
                # Pick a candidate
                # Bias towards smaller candidates to minimize scope
                # Use exponential distribution?
                # Or just pick the smallest with probability 0.8, random with 0.2?
                if rng.random() < 0.7:
                    choice = candidates[0]  # Smallest
                else:
                    choice = rng.choice(candidates)
                
                row.append(choice)
                current = choice
                # Update used_diffs locally? 
                # No, we need to check against the GLOBAL used_diffs (others) + the diffs of the row we are building.
                # My `new_diffs` check above only checked against `used_diffs` (others).
                # I need to also check against diffs already in `row`.
                # Wait, `new_diffs = {c - a for a in row}` checks against current row.
                # But I didn't add `new_diffs` to a local "row_used" set.
                # The check `if not (new_diffs & used_diffs)` is only against OTHERS.
                # I need to ensure `new_diffs` are distinct within the row too.
                # Since `row` is strictly increasing, diffs within the row are distinct?
                # No, 0, 1, 3 -> diffs 1, 3, 2. All distinct.
                # 0, 2, 4 -> diffs 2, 4, 2. Duplicate!
                # So I MUST check for duplicates within the row.
                
                # Let's refine the candidate check.
                # We need `new_diffs` to be disjoint from `used_diffs` (others) AND `row_diffs_so_far`.
                
                # Let's restart the function logic properly.
                pass
            
            return row

        # Corrected Randomized Greedy
        def gen_row(used_diffs_others, rng):
            row = [0]
            row_diffs = set()
            current = 0
            for _ in range(num_elements - 1):
                candidates = []
                # Search in a window
                # We want to keep the row small, so we search small numbers first.
                # But we must allow some randomness to escape local minima.
                
                # Let's try candidates from current+1 up to current+20.
                # If none work, expand window.
                found = False
                limit = current + 20
                while not found and limit < 200:
                    for c in range(current + 1, limit + 1):
                        new_diffs = {c - a for a in row}
                        # Check against others
                        if new_diffs & used_diffs_others:
                            continue
                        # Check against self
                        if new_diffs & row_diffs:
                            continue
                        candidates.append(c)
                    if candidates:
                        found = True
                        break
                    limit *= 2
                    # Cap limit to avoid infinite loop if impossible
                    if limit > 500:
                        break
                
                if not candidates:
                    return None
                
                # Selection
                # Bias to smallest
                if rng.random() < 0.8:
                    c = candidates[0]
                else:
                    c = rng.choice(candidates)
                
                row.append(c)
                current = c
                # Update row_diffs
                # We can just recompute or add the new ones.
                # new_diffs was computed for c.
                # We need to add them to row_diffs.
                # But `new_diffs` was a local var in the loop. 
                # Let's recompute or store it.
                # For efficiency, let's just recompute the set of diffs for the row? 
                # No, that's O(k^2). k is small (6). It's fine.
                row_diffs = get_diffs(row)
                
            return row

        # Now, the SA loop.
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.99
        
        current_rows = [row[:] for row in best_rows]
        current_scope = get_scope(current_rows)
        
        # Pre-compute diffs for all rows to speed up "others" calculation?
        # Or just compute on the fly.
        
        steps = 0
        while time.time() < DEADLINE:
            # Pick a random row to rebuild
            idx = rng.randint(0, n_rows - 1)
            
            # Calculate used_diffs_others
            # This is O(n_rows * k^2). 
            # For n=7, k=6, this is fast.
            all_diffs = set()
            for i, row in enumerate(current_rows):
                if i == idx:
                    continue
                all_diffs |= get_diffs(row)
            
            # Generate a new row
            new_row = gen_row(all_diffs, rng)
            
            if new_row is None:
                # Failed to generate a valid row. 
                # This is a "hard" constraint violation. 
                # In SA, we usually avoid hard violations by only proposing valid moves.
                # If gen_row fails, we can't propose this move.
                # We can try a different random seed or just skip this step.
                continue
            
            # Calculate new scope
            # The scope is max(max(row) for all rows).
            # Only row idx changed.
            old_max = current_rows[idx][-1]
            new_max = new_row[-1]
            
            # Other rows' maxes are unchanged.
            # We need the global max.
            # Let's keep track of the max of other rows.
            # For simplicity, just compute get_scope(new_rows).
            
            temp_rows = current_rows[:]
            temp_rows[idx] = new_row
            new_scope = get_scope(temp_rows)
            
            delta = new_scope - current_scope
            
            if delta < 0 or rng.random() < pow(2.71828, -delta / T):
                # Accept
                current_rows = temp_rows
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [row[:] for row in current_rows]
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # Reheat?
                # If we hit T_min, we might be stuck. 
                # Let's restart T if we haven't improved in a while.
                # Simple reheat:
                if rng.random() < 0.1:
                    T = 5.0
            
            steps += 1
            
            # Check time
            if time.time() > DEADLINE - 2.0:
                break
        
        # Final check
        if is_valid(best_rows):
            return best_rows
        else:
            # Fallback
            return generate_initial_solution()

# EVOLVE-BLOCK-END
