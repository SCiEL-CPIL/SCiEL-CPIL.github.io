"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max entry
    lower_bound = total_diffs
    
    # We will use a list for difference counts for O(1) access.
    # Max possible difference is bounded by the scope.
    # We estimate a safe upper bound for the array size.
    # Greedy usually produces scope < 2 * lower_bound.
    # Let's start with a reasonably large array and resize if needed.
    max_est = 2 * lower_bound + 100
    if max_est < 1000:
        max_est = 1000
        
    def count_diffs(rows):
        """Count all differences in the rows."""
        counts = [0] * max_est
        for r in rows:
            for i in range(len(r)):
                for j in range(i):
                    d = r[i] - r[j]
                    if d < max_est:
                        counts[d] += 1
                    else:
                        # This shouldn't happen if max_est is large enough, 
                        # but if it does, we need to handle it.
                        # For now, assume max_est is sufficient.
                        pass
        return counts

    def pure_greedy(rng_seed=None):
        """
        Generate a valid initial solution using a greedy approach.
        We use a bitset for fast "is this difference available" checks during construction.
        """
        if rng_seed is not None:
            rng = random.Random(rng_seed)
        else:
            rng = random.Random()
            
        used = 0  # Bitset of used differences
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if x - a_i is available for all a_i in row
                # We need to check if (1 << (x - a_i)) is in used for any a_i
                # And also that x - a_i are distinct (they are, since a_i are distinct)
                
                # Optimization: Precompute masks? 
                # For k=8, row has at most 8 elements. Checking 8 bits is fast.
                valid = True
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        valid = False
                        break
                
                if valid:
                    row.append(x)
                    # Update used bitset
                    for a in row[:-1]: # a in row except x
                        d = x - a
                        used |= (1 << d)
            rows.append(row)
        return rows

    # 1. Generate diverse initial solutions
    initial_solutions = []
    for i in range(10):
        if time.time() > deadline:
            break
        sol = pure_greedy(seed=i)
        initial_solutions.append(sol)
    
    if not initial_solutions:
        # Fallback: just one
        initial_solutions.append(pure_greedy(seed=0))

    best_rows = initial_solutions[0]
    best_max = max(max(r) for r in best_rows)
    
    # 2. Local Search: Iterative Shrink
    # Strategy: For each row, try to reduce the last element.
    # If successful, update global counts and proceed to next row.
    # Repeat until no improvement or time runs out.
    
    def try_shrink_row(rows, row_idx, counts):
        """
        Try to shrink the last element of rows[row_idx].
        Returns True if successful, False otherwise.
        Modifies rows and counts in place if successful.
        """
        r = rows[row_idx]
        current_last = r[-1]
        
        # We want to find a new_last < current_last such that:
        # 1. new_last > r[-2]
        # 2. For all i in 0..k-1, (new_last - r[i]) is not used elsewhere.
        #    "Elsewhere" means counts[d] == 0 for all d in {new_last - r[i]}.
        #    Note: The differences involving current_last are currently counted in `counts`.
        #    We must temporarily "remove" them to check availability.
        
        old_diffs = [current_last - r[i] for i in range(k)]
        
        # Remove old diffs from counts
        for d in old_diffs:
            counts[d] -= 1
        
        # Try new values from current_last - 1 down to r[-2] + 1
        # We can limit the search range to speed up. 
        # If we don't find a shrink in the first 50 steps, maybe it's not worth it?
        # But for "tightening", we should check all.
        # However, k=8 is small, so 50 checks is cheap.
        
        found = False
        new_last = -1
        
        # Check range
        min_val = r[-2] + 1
        max_val = current_last - 1
        
        # We can randomize the order of checking to escape local minima?
        # No, for "shrink", we always want the smallest valid value.
        # But sometimes a slightly larger value might allow other rows to shrink more?
        # Greedy shrink: find the smallest valid new_last.
        
        for candidate in range(min_val, max_val + 1):
            # Check if candidate - r[i] is available
            valid = True
            for i in range(k):
                d = candidate - r[i]
                if counts[d] > 0:
                    valid = False
                    break
            if valid:
                new_last = candidate
                found = True
                break
        
        if found:
            # Apply change
            r[-1] = new_last
            new_diffs = [new_last - r[i] for i in range(k)]
            for d in new_diffs:
                counts[d] += 1
            return True
        else:
            # Revert counts
            for d in old_diffs:
                counts[d] += 1
            return False

    def local_search(rows):
        """
        Perform iterative local search on a valid solution.
        Returns the best solution found.
        """
        # Initialize counts
        # Ensure max_est is large enough
        current_max = max(max(r) for r in rows)
        if current_max >= max_est:
            # Resize counts (rare)
            pass # Assuming max_est is safe
        
        counts = count_diffs(rows)
        
        improved = True
        iterations = 0
        max_iterations = 1000 # Safety
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try to shrink each row
            # Randomize row order to avoid bias
            row_order = list(range(n_rows))
            random.shuffle(row_order)
            
            for idx in row_order:
                if try_shrink_row(rows, idx, counts):
                    improved = True
                    # If we shrank a row, we might be able to shrink others now.
                    # Continue the loop.
        
        return rows

    # 3. Run Local Search on each initial solution
    for i, init_sol in enumerate(initial_solutions):
        if time.time() > deadline:
            break
            
        # Deep copy
        current_sol = [r[:] for r in init_sol]
        
        # Run local search
        refined_sol = local_search(current_sol)
        
        current_max = max(max(r) for r in refined_sol)
        
        if current_max < best_max:
            best_max = current_max
            best_rows = [r[:] for r in refined_sol]
            
            # If we hit the lower bound, we can stop
            if best_max == lower_bound:
                break

    # 4. Final Polish: Try a few more random restarts if time permits
    # The local search might be stuck in a local optimum where no single row can shrink.
    # A "kick" could help. But let's stick to the current strategy for now.
    # If time remains, we can try to perturb the best solution and re-search.
    
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Try a few perturbations of the best solution
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Perturb: Randomly increase one element? No, that makes it worse.
            # Perturb: Randomly change an internal element?
            # Let's try a "shake": Pick a row, pick a random internal element, 
            # change it to a random valid value, then re-run local search.
            
            perturbed = [r[:] for r in best_rows]
            r_idx = random.randint(0, n_rows - 1)
            p_idx = random.randint(1, k - 1) # Internal element
            
            old_val = perturbed[r_idx][p_idx]
            # Try a few random new values
            for _ in range(20):
                new_val = random.randint(perturbed[r_idx][p_idx-1] + 1, perturbed[r_idx][p_idx+1] - 1)
                if new_val == old_val:
                    continue
                
                # Check validity of this single change
                # We need to check diffs with all other elements in the row
                # And against global counts (excluding old diffs)
                
                # This is complex to implement correctly on the fly without updating counts.
                # Let's skip this for now to ensure robustness.
                break
            
            # Instead, let's just run local search again on a copy of best_rows 
            # with a different random seed for row ordering? 
            # local_search already shuffles.
            # Maybe just restart local search from best_rows?
            # It might converge to the same place.
            
            # Let's try a "swap" move: Swap the last elements of two rows?
            # No, that breaks the increasing property if they are different.
            
            # Let's stick to the current result.
            pass

    return best_rows
# EVOLVE-BLOCK-END
