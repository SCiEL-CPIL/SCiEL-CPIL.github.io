import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Total number of distinct differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Simple counting bound
    lower_bound = int(math.ceil(math.sqrt(2 * total_diffs)))

    def validate(rows):
        """Check if rows are valid: all positive differences distinct across all rows."""
        all_diffs = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def get_score(rows):
        return max(max(row) for row in rows)

    # Strategy 1: Deterministic greedy (fast baseline)
    def greedy_deterministic():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows

    # Get initial solution
    best_solution = greedy_deterministic()
    if not best_solution or not validate(best_solution):
        # Fallback if greedy fails (unlikely for small n_rows, k)
        return []
    
    best_score = get_score(best_solution)
    current_rows = [list(r) for r in best_solution]
    
    # Precompute difference sets for each row and position to speed up move evaluation
    # Actually, for SA, we need to quickly check if a move is valid.
    # A move changes row[r][p] from old_val to new_val.
    # This removes diffs: {old_val - row[r][i] for i < p} and {row[r][j] - old_val for j > p}
    # And adds diffs: {new_val - row[r][i] for i < p} and {row[r][j] - new_val for j > p}
    # We need to ensure:
    # 1. new_val is strictly between neighbors.
    # 2. The set of added diffs does not intersect with (Global Used Diffs - Removed Diffs).
    
    # Maintain a global counter of differences to allow O(1) lookup of "is this diff used elsewhere?"
    # Since we need to remove and add, we can use a dictionary of counts.
    
    def build_diff_counts(rows):
        counts = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    diff_counts = build_diff_counts(current_rows)
    
    # Simulated Annealing
    rng = random.Random(12345)
    T = 100.0
    cooling_rate = 0.9999
    min_T = 0.1
    
    # We want to minimize the max entry.
    # Objective function: max_entry.
    # Acceptance: 
    # If new_score < current_score: Accept.
    # Else: Accept with probability exp(-(new_score - current_score) / T).
    
    # To make it efficient, we only consider moves that potentially reduce the max entry
    # or maintain it. Moves that increase the max entry are generally bad unless they
    # allow a subsequent much larger reduction. However, standard SA allows worsening.
    # Given the discrete nature and the specific objective (minimize max), 
    # "worsening" means increasing the max. We should be very conservative with that.
    
    # Optimization: Only try to move elements that are part of the current maximum?
    # No, moving a non-max element might free up a difference that allows a max element to shrink.
    # But that's indirect.
    
    # Let's stick to a standard SA loop but with a smart move generation.
    
    iterations = 0
    while time.time() < deadline - 5:
        if T < min_T:
            # Reheat occasionally? Or just stop? 
            # Let's do a quick reheat if we've been stuck for a while.
            T = 10.0
            
        # Pick a random row and position
        r = rng.randint(0, n_rows - 1)
        p = rng.randint(1, k) # 1 to k, 0 is fixed
        
        old_val = current_rows[r][p]
        
        # Determine valid range for new_val
        lower = current_rows[r][p-1] + 1
        upper = current_rows[r][p+1] - 1 if p < k else old_val + 50 # Allow expansion? No, we want to minimize.
        
        # If we are trying to minimize the max, we should only try values <= old_val if old_val is the max?
        # Actually, let's try a random value in [lower, upper].
        # To bias towards reduction, if old_val is the current max, we should prefer lower values.
        
        current_max = get_score(current_rows)
        
        # Generate candidate
        # If p is the last element and it's the max, try to shrink it.
        if p == k and old_val == current_max:
            # Try values in [lower, old_val - 1]
            if lower < old_val:
                new_val = rng.randint(lower, old_val - 1)
            else:
                continue
        else:
            # Random value in range, maybe biased towards center or lower
            if upper <= lower:
                continue
            # Bias: 50% chance to pick from lower half, 50% from upper half?
            # Or just uniform.
            new_val = rng.randint(lower, upper)
            
            if new_val == old_val:
                continue
            
        # Check validity
        # Remove old diffs
        removed_diffs = set()
        for i in range(p):
            d = old_val - current_rows[r][i]
            removed_diffs.add(d)
        for j in range(p + 1, len(current_rows[r])):
            d = current_rows[r][j] - old_val
            removed_diffs.add(d)
            
        # Add new diffs
        added_diffs = set()
        for i in range(p):
            d = new_val - current_rows[r][i]
            added_diffs.add(d)
        for j in range(p + 1, len(current_rows[r])):
            d = current_rows[r][j] - new_val
            added_diffs.add(d)
            
        # Check conflicts
        # A diff is valid if it's in added_diffs and:
        # 1. It was in removed_diffs (so count goes back to 1 if it was 2, or 0 if it was 1... wait)
        # We need to check if any d in added_diffs has count > 1 in the *new* state.
        # Current count of d is c.
        # If d is in removed_diffs, new_count = c - 1.
        # If d is in added_diffs, new_count = c + 1.
        # If d is in both, new_count = c.
        # We need new_count == 1 for all d in added_diffs? No, we need new_count >= 1 (always true if added) 
        # and new_count <= 1 (so it's unique).
        # So for every d in added_diffs:
        #   if d in removed_diffs:
        #       # count unchanged, so it must have been 1 (valid)
        #       # Actually if it was 1 and we remove and add, it stays 1. OK.
        #       pass
        #   else:
        #       # count increases by 1. So current count must be 0.
        #       if diff_counts.get(d, 0) != 0:
        #           conflict = True
        #   # Also need to check if d is in removed_diffs but NOT in added_diffs?
        #   # That just reduces count, which is fine as long as it doesn't go to 0? 
        #   # No, differences must exist. If we remove a diff, it's gone from that row. 
        #   # But wait, the diff must be distinct *over all rows*. 
        #   # If a diff appears in row A and we remove it from row A, but it's also in row B, 
        #   # then it's still valid (count 1). 
        #   # The constraint is: all positive differences ... are distinct over all rows.
        #   # This means each difference value can appear AT MOST ONCE in the entire set of rows.
        #   # So count must be 0 or 1.
        #   # If count is 1 and we remove it, count becomes 0. That's fine.
        #   # If count is 1 and we add it, count becomes 2. INVALID.
        #   # If count is 0 and we add it, count becomes 1. VALID.
        
        valid = True
        for d in added_diffs:
            if d in removed_diffs:
                # Count unchanged. If it was valid before, it's valid now.
                # But we must ensure it wasn't 2 before? No, current state is valid, so count is 1.
                # So it stays 1. OK.
                pass
            else:
                # Count increases by 1. Must have been 0.
                if diff_counts.get(d, 0) != 0:
                    valid = False
                    break
        
        if not valid:
            continue
            
        # Move is valid. Calculate new score.
        # Only the max of row r might change.
        # New max of row r:
        # If p < k, max is max(current_rows[r][p+1], new_val) ... wait, strictly increasing.
        # So max is current_rows[r][k] if p < k? No, if p < k, the last element is unchanged.
        # If p == k, max is new_val.
        # Actually, if p < k, the max of the row is unchanged (it's the last element).
        # If p == k, the max of the row becomes new_val.
        
        old_row_max = max(current_rows[r])
        if p == k:
            new_row_max = new_val
        else:
            new_row_max = old_row_max # Unchanged
            
        # Global max
        # If old_row_max was the global max, and we reduced it (p==k and new_val < old_row_max),
        # we need to find the new global max.
        # This is O(n_rows * k).
        
        if new_row_max < old_row_max and old_row_max == current_max:
             # Find new global max
             new_global_max = 0
             for row in current_rows:
                 if r == current_rows.index(row): # This is O(N)
                     continue
                 m = max(row)
                 if m > new_global_max:
                     new_global_max = m
             # Include the modified row's new max
             if new_row_max > new_global_max:
                 new_global_max = new_row_max
             new_score = new_global_max
        else:
             # Max didn't decrease enough to drop below current global max, or wasn't the global max
             if new_row_max > current_max:
                 new_score = new_row_max
             else:
                 new_score = current_max
        
        # Acceptance criterion
        if new_score < best_score:
             # Always accept improvement
             pass_accept = True
        elif new_score == current_max:
             # Accept if T is high enough? Or always accept to explore?
             # Accept with prob 1.0 if T > 1, else exp(...)
             # Since score is integer, diff is 0. exp(0)=1.
             pass_accept = True
        else:
             # new_score > current_max
             delta = new_score - current_max
             if delta == 0:
                 pass_accept = True
             else:
                 # Probability of accepting worse solution
                 p_accept = math.exp(-delta / T) if T > 0 else 0
                 pass_accept = rng.random() < p_accept
                 
        if pass_accept:
            # Perform move
            current_rows[r][p] = new_val
            # Update diff_counts
            for d in removed_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in added_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            current_max = new_score
            if new_score < best_score:
                best_score = new_score
                best_solution = [list(row) for row in current_rows]
                
            T *= cooling_rate
            iterations += 1
        else:
            # Reject
            T *= cooling_rate
            iterations += 1

    return best_solution

