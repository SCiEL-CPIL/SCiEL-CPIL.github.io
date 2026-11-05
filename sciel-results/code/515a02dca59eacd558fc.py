import sys
import time
import random
from collections import deque

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a local search approach:
    1. Greedy initialization to get a valid starting point.
    2. Iteratively try to reduce the maximum entry by reconstructing rows 
       within a tighter bound.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: m*(m+1)/2 >= total_diffs_needed
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1

    # Greedy baseline
    def greedy_construct(n_rows, k, max_entry=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if max_entry is not None and x > max_entry:
                    return None
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    initial_rows = greedy_construct(n_rows, k)
    if initial_rows is None:
        # Fallback if greedy fails (shouldn't happen for reasonable inputs)
        return [[0] * (k + 1)] * n_rows
        
    best_rows = initial_rows
    best_max = max(max(r) for r in best_rows)
    
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    # Helper to check if a row is valid given used diffs
    def is_row_valid(row, used_diffs):
        row_diffs = get_diffs(row)
        return not (row_diffs & used_diffs)

    # Heuristic to build a single row within a bound, avoiding used_diffs
    def build_row_heuristic(used_diffs, max_entry, k, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        row = [0]
        # We need k more elements
        # We want to keep them small to minimize max entry, but avoid conflicts
        
        # Strategy: Try to place elements one by one.
        # At each step, choose the smallest valid x > row[-1] such that x <= max_entry
        # and no conflict with used_diffs or current row diffs.
        
        # To add diversity, we can occasionally pick a slightly larger valid x.
        
        for _ in range(k):
            last = row[-1]
            found = False
            # Range of candidates: last+1 to max_entry
            # We prioritize small values, but might skip some if they cause
            # "tight" conflicts later? For now, simple greedy with randomization.
            
            candidates = []
            for x in range(last + 1, max_entry + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    candidates.append(x)
                    # Don't collect too many, keep it fast
                    if len(candidates) > 50:
                        break
            
            if not candidates:
                return None
            
            # Usually pick the smallest, but with some probability pick a random one
            # from the top few to escape local minima
            if random.random() < 0.1 and len(candidates) > 1:
                # Pick from the first 10 candidates randomly
                choice = random.choice(candidates[:min(10, len(candidates))])
            else:
                choice = candidates[0]
            
            row.append(choice)
            
        return row

    # Local Search Loop
    # We try to reduce best_max from its current value down to lb
    current_max = best_max
    
    # We will iterate, trying to find a solution with a smaller max entry.
    # We do this by attempting to reconstruct rows within a tighter bound.
    
    # To speed up, we don't try every integer below current_max immediately.
    # We can do a binary search or linear search. Given the small range usually,
    # linear is fine, but we can step down by 1 or 2.
    
    # Let's use a randomized restart strategy for the whole set of rows
    # within a specific max_entry bound.
    
    def try_reduce_max(target_max, time_budget):
        """
        Attempts to find a valid set of rows with all entries <= target_max.
        Uses randomized restarts and row rebuilding.
        Returns list of rows or None.
        """
        deadline = time.time() + time_budget
        
        # We maintain a list of rows and a set of used diffs.
        # We start with a valid set (from previous iteration or greedy)
        # and try to "push" rows to fit within target_max.
        
        # Actually, it's better to start from scratch or from a relaxed state?
        # Starting from scratch with a tight bound is hard.
        # Let's try to modify the existing best_rows.
        
        # However, if best_rows has max > target_max, we must change those rows.
        
        # Strategy:
        # 1. Identify rows that violate the bound (max > target_max).
        # 2. Try to rebuild them one by one, ensuring they fit in target_max
        #    and don't conflict with other rows.
        # 3. If stuck, try shuffling the order of rows or restarting.
        
        # A simpler robust approach:
        # Randomized Greedy Construction with bound.
        # Try to build n_rows rows from scratch, all <= target_max.
        
        for attempt in range(100): # Max attempts
            if time.time() > deadline:
                return None
            
            random.seed(attempt + int(time.time() * 1000) % 10000)
            
            rows = []
            used_diffs = set()
            success = True
            
            # Process rows in a random order? Or fixed?
            # Fixed order might be better for consistency, random for diversity.
            # Let's shuffle the row indices to break symmetry.
            row_order = list(range(n_rows))
            random.shuffle(row_order)
            
            # We need to assign a row to each index.
            # Since rows are indistinguishable in the final output (just a list of lists),
            # we can just build them in any order.
            
            for idx in row_order:
                # Build a row for this slot
                row = build_row_heuristic(used_diffs, target_max, k)
                if row is None:
                    success = False
                    break
                rows.append(row)
                used_diffs |= get_diffs(row)
            
            if success:
                # Check if all rows are valid and within bound
                # (build_row_heuristic already ensures within bound and no conflict with used_diffs)
                # But we need to ensure the rows themselves are strictly increasing and start at 0.
                # build_row_heuristic does this.
                
                # Final validation
                all_diffs = set()
                for r in rows:
                    if r[0] != 0:
                        return None
                    if not all(r[i] < r[i+1] for i in range(len(r)-1)):
                        return None
                    if max(r) > target_max:
                        return None
                    rd = get_diffs(r)
                    if rd & all_diffs:
                        return None
                    all_diffs |= rd
                
                return rows
        
        return None

    # Main loop: Try to reduce the max entry
    # We start from best_max and go down to lb
    # We use a time budget for each target_max
    
    # To be efficient, we can try a few steps down first.
    # If we succeed at M, we set best_max = M and continue to M-1.
    
    # We will iterate target_max from best_max - 1 down to lb.
    # We allocate a small time budget for each.
    
    # However, if best_max is large, we might want to jump.
    # But greedy usually gives a reasonably tight bound.
    
    # Let's do a linear search downwards.
    # We give each step a time slice.
    
    # Total time available: time_limit - elapsed
    # Steps: best_max - lb
    # Time per step: total_time / steps
    
    # But if we find a solution, we can keep going.
    # If we fail, we can try again with more time?
    
    # Let's use a fixed number of restarts for each target_max.
    
    target = best_max - 1
    while target >= lb:
        if time.time() > start_time + time_limit:
            break
            
        # Allocate time based on how close we are to lb?
        # Or just a fixed small amount per target?
        # If we have 100s left and 50 targets, 2s per target.
        
        # Let's try a dynamic budget.
        # If we haven't improved in a while, maybe we are stuck.
        
        # Simple strategy: Try to solve for 'target' with a budget of 1.0s.
        # If successful, update best and continue to target-1.
        # If failed, try target-1? Or skip?
        # If target is infeasible, target-1 is also infeasible.
        # So if we fail, we should probably stop or try a much lower target?
        # Actually, feasibility is monotonic. If M is infeasible, M-1 is infeasible.
        # So if we fail to find a solution for M, we can stop?
        # Wait, the search is not exhaustive. Failing to find a solution in 1s doesn't mean it's infeasible.
        # But if we fail many times, it's likely infeasible or very hard.
        
        # Let's try a few times for the current target.
        
        # Budget: 1.5 seconds per target value.
        budget = 1.5
        result = try_reduce_max(target, budget)
        
        if result is not None:
            best_rows = result
            best_max = target
            # Continue to next lower target
            target -= 1
        else:
            # Failed to find solution for this target within budget.
            # It might be infeasible, or just hard.
            # Let's try one more time with a larger budget?
            # Or just stop?
            # Given the monotonicity, if it's infeasible, lower ones are too.
            # If it's feasible but hard, we might miss it.
            # But we have a valid solution at best_max.
            # Let's try a few more targets quickly?
            # No, if we can't pack it in M, we can't pack it in M-1.
            # So if we believe it's infeasible, we stop.
            # But we don't know.
            # Let's try one more attempt with 3s.
            if time.time() < start_time + time_limit - 5:
                 result = try_reduce_max(target, 3.0)
                 if result is not None:
                     best_rows = result
                     best_max = target
                     target -= 1
                 else:
                     # Assume infeasible or too hard, stop reducing
                     break
            else:
                break

    # Final validation
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True

    if not validate(best_rows, n_rows, k):
        # Fallback to greedy
        best_rows = greedy_construct(n_rows, k)
        if best_rows is None:
            best_rows = [[0] * (k + 1)] * n_rows

    return best_rows

# EVOLVE-BLOCK-END
