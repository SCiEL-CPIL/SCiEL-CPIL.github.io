# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a multi-start greedy construction with dynamic "difference pressure" ordering.
    Rows are built sequentially by selecting the next smallest candidate entry that
    minimizes the increase in the maximum used difference, thereby naturally packing
    differences into the lower range without requiring global backtracking.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: We need 'total_diffs_needed' distinct differences.
    # The differences are integers from 1 to M.
    # So M must be at least total_diffs_needed.
    lb = total_diffs_needed
    
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Greedy baseline (simple)
    def greedy_construct(n_rows, k):
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
        return rows
    
    # Dynamic greedy with difference pressure
    # Strategy: For each row, pick the next element x that minimizes the "cost"
    # Cost is primarily the new maximum difference introduced, secondarily x itself.
    # This packs differences densely.
    
    def dynamic_greedy(n_rows, k, max_entry, seed=0):
        """
        Build rows sequentially. For each position, evaluate all valid candidates x
        and pick the one that minimizes (new_max_diff, x).
        Returns rows or None if max_entry is too small.
        """
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            current_max_diff = 0
            
            while len(row) < k + 1:
                last = row[-1]
                # We need to fit (k+1 - len(row)) more elements.
                # The next element x must be at least last + 1.
                # The last element of the row must be <= max_entry.
                # So x <= max_entry - (k + 1 - len(row) - 1)
                
                remaining_slots = k + 1 - len(row)
                min_x = last + 1
                max_x = max_entry - (remaining_slots - 1)
                
                if min_x > max_x:
                    return None
                
                best_x = None
                best_cost = None
                best_diffs = None
                
                # To add some randomness without breaking the "greedy" nature too much,
                # we can slightly perturb the cost or randomize tie-breaking.
                # But strict greedy (min max_diff) is a strong heuristic for packing.
                
                # Iterate through candidates.
                # Range size is max_x - min_x + 1.
                # For small max_entry, this is small.
                # For larger max_entry, this can be large, but we break early if we find a good one?
                # No, we need to check all to find the min cost.
                # Optimization: If we find a candidate with new_max_diff == current_max_diff (i.e., no new max diff introduced),
                # that's ideal. We can stop searching for better max_diffs and just pick the smallest x among those?
                # Actually, picking smallest x among those that don't increase max_diff is good for packing.
                
                found_no_increase = False
                best_x_no_increase = None
                
                for x in range(min_x, max_x + 1):
                    # Check validity
                    valid = True
                    new_diffs = []
                    new_max = current_max_diff
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                        if d > new_max:
                            new_max = d
                    
                    if not valid:
                        continue
                    
                    # Cost: (new_max, x)
                    # We want to minimize new_max first, then x.
                    
                    if new_max == current_max_diff:
                        # This is the best possible max_diff outcome.
                        # We prefer smaller x to keep the row tight.
                        if best_x_no_increase is None or x < best_x_no_increase:
                            best_x_no_increase = x
                            best_diffs = new_diffs
                        # We can't do better than current_max_diff, so we can keep looking for smaller x?
                        # Yes, smaller x is better.
                        # But we don't need to look for other x with same max_diff if we just want the smallest x?
                        # Actually, if we find x, any larger x with same max_diff is worse.
                        # So we can record the smallest x that doesn't increase max_diff.
                        # But we must finish the loop to ensure no smaller x was missed?
                        # We are iterating x in increasing order.
                        # So the FIRST valid x that doesn't increase max_diff is the best.
                        if best_x_no_increase is None:
                            best_x_no_increase = x
                            best_diffs = new_diffs
                            # We can break if we want strictly smallest x with no increase?
                            # Yes, since x is increasing, the first one found is the smallest.
                            # However, is it always optimal to pick the smallest x that doesn't increase max_diff?
                            # It keeps the row compact.
                            # Let's assume yes for now.
                            # But wait, maybe a slightly larger x that doesn't increase max_diff opens up better options later?
                            # Unlikely, since smaller x leaves more room.
                            # Let's break to save time.
                            break
                    else:
                        # new_max > current_max_diff
                        # This is worse than any solution that doesn't increase max_diff.
                        # But if no solution doesn't increase max_diff, we need the best among these.
                        # Cost: (new_max, x)
                        if best_cost is None or (new_max, x) < best_cost:
                            best_cost = (new_max, x)
                            best_x = x
                            best_diffs = new_diffs
                            # If new_max is very large, we might want to stop?
                            # No, we need to check all to find the min new_max.
                
                # Decision:
                if best_x_no_increase is not None:
                    x = best_x_no_increase
                elif best_x is not None:
                    x = best_x
                else:
                    return None # No valid candidate
                
                row.append(x)
                used.update(best_diffs)
                current_max_diff = max(current_max_diff, max(best_diffs) if best_diffs else 0)
            
            rows.append(row)
        
        return rows

    # Local search / Hill Climbing improvement
    # Try to reduce the max entry by rebuilding individual rows or swapping elements.
    # Given the strict constraints, simple swaps are hard.
    # Instead, we can try to rebuild a specific row with a tighter bound.
    
    def try_improve(rows, n_rows, k, max_entry, timeout, seed=0):
        """
        Try to improve the current solution by rebuilding rows one by one
        with a potentially smaller max_entry.
        """
        deadline = time.time() + timeout
        rng = random.Random(seed)
        current_max = max(max(r) for r in rows)
        
        if current_max <= max_entry:
            return rows, current_max
        
        # Try to reduce max_entry
        # We can try to rebuild the row that has the largest element.
        
        # Identify the row with the largest element
        max_val = 0
        max_row_idx = 0
        for i, r in enumerate(rows):
            if max(r) > max_val:
                max_val = max(r)
                max_row_idx = i
        
        # Try to rebuild this row with max_entry = current_max - 1
        # We need to remove its differences from 'used'
        
        # Reconstruct used set from all rows
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        
        # Remove differences of the row to be rebuilt
        row_to_rebuild = rows[max_row_idx]
        for i in range(len(row_to_rebuild)):
            for j in range(i + 1, len(row_to_rebuild)):
                used.remove(row_to_rebuild[j] - row_to_rebuild[i])
        
        # Try to rebuild
        # We need to find a new row that fits in max_entry
        # We can use the dynamic greedy for a single row, but we need to ensure it fits
        # and doesn't conflict with 'used'.
        
        # Since dynamic_greedy builds all rows, we need a variant that builds one row
        # given a fixed 'used' set and max_entry.
        
        def build_single_row(k, used, max_entry, deadline):
            row = [0]
            current_max_diff = 0
            # We don't really care about current_max_diff for the row itself,
            # just that it fits in max_entry and uses distinct diffs not in 'used'.
            
            while len(row) < k + 1:
                last = row[-1]
                remaining_slots = k + 1 - len(row)
                min_x = last + 1
                max_x = max_entry - (remaining_slots - 1)
                
                if min_x > max_x:
                    return None
                
                best_x = None
                best_diffs = None
                best_cost = None
                
                for x in range(min_x, max_x + 1):
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # Cost: minimize max(new_diffs), then x
                    # Actually, to pack tight, we want small x.
                    # But we also want to avoid using large diffs if possible?
                    # The global max entry is the constraint.
                    # Let's just pick the smallest valid x.
                    # This is the "greedy" part.
                    # To add diversity, we could randomize.
                    # But for now, smallest x is a good heuristic for local repair.
                    
                    best_x = x
                    best_diffs = new_diffs
                    break # Take the first valid one (smallest x)
                
                if best_x is None:
                    return None
                
                row.append(best_x)
                used.update(best_diffs)
            
            return row
        
        new_row = build_single_row(k, used, current_max - 1, deadline)
        
        if new_row is not None:
            rows[max_row_idx] = new_row
            # Update used set (already updated in build_single_row)
            # Recalculate max
            new_max = max(max(r) for r in rows)
            return rows, new_max
        
        # If failed, restore original row
        used.update(row_to_rebuild[j] - row_to_rebuild[i] for i in range(len(row_to_rebuild)) for j in range(i + 1, len(row_to_rebuild)))
        
        return rows, current_max

    # Main search loop
    best_rows = None
    best_max = float('inf')
    
    # 1. Initial greedy to get a baseline
    # We don't have a max_entry bound for the initial greedy, so we let it run free?
    # No, dynamic_greedy requires max_entry.
    # Let's use the simple greedy first to get an upper bound.
    greedy_rows = greedy_construct(n_rows, k)
    best_rows = greedy_rows
    best_max = max(max(r) for r in best_rows)
    
    # 2. Try to improve using dynamic_greedy with decreasing max_entry
    # Start from best_max and go down to lb
    
    # We can also try random restarts of dynamic_greedy
    # But dynamic_greedy is deterministic if we always pick smallest x.
    # To add randomness, we can shuffle the candidate selection slightly.
    
    # Let's create a randomized version of dynamic_greedy
    def dynamic_greedy_random(n_rows, k, max_entry, seed=0):
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            
            while len(row) < k + 1:
                last = row[-1]
                remaining_slots = k + 1 - len(row)
                min_x = last + 1
                max_x = max_entry - (remaining_slots - 1)
                
                if min_x > max_x:
                    return None
                
                candidates = []
                for x in range(min_x, max_x + 1):
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        candidates.append((x, new_diffs))
                
                if not candidates:
                    return None
                
                # Heuristic: pick from candidates.
                # Prefer smaller x.
                # To add randomness, we can pick from the top few smallest x.
                # Sort by x
                candidates.sort(key=lambda item: item[0])
                
                # Pick randomly from the first min(3, len(candidates))
                # Or just pick the smallest?
                # Let's pick the smallest to be safe, but if we want to escape local minima,
                # we need more randomness.
                # Let's pick randomly from the first 5 candidates.
                pick_range = min(5, len(candidates))
                x, new_diffs = candidates[rng.randint(0, pick_range - 1)]
                
                row.append(x)
                used.update(new_diffs)
            
            rows.append(row)
        
        return rows

    # Search loop
    # We iterate max_entry from best_max down to lb.
    # For each max_entry, we try a few random restarts.
    
    # Time management
    # Allocate more time to higher max_entry values? No, we want the smallest.
    # So we should try lower max_entry values first?
    # But if we start from lb, it's very likely to fail.
    # It's better to start from a known good solution (best_max) and try to reduce.
    # Or start from a value slightly above lb and go up?
    # The prompt says "minimize the largest entry".
    # The current program iterates from lb upwards.
    # That is risky because lb is often too small.
    # Let's iterate from best_max downwards.
    
    # However, the "difference pressure" greedy is very fast.
    # We can afford many restarts.
    
    # Let's try a range of max_entry values.
    # We'll try values from best_max down to lb.
    # But we can't try all if best_max is large.
    # Let's try a few specific values: best_max, best_max-1, ..., lb.
    # We'll prioritize values closer to lb if we have time, but first ensure we find *a* solution.
    # Since we already have a solution (greedy), we can try to improve it.
    
    # Strategy:
    # 1. Try max_entry = best_max - 1, best_max - 2, ... down to lb.
    # 2. For each max_entry, try a few random restarts.
    # 3. If found, update best_max and continue trying to go lower.
    
    # To be efficient, we can binary search? No, the property is not monotonic in a simple way for random restarts.
    # But if a solution exists for M, it exists for M+1.
    # So we can binary search for the minimum M?
    # But finding a solution is NP-hard-ish, so random restarts might fail for small M even if a solution exists.
    # So we should try small M with many restarts, and large M with few restarts.
    
    # Let's just iterate from best_max down to lb.
    # We'll break if we run out of time.
    
    # To speed up, we can skip some values if we have time pressure.
    
    # Let's define a time budget for each max_entry.
    # Total time: 145s
    # We want to spend most time on the smallest M.
    # But we don't know the smallest M.
    # Let's assume the optimal M is close to lb.
    # So we should try lb, lb+1, ... first?
    # If we fail at lb, we try lb+1, etc.
    # This is what the current program does.
    # The problem is that for small M, the search space is very constrained, so random restarts might fail.
    # For larger M, it's easier.
    
    # Let's stick to the current program's strategy: iterate from lb upwards.
    # But use the new dynamic_greedy_random.
    
    for max_entry in range(lb, best_max + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        
        if remaining_time < 0.5:
            break
        
        # Allocate time
        # More time for smaller M (closer to lb)
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(10.0, remaining_time)
        elif distance_from_lb <= 5:
            budget = min(5.0, remaining_time)
        elif distance_from_lb <= 15:
            budget = min(2.0, remaining_time)
        else:
            budget = min(0.5, remaining_time)
        
        if budget < 0.1:
            continue
        
        # Try random restarts
        # Number of restarts depends on budget
        # Each restart is very fast (greedy)
        # Let's say each restart takes ~0.01s
        # So we can do ~100 restarts per second
        num_restarts = int(budget * 50) # Conservative estimate
        if num_restarts < 1:
            num_restarts = 1
        
        found = False
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            
            result = dynamic_greedy_random(n_rows, k, max_entry, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
            # Found a solution for this max_entry.
            # Since we iterate from lb upwards, this is the smallest M found so far.
            # We can break if we want to stop at the first found?
            # The current program breaks.
            # But maybe we can find an even smaller M with more time?
            # No, we are iterating upwards. If we found M, any M' < M was already tried and failed.
            # So this is the best we can do with this strategy.
            # We can break.
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
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
