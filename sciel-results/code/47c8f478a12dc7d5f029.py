import time
import random
import sys
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Theoretical lower bound for scope
    min_scope = total_diffs
    
    # Get a baseline greedy solution first
    def greedy_baseline():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                diffs = {x - a for a in row}
                if not diffs & used:
                    row.append(x)
                    used.update(diffs)
            rows.append(row)
        return rows

    best_rows = greedy_baseline()
    best_scope = max(max(r) for r in best_rows)
    
    # If k is small, we can try to find better solutions
    # For k=7, n_rows=4, total_diffs = 4*7*8/2 = 112
    # Greedy usually gives around 130-140
    
    # Strategy: Backtracking search for difference sets
    # Each row i has k*(k+1)/2 differences
    # We need to assign differences to rows such that:
    # 1. All differences are distinct
    # 2. The differences in each row can form a valid increasing sequence starting at 0
    
    # A row is defined by its set of differences. If we have differences d1, d2, ..., dm
    # where m = k*(k+1)/2, we need to find a0=0, a1, ..., ak such that
    # all ai - aj (i>j) are exactly the set {d1, ..., dm}
    
    # This is equivalent to: the set of differences must be "closed" under some structure
    # Actually, a simpler view: if we have the set of all pairwise differences,
    # the largest difference is the scope of that row (ak - a0 = ak)
    # And the differences must be consistent: if d = ai - aj and e = aj - ak, then d+e = ai - ak
    
    # Let's use a different approach: search for the rows directly using backtracking
    # but with a focus on minimizing the maximum entry.
    
    # Actually, let's try a simulated annealing or hill climbing approach on the rows
    # Start with greedy, then try to modify rows to reduce the global max
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def get_all_diffs(rows):
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    diffs.add(r[j] - r[i])
        return diffs
    
    def is_valid(rows):
        # Check all differences are distinct
        seen = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    # Hill climbing: try to improve the current solution by modifying individual rows
    # or swapping differences
    
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    
    # Precompute current differences
    current_diffs = get_all_diffs(current_rows)
    
    # For hill climbing, we can try to "rebuild" a row given the used differences
    # This is similar to the greedy approach but constrained by the existing used diffs
    
    def rebuild_row(used_diffs, max_val):
        """Rebuild a row given the set of used differences (from other rows) and max allowed value"""
        # This is essentially the randomized greedy but with a fixed set of forbidden diffs
        # We want to find a row of length k+1 starting at 0 with max <= max_val
        # such that all its differences are not in used_diffs
        
        # Use backtracking with randomization
        row = [0]
        
        def backtrack(pos, current_row, forbidden):
            if len(current_row) == k + 1:
                return current_row[:]
            
            last = current_row[-1]
            # Try values from last+1 to max_val
            # To be efficient, we can try in a random order or with some bias
            
            # Collect valid candidates
            candidates = []
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in current_row}
                if not (new_diffs & forbidden):
                    candidates.append(x)
            
            if not candidates:
                return None
            
            # Randomize order to escape local minima
            random.shuffle(candidates)
            
            # Try each candidate
            for x in candidates:
                new_diffs = {x - a for a in current_row}
                current_row.append(x)
                result = backtrack(pos + 1, current_row, forbidden | new_diffs)
                if result is not None:
                    return result
                current_row.pop()
            
            return None
        
        return backtrack(0, row, used_diffs)
    
    # Main search loop: try to improve by rebuilding rows
    # We'll do multiple passes, each time trying to reduce the scope
    
    improved = True
    while time.time() < deadline and improved:
        improved = False
        # Try to reduce the scope by 1
        target_scope = current_scope - 1
        
        # Try to find a valid configuration with max <= target_scope
        # We can do this by trying to rebuild each row with the constraint
        
        # Strategy: Try to rebuild rows one by one, maintaining validity
        # This is tricky because rebuilding one row changes the forbidden diffs for others
        
        # Alternative: Use a randomized restart approach
        # Try to construct the whole set from scratch with the limit
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Try to construct with the target scope
            random.seed(int(time.time() * 1000) + attempt)
            
            # Use a modified greedy that tries to stay within the limit
            used = set()
            new_rows = []
            success = True
            
            for _ in range(n_rows):
                row = [0]
                current_max = 0
                while len(row) < k + 1:
                    # Find valid next marks
                    candidates = []
                    for x in range(current_max + 1, target_scope + 1):
                        diffs = {x - a for a in row}
                        if not (diffs & used):
                            candidates.append(x)
                    
                    if not candidates:
                        success = False
                        break
                    
                    # Choose randomly from candidates, with bias towards smaller
                    # to try to minimize the max
                    # But we need to ensure we can complete the row
                    # Simple heuristic: pick randomly from the first few
                    window = min(5, len(candidates))
                    next_x = random.choice(candidates[:window])
                    
                    row.append(next_x)
                    current_max = next_x
                    used.update({next_x - a for a in row[:-1]})
                
                if not success:
                    break
                new_rows.append(row)
            
            if success:
                # Verify
                if is_valid(new_rows):
                    new_scope = max(max(r) for r in new_rows)
                    if new_scope < current_scope:
                        current_rows = new_rows
                        current_scope = new_scope
                        best_rows = new_rows
                        best_scope = new_scope
                        improved = True
                        # Don't break, try to improve further
                        break  # Break out of attempt loop to try lower scope
    
    return best_rows
