"""
Strategy: CSP Backtracking with Forward Checking and MRV.
Instead of local search on integers, we build rows one by one.
For each row, we select elements such that their differences don't conflict with existing global differences.
We use MRV (Minimum Remaining Values) to choose the next element to fix, ensuring we pick the most constrained spot first.
"""

import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Total differences per row: k*(k+1)/2
    # Total differences overall: n_rows * k*(k+1)/2
    # We need to pack these into [1, M]
    
    # Initial greedy construction to have a valid starting point
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
        return rows, max(max(r) for r in rows)

    baseline_rows, baseline_max = pure_greedy()
    best_rows = [r[:] for r in baseline_rows]
    best_max = baseline_max
    
    if best_max == n_rows * k * (k + 1) // 2:
        return best_rows

    # CSP State:
    # rows: list of lists, partially filled
    # used_diffs: set of all differences used so far
    # We will build rows one by one. 
    # To allow optimization, we can also try to optimize individual rows given the global used set.
    # However, building rows sequentially with backtracking is the core CSP approach.
    
    # To improve on greedy, we can use randomized backtracking with restarts.
    # We aim to find a solution with a smaller max value.
    
    # Heuristic: 
    # When building a row, we have k+1 slots. Slot 0 is 0.
    # We choose the next slot to fill using MRV? 
    # Actually, since the row must be increasing, the order is fixed (0, x1, x2...).
    # But we can choose the *value* for the next slot.
    # MRV applies to which *value* is most constrained? 
    # Or we can view it as: we have a set of available numbers [1..M]. 
    # For each row, we pick k numbers.
    
    # Let's try a different angle: 
    # 1. Start with greedy.
    # 2. Use a "repair" phase: pick a row, try to reconstruct it to be "tighter" (smaller max) 
    #    while respecting the differences used by OTHER rows.
    #    This is effectively a single-row CSP with a fixed set of forbidden differences.
    
    def try_reconstruct_row(row_index, forbidden_diffs, current_max_limit, rng):
        """
        Try to build a valid row for row_index such that:
        1. It is increasing, starts at 0.
        2. All internal differences are distinct.
        3. No internal difference is in forbidden_diffs.
        4. The last element (max) is <= current_max_limit.
        Returns the new row or None.
        """
        # We need to pick k numbers from [1, current_max_limit]
        # such that their pairwise differences (and differences with 0) are valid.
        
        # Backtracking search for this single row
        # State: current_row (list), used_in_row (set of diffs within this row)
        
        def backtrack(row, idx, used_in_row):
            if idx == k + 1:
                return row[:]
            
            # Determine the range for the next value
            # Must be > row[-1]
            # Must be <= current_max_limit
            
            # Heuristic: Try smaller values first to keep the row tight?
            # Or randomize to escape local minima?
            # Let's try a mix: bias towards smaller values, but allow random jumps.
            
            start_val = row[-1] + 1
            end_val = current_max_limit
            
            if start_val > end_val:
                return None
            
            # Generate candidates
            # To speed up, we can check validity before recursing.
            # Validity: for all x in row, (new_val - x) not in forbidden_diffs AND (new_val - x) not in used_in_row
            
            # Shuffling candidates for randomization
            # If the range is large, we can't shuffle all. 
            # We can sample.
            
            # Let's iterate from start_val to end_val.
            # To add randomness, we can start from a random offset or shuffle the first few.
            # For simplicity and speed, let's just iterate, but we can restart the whole row construction 
            # with different random seeds if we fail.
            
            # Optimization: Pre-filter? 
            # For each candidate v, check if {v - x for x in row} intersects (forbidden_diffs U used_in_row).
            
            for v in range(start_val, end_val + 1):
                # Check conflicts
                conflict = False
                new_diffs = []
                for x in row:
                    d = v - x
                    if d in forbidden_diffs or d in used_in_row:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if not conflict:
                    # Recursive step
                    result = backtrack(row + [v], idx + 1, used_in_row | set(new_diffs))
                    if result:
                        return result
            
            return None
        
        # Initial call
        # We can add a small randomization to the start if we fail, but backtrack is deterministic here.
        # To make it stochastic, we can randomize the order of checking v? 
        # That's expensive. Instead, we can just call this function multiple times with different 
        # "perturbations" or just rely on the fact that if greedy failed to find a tighter row, 
        # maybe a different global configuration allows it.
        
        # Actually, a pure deterministic backtrack might be slow if the space is large.
        # Let's add a timeout per row reconstruction.
        
        row_start = [0]
        used_in_row = set()
        
        # We'll use a recursive function with a time check
        def backtrack_with_time(row, idx, used_in_row, time_limit):
            if time.time() > time_limit:
                return None
            if idx == k + 1:
                return row[:]
            
            start_val = row[-1] + 1
            end_val = current_max_limit
            
            if start_val > end_val:
                return None
            
            # Try values
            # To introduce randomness, we can shuffle the candidates if the range is small enough
            candidates = list(range(start_val, end_val + 1))
            if len(candidates) < 100:
                rng.shuffle(candidates)
            else:
                # If range is large, we can't shuffle all. 
                # We can try a random subset first? 
                # Or just iterate. The "tightness" bias (smaller values first) is good for minimizing max.
                # So we iterate normally (small to large).
                pass
                
            for v in candidates:
                conflict = False
                new_diffs = []
                for x in row:
                    d = v - x
                    if d in forbidden_diffs or d in used_in_row:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if not conflict:
                    result = backtrack_with_time(row + [v], idx + 1, used_in_row | set(new_diffs), time_limit)
                    if result:
                        return result
            return None
        
        # Give this row 2 seconds
        return backtrack_with_time([0], 1, set(), time.time() + 2.0)

    # Main Loop: Iterative Improvement
    # 1. Compute global used diffs
    # 2. Pick a random row
    # 3. Try to reconstruct it with a slightly smaller max limit
    # 4. If successful, update global state
    
    current_rows = [r[:] for r in baseline_rows]
    
    while time.time() < deadline:
        # Calculate current used diffs
        current_used = set()
        for r in current_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    current_used.add(r[j] - r[i])
        
        current_max = max(r[-1] for r in current_rows)
        
        if current_max < best_max:
            best_max = current_max
            best_rows = [r[:] for r in current_rows]
            if best_max == n_rows * k * (k + 1) // 2:
                break
            
        # Pick a row to improve
        # Bias towards rows with larger max? Or random?
        # Let's pick the row with the current global max, or a random one.
        # Picking the max row is more likely to reduce the global max.
        max_val = max(r[-1] for r in current_rows)
        candidates = [i for i, r in enumerate(current_rows) if r[-1] == max_val]
        if not candidates:
            candidates = list(range(n_rows))
        
        r_idx = random.choice(candidates)
        row = current_rows[r_idx]
        row_max = row[-1]
        
        # We want to try to fit this row into a smaller space
        # Target max: row_max - 1, or maybe lower?
        # Let's try to fit into [0, row_max - 1] first.
        # If that fails, try [0, row_max - 2], etc.
        # Or just try [0, row_max - 1] and if it fails, move on.
        
        # Forbidden diffs: all diffs used by OTHER rows
        forbidden = set(current_used)
        # Remove diffs from the current row
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                forbidden.discard(row[j] - row[i])
        
        # Try to reconstruct
        # We attempt to fit the row into a max value of row_max - 1
        # If that works, we improve.
        # If not, maybe try row_max - 2? 
        # To save time, we only try one step down per iteration.
        
        new_max_limit = row_max - 1
        
        # If new_max_limit is too small (less than k), it's impossible
        if new_max_limit < k:
            continue
            
        new_row = try_reconstruct_row(r_idx, forbidden, new_max_limit, random.Random(int(time.time()*1000) % 10000))
        
        if new_row:
            # Verify validity (safety check)
            valid = True
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d in forbidden or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
            if valid:
                current_rows[r_idx] = new_row
                # Loop back to recalculate global state

    return best_rows

# EVOLVE-BLOCK-END
