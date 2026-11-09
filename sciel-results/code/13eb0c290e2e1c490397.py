import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Greedy initial solution
    used_diffs = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        greedy_rows.append(row)
    
    best_max = max(max(r) for r in greedy_rows)
    best_rows = greedy_rows
    
    # Bitset-based representation for used differences
    # Max difference is at most best_max, so we need bits 1..best_max
    # Use a Python integer as a bitset where bit d is set if diff d is used
    
    def used_to_bitset(used_set):
        bits = 0
        for d in used_set:
            bits |= (1 << d)
        return bits
    
    def bitset_to_used(bits):
        used = set()
        d = 1
        while (1 << d) & bits:
            if (1 << d) & bits:
                used.add(d)
            d += 1
        return used
    
    def diffs_bitset(row):
        """Compute bitset of all positive differences in a row."""
        bits = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                bits |= (1 << (row[j] - row[i]))
        return bits
    
    def check_row_valid(row, used_bits):
        """Check if a row's differences are all distinct and not in used_bits."""
        row_bits = diffs_bitset(row)
        return (row_bits & used_bits) == 0
    
    # Precompute: for a partial row, compute the bitset of differences
    # We'll use a recursive approach with bitsets for speed
    
    def try_construct_bitset(target_max, n_rows, k, random_state, time_limit):
        """Try to construct n_rows rows with max entry <= target_max using bitset backtracking."""
        used_bits = 0
        rows = []
        
        def backtrack_row(idx, row, used_bits, time_limit):
            """Build a row of k+1 elements starting from [0], current index is idx."""
            if time.time() > time_limit:
                return None
            
            if idx == k:
                return row[:]
            
            last = row[-1]
            
            # Collect valid candidates
            valid_candidates = []
            for x in range(last + 1, target_max + 1):
                # Check differences x - a_i for all a_i in row
                # These must not overlap with used_bits
                conflict = False
                for a in row:
                    d = x - a
                    if (1 << d) & used_bits:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                return None
            
            # Try candidates in randomized order from a window
            W = 7
            candidates_to_try = valid_candidates[:W] if len(valid_candidates) > W else valid_candidates[:]
            random_state.shuffle(candidates_to_try)
            
            for x in candidates_to_try:
                new_row = row + [x]
                # Compute new bits for this step
                new_bits = 0
                for a in row:
                    new_bits |= (1 << (x - a))
                
                res = backtrack_row(idx + 1, new_row, used_bits | new_bits, time_limit)
                if res is not None:
                    return res
            
            return None
        
        for _ in range(n_rows):
            if time.time() > time_limit:
                return None
            row = [0]
            res = backtrack_row(0, row, used_bits, time_limit)
            if res is None:
                return None
            # Update used_bits with all differences in the complete row
            used_bits |= diffs_bitset(res)
            rows.append(res)
        
        return rows
    
    # Main search: try to improve best_max
    while time.time() < deadline:
        target_max = best_max - 1
        if target_max < k + 1:
            break  # Can't go lower than k+1 (minimum possible for k+1 increasing elements starting at 0)
        
        found_improvement = False
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Per-restart time limit
            restart_deadline = min(deadline, time.time() + 0.8)
            random_state = random.Random()
            
            res = try_construct_bitset(target_max, n_rows, k, random_state, restart_deadline)
            if res is not None:
                current_max = max(max(r) for r in res)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = res
                    found_improvement = True
                    break
        
        if not found_improvement:
            # Try with a slightly higher target to see if we get a better arrangement
            # that might enable further reduction later? No, that doesn't help.
            # Instead, try more aggressive randomization or different window sizes.
            # For now, just keep trying the same target with more restarts in next loop.
            # Actually, let's try a different strategy: use a larger window or different random seed strategy.
            
            # Try with window size 10 and more restarts
            for restart in range(100):
                if time.time() > deadline:
                    break
                restart_deadline = min(deadline, time.time() + 0.5)
                random_state = random.Random(restart * 12345 + 67890)
                
                # Modified: use a different candidate selection strategy
                # Try picking from the middle of the candidate list too
                res = try_construct_bitset(target_max, n_rows, k, random_state, restart_deadline)
                if res is not None:
                    current_max = max(max(r) for r in res)
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = res
                        found_improvement = True
                        break
            
            if not found_improvement:
                # We're stuck at this target_max. Try to see if we can find a solution
                # with the current best_max that has a different structure, which might
                # help in finding a solution with best_max - 1 later.
                # For now, just continue the loop; the next iteration will try the same target.
                # To avoid infinite loop, add a small delay or break if we've tried many times.
                pass
    
    return best_rows
