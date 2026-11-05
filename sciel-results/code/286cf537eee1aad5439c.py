# EVOLVE-BLOCK-START
import sys
import time
import random
from collections import deque

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a randomized CSP approach with bitset-based difference constraint propagation.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: We need 'total_diffs_needed' distinct differences.
    # The differences are integers from 1 to M.
    # So M must be at least total_diffs_needed.
    lb = total_diffs_needed
    
    # Greedy baseline
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
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Bitset-based solver using Python integers as bitsets
    # Bit i is set if difference i is used
    
    def solve_csp_random(n_rows, k, max_entry, timeout, seed=0):
        """
        CSP solver using bitset difference tracking and randomized backtracking with lookahead.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        rng = random.Random(seed)
        
        # Helper to check if a value x can be appended to current_row given used_bits
        # Returns (new_diff_bits, valid)
        def check_candidate(x, current_row, used_bits):
            new_diff_bits = 0
            for a in current_row:
                d = x - a
                if (used_bits >> d) & 1:
                    return 0, False
                new_diff_bits |= (1 << d)
            return new_diff_bits, True

        def has_valid_next(current_row, used_bits, max_entry, remaining_after_x):
            """
            Lookahead: check if there is at least one valid x for the next position.
            current_row is the row after appending x.
            remaining_after_x is how many elements are left to fill.
            """
            if remaining_after_x == 0:
                return True
            
            last = current_row[-1]
            min_x = last + 1
            # We need to fit 'remaining_after_x' elements.
            # The smallest next element is min_x.
            # The largest element in the whole row is max_entry.
            # The next element x_next must satisfy x_next <= max_entry - (remaining_after_x - 1).
            max_x = max_entry - (remaining_after_x - 1)
            
            if min_x > max_x:
                return False
            
            # Check if any x in [min_x, max_x] is valid
            # To speed up, we don't need to check all, just find one.
            # However, checking all is O(K * range). Range is small (~20-30).
            for x in range(min_x, max_x + 1):
                _, valid = check_candidate(x, current_row, used_bits)
                if valid:
                    return True
            return False

        def build_row(current_row, used_bits, deadline):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            candidates = []
            for x in range(min_x, max_x + 1):
                new_diff_bits, valid = check_candidate(x, current_row, used_bits)
                if not valid:
                    continue
                
                # Lookahead: check if this choice allows completing the row
                # remaining_after_x = remaining - 1
                if remaining > 1:
                    # Simulate appending x
                    temp_row = current_row + [x]
                    temp_used = used_bits | new_diff_bits
                    if not has_valid_next(temp_row, temp_used, max_entry, remaining - 1):
                        continue
                
                candidates.append((x, new_diff_bits))
            
            if not candidates:
                return False
            
            # Heuristic: Try smaller x first to pack tight?
            # Or randomize?
            # Let's sort by x, but add some randomness to avoid getting stuck in bad local minima.
            # We will shuffle the candidates.
            rng.shuffle(candidates)
            
            for x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline):
                    return True
                
                current_row.pop()
            
            return False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline):
                rows.append(current_row)
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
            else:
                return None
        
        return rows

    # Try to find better solution than greedy
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Strategy: try max_entry values from lb upward
    # lb is the theoretical minimum.
    # We try a range of values.
    
    search_upper_bound = greedy_max
    
    for max_entry in range(lb, search_upper_bound + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2  # leave 2 sec safety
        
        if remaining_time < 0.5:
            break
        
        # Allocate time: more time for values closer to lb
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(15.0, remaining_time)
        elif distance_from_lb <= 5:
            budget = min(10.0, remaining_time)
        elif distance_from_lb <= 15:
            budget = min(5.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.1:
            continue
            
        # Try multiple restarts for robustness
        found = False
        # Use more restarts for lower M to ensure we don't miss the optimum
        num_restarts = max(2, int(budget / 0.5))
        
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            
            # Distribute time among restarts
            restart_budget = budget / (num_restarts - seed)
            
            result = solve_csp_random(n_rows, k, max_entry, restart_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
            # Found a solution for this max_entry.
            # Since we iterate from lb upwards, this is the smallest M found so far.
            # We break because any larger M is worse.
            break
    
    # Validate
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
