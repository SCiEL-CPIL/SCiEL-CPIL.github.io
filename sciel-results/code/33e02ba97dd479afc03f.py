# EVOLVE-BLOCK-START
import sys
import time
import random

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
    
    def solve_csp_optimized(n_rows, k, max_entry, timeout, seed=0):
        """
        Optimized CSP solver using bitsets and forward checking.
        """
        deadline = time.time() + timeout
        used_bits = 0
        rows = []
        rng = random.Random(seed)
        
        # We build rows one by one.
        # To improve search, we can randomize the order of rows?
        # The rows are indistinguishable in the final output, so order doesn't matter for validity.
        # However, the constraints are global.
        # Let's stick to fixed order but randomize value choices.
        
        def build_row(current_row, used_bits):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            min_x = last + 1
            # We need to fit 'remaining' more elements, so the last one can be at most max_entry.
            # The current x must be <= max_entry - (remaining - 1)
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            candidates = []
            # Pre-compute bitsets for differences?
            # For each x, we need to check if (x - a) is in used_bits for all a in current_row.
            
            for x in range(min_x, max_x + 1):
                new_diff_bits = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if not conflict:
                    candidates.append((x, new_diff_bits))
            
            if not candidates:
                return False
            
            # Heuristic: Shuffle candidates
            rng.shuffle(candidates)
            
            for x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used):
                    return True
                
                current_row.pop()
            
            return False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits):
                rows.append(current_row)
                # Update used_bits with the full row's differences
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
            else:
                return None
        
        return rows

    best_rows = greedy_rows
    best_max = greedy_max
    
    search_upper_bound = greedy_max
    
    # We try to find a solution for M starting from lb.
    # We allocate more time to lower M values as they are more likely to be the optimal answer.
    
    for max_entry in range(lb, search_upper_bound + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        
        if remaining_time < 0.5:
            break
        
        # Time allocation strategy:
        # If we are close to lb, give more time.
        # If we are far, give less time, hoping we find a solution quickly or move on.
        distance = max_entry - lb
        if distance == 0:
            budget = min(15.0, remaining_time)
        elif distance <= 5:
            budget = min(10.0, remaining_time)
        elif distance <= 20:
            budget = min(5.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
            
        if budget < 0.1:
            continue
            
        # Try multiple restarts
        # Each restart gets a slice of the budget
        num_restarts = max(1, int(budget / 2.0))
        restart_budget = budget / num_restarts
        
        found = False
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
                
            result = solve_csp_optimized(n_rows, k, max_entry, restart_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
            # Since we iterate M upwards, the first M for which we find a solution is the best.
            # We break because any larger M is worse.
            break
    
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
