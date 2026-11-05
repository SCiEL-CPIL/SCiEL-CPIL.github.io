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
    
    Uses a CSP approach with bitset-based difference constraint propagation.
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
    
    def solve_csp(n_rows, k, max_entry, timeout, seed=0):
        """
        CSP solver using bitset difference tracking and backtracking.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        
        # We build rows one by one.
        # To improve performance, we can try to build the "hardest" row first?
        # Or just build in fixed order.
        # For now, we build in fixed order.
        
        def build_row(current_row, used_bits, deadline):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Generate candidate values
            # We need to pick x such that x > last and x <= max_entry
            # Also, we must be able to fit 'remaining' more elements after x.
            # So x <= max_entry - remaining + 1
            
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            # Heuristic: Try smaller x first to keep rows compact?
            # Or try x that minimizes the "waste" of large differences?
            # For difference triangles, compact rows are usually better for global packing.
            
            # We iterate from min_x to max_x.
            # We can add some randomness if we want, but deterministic is often faster for CSPs with good heuristics.
            # Let's try a mix: if we fail, we might restart with different heuristics.
            # For now, simple increasing order.
            
            for x in range(min_x, max_x + 1):
                if time.time() > deadline:
                    return False
                
                # Check conflicts
                # new_diff_bits = OR of (1 << (x - a)) for a in current_row
                # If (used_bits & new_diff_bits) != 0, conflict.
                
                new_diff_bits = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if conflict:
                    continue
                
                # Place x
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline):
                    return True
                
                current_row.pop()
            
            return False
        
        # Try to build all rows
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline):
                rows.append(current_row)
                # Update used_bits with all differences in this row
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
    
    # We can also try to jump to a value slightly above lb if lb is very small
    # But for this problem lb is tight (112).
    
    # We will iterate M from lb to search_upper_bound
    # We allocate time based on how close M is to lb.
    
    # To handle timeouts gracefully, we break out if time is low.
    
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
        elif distance_from_lb <= 20:
            budget = min(5.0, remaining_time)
        else:
            budget = min(1.0, remaining_time)
        
        if budget < 0.1:
            continue
            
        # Try to solve with this max_entry
        # We might do a few restarts with different random seeds if we modify the solver to be random.
        # For now, the solver is deterministic. 
        # If it fails, it likely fails for this M.
        # We can try to add randomness to the candidate selection if we want.
        
        # Let's try one deterministic run.
        # If we want to be more robust, we can shuffle the order of rows or candidates.
        
        # To allow for some exploration, let's make the candidate selection slightly random 
        # if we have time, but for now deterministic is a good baseline.
        
        # Actually, for CSP, deterministic with good heuristics is often best.
        # The heuristic "smallest x first" is good for keeping rows small.
        
        result = solve_csp(n_rows, k, max_entry, budget, seed=0)
        if result is not None:
            best_rows = result
            best_max = max_entry
            # Since we are iterating M from lb upwards, this is the best possible for this M
            # And since M is increasing, this is the global minimum found so far.
            # We can break if we want the absolute best, but maybe we can find a smaller M?
            # No, we started from lb. If we found one at M, it's the best.
            # Wait, if we found a solution at M, it's valid.
            # Is it possible a smaller M exists? No, we checked all smaller M.
            # So we can break.
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
