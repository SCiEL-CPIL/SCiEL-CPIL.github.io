"""Improved CSP solver with forward checking, bitset ops, and local search polish."""
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
    
    Uses a CSP approach with bitset-based difference constraint propagation,
    forward checking, dynamic variable ordering, and a local search polish pass.
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
    time_limit = 118.0
    
    def compute_diff_bits(row):
        """Compute bitset of all positive differences in a row."""
        bits = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                bits |= (1 << (row[j] - row[i]))
        return bits
    
    def solve_csp(n_rows, k, max_entry, timeout, seed=0):
        """
        CSP solver using bitset difference tracking, forward checking,
        and dynamic ordering.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        rng = random.Random(seed)
        used_bits = 0  # bitset of used differences
        rows = []
        
        # Precompute small diff mask for fast counting
        small_mask = (1 << 16) - 1  # bits 0-15
        
        def build_row(current_row, used_bits, deadline, rng):
            """
            Recursively build a row using backtracking with bitset checks
            and forward checking.
            """
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Forward checking: ensure we can still fill remaining positions
            if max_entry - last < remaining:
                return False
            
            # Generate candidates
            candidates = []
            for x in range(last + 1, max_entry + 1):
                # Prune: need room for remaining-1 more elements after x
                if max_entry - x < remaining - 1:
                    break
                
                # Compute new differences as bitset
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
                
                # Count small differences used
                small_count = bin(new_diff_bits & small_mask).count('1')
                
                # Minimum distance to existing elements
                min_dist = min(x - a for a in current_row)
                
                # Score: prefer fewer small diffs, larger min distance
                score = small_count * 20 + (15 - min(min_dist, 15)) * 3
                
                candidates.append((score, x, new_diff_bits))
            
            if not candidates:
                return False
            
            # Sort by score with some randomization for diversity
            candidates.sort(key=lambda c: c[0] + rng.random() * 5)
            
            for score, x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline, rng):
                    return True
                
                current_row.pop()
            
            return False
        
        # Build rows one by one
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline, rng):
                rows.append(current_row)
                # Update used_bits with all diffs from this row
                row_bits = compute_diff_bits(current_row)
                used_bits |= row_bits
            else:
                return None
        
        return rows
    
    def local_search_polish(rows, max_entry, timeout, seed=0):
        """
        Try to reduce the largest entry by local search:
        - Try replacing elements in each row to reduce the max
        - Use simulated annealing-style acceptance
        """
        deadline = time.time() + timeout
        rng = random.Random(seed)
        best = [list(r) for r in rows]
        best_max = max(max(r) for r in best)
        
        # Compute used differences for current solution
        def get_used_bits(sol):
            bits = 0
            for row in sol:
                bits |= compute_diff_bits(row)
            return bits
        
        def is_valid(sol):
            bits = get_used_bits(sol)
            # Check no duplicate differences
            # Actually, if we build correctly, bits won't have conflicts
            # But we need to check that all diffs are distinct
            used = set()
            for row in sol:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in used:
                            return False
                        used.add(d)
            return True
        
        current = [list(r) for r in best]
        current_max = best_max
        
        # Temperature for simulated annealing
        T = 2.0
        cooling = 0.995
        
        while time.time() < deadline:
            # Pick a random row and a random position to modify
            row_idx = rng.randrange(n_rows)
            pos = rng.randrange(1, k + 1)  # don't modify position 0 (which is 0)
            
            # Try to decrease the element at pos
            old_val = current[row_idx][pos]
            if old_val <= 1:
                continue
            
            # Try new values from old_val - 1 down to pos (must be > previous)
            prev_val = current[row_idx][pos - 1] if pos > 0 else 0
            
            new_val = old_val - 1
            found = False
            while new_val > prev_val and new_val >= pos:
                # Check if new_val creates valid differences
                new_row = list(current[row_idx])
                new_row[pos] = new_val
                
                # Check increasing
                if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
                    new_val -= 1
                    continue
                
                # Check differences
                new_row_bits = compute_diff_bits(new_row)
                
                # Check against other rows
                conflict = False
                for other_idx in range(n_rows):
                    if other_idx == row_idx:
                        continue
                    other_bits = compute_diff_bits(current[other_idx])
                    if new_row_bits & other_bits:
                        conflict = True
                        break
                
                if not conflict:
                    # Check within the row itself (differences should be distinct)
                    diffs = []
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            if d in diffs:
                                conflict = True
                                break
                            diffs.append(d)
                    if conflict:
                        new_val -= 1
                        continue
                
                # Accept the move
                current[row_idx] = new_row
                new_max = max(max(r) for r in current)
                
                if new_max < current_max:
                    current_max = new_max
                    if current_max < best_max:
                        best = [list(r) for r in current]
                        best_max = current_max
                    found = True
                    break
                
                new_val -= 1
            
            # Cooling
            T *= cooling
        
        return best, best_max
    
    # Try to find better solution than greedy
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Strategy: try max_entry values from lb upward with time budgets
    max_search_range = min(greedy_max, lb + 30)
    
    # Try multiple seeds for diversity
    seeds = [0, 1, 42, 123, 777, 2024]
    
    for max_entry in range(lb, max_search_range + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 3
        
        if remaining_time < 0.5:
            break
        
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(10.0, remaining_time / max(1, len(seeds)))
        elif distance_from_lb <= 3:
            budget = min(5.0, remaining_time / max(1, len(seeds)))
        elif distance_from_lb <= 10:
            budget = min(2.5, remaining_time / max(1, len(seeds)))
        else:
            budget = min(1.0, remaining_time / max(1, len(seeds)))
        
        if budget < 0.2:
            break
        
        found = False
        for seed in seeds:
            if time.time() > start_time + time_limit - 3:
                break
            
            result = solve_csp(n_rows, k, max_entry, budget, seed)
            if result is not None:
                # Try to polish
                elapsed2 = time.time() - start_time
                polish_time = min(remaining_time - (time.time() - start_time - elapsed2) - 2, 5.0)
                if polish_time > 0.5:
                    polished, polished_max = local_search_polish(result, max_entry, polish_time, seed)
                    if polished_max < max_entry:
                        result = polished
                        max_entry = polished_max
                
                best_rows = result
                best_max = max(max(r) for r in result)
                found = True
                break
        
        if found:
            break
    
    # If we didn't find anything better, try a mid-point with more time
    if best_max == greedy_max and greedy_max > lb:
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 3
        if remaining_time > 3.0:
            mid = (lb + greedy_max) // 2
            if mid > lb:
                budget = min(remaining_time - 2, 6.0)
                result = solve_csp(n_rows, k, mid, budget, 0)
                if result is not None:
                    best_rows = result
                    best_max = max(max(r) for r in best_rows)
    
    # Final polish with remaining time
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed - 2
    if remaining_time > 1.0:
        polished, polished_max = local_search_polish(best_rows, best_max, remaining_time, 999)
        if polished_max < best_max:
            best_rows = polished
            best_max = polished_max
    
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
