"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    
    Uses a CSP approach with bitset-based difference constraint propagation
    and dynamic variable ordering to find tight solutions.
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
    time_limit = 115.0
    
    # Bitset-based solver using Python integers as bitsets
    # Bit i is set if difference i is used
    # We need differences from 1 to max_entry, so we need max_entry bits
    
    def solve_csp(n_rows, k, max_entry, timeout):
        """
        CSP solver using bitset difference tracking and dynamic ordering.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        
        # Dynamic ordering: process rows in order of tightest remaining capacity
        # For now, process in order but use good heuristics within each row
        
        def build_row(current_row, used_bits, deadline):
            """
            Recursively build a row using backtracking with bitset checks.
            Uses MRV-like heuristic: try values that leave most flexibility.
            """
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Generate candidate values in order of "flexibility"
            # Prefer values that don't create many small differences
            # (small differences are more likely to conflict)
            
            candidates = []
            for x in range(last + 1, max_entry + 1):
                # Prune: need room for remaining-1 more elements
                if max_entry - x < remaining - 1:
                    break
                
                # Compute new differences as bitset
                new_diff_bits = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if d <= 0:
                        continue
                    # Check if d is already used
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if conflict:
                    continue
                
                # Score: prefer values that use fewer "popular" differences
                # Small differences are more constrained
                score = sum(1 for i in range(1, min(d, 20) + 1) if (new_diff_bits >> i) & 1)
                candidates.append((score, x, new_diff_bits))
            
            # Sort by score (fewer small diffs = more flexible)
            candidates.sort()
            
            # Add some randomization for diversity across restarts
            if len(candidates) > 1:
                # Shuffle a subset of candidates to explore different orderings
                shuffle_start = len(candidates) // 2
                candidates[shuffle_start:] = random.sample(candidates[shuffle_start:], len(candidates) - shuffle_start)
            
            for score, x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline):
                    return True
                
                current_row.pop()
            
            return False
        
        # Build rows one by one
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline):
                rows.append(current_row)
                # Update used_bits with all diffs from this row
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
    
    # Strategy: randomized restarts with short per-restart budgets
    # This allows more diverse search paths within the same total budget
    
    max_search_range = min(greedy_max, lb + 30)
    deadline = start_time + time_limit - 3  # safety margin
    
    # Phase 1: Try each max_entry from lb upward with multiple short restarts
    for max_entry in range(lb, max_search_range + 1):
        if time.time() > deadline:
            break
        
        # For each max_entry, try up to N restarts with short budgets
        # More restarts for values closer to lb (more likely to be feasible)
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            max_restarts = 8
            per_restart = 2.0
        elif distance_from_lb <= 3:
            max_restarts = 5
            per_restart = 1.5
        elif distance_from_lb <= 10:
            max_restarts = 3
            per_restart = 1.0
        else:
            max_restarts = 2
            per_restart = 0.8
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Use a random seed for diversity
            random.seed(hash((n_rows, k, max_entry, restart)) % (2**32))
            
            result = solve_csp(n_rows, k, max_entry, per_restart)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break  # Found best possible for this max_entry
        
        if found:
            break  # Found a valid solution at this max_entry
    
    # Phase 2: If we still have time and haven't found anything better than greedy,
    # try some intermediate values with longer individual runs
    if best_max == greedy_max and greedy_max > lb:
        remaining = deadline - time.time()
        if remaining > 5.0:
            # Try a few strategic intermediate points
            candidates = []
            for frac in [0.3, 0.5, 0.7]:
                mid = lb + int((greedy_max - lb) * frac)
                if mid > lb:
                    candidates.append(mid)
            
            for mid in candidates:
                if time.time() > deadline:
                    break
                budget = min(remaining - 3, 8.0)
                if budget < 1.0:
                    break
                result = solve_csp(n_rows, k, mid, budget)
                if result is not None:
                    best_rows = result
                    best_max = mid
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
