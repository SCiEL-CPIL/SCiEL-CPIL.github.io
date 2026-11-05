"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
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
    
    def solve_csp(n_rows, k, max_entry, timeout):
        """
        CSP solver using bitset difference tracking and forward checking.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        
        # Precompute bit masks for efficiency
        # For a given x and current_row, compute the bitset of new diffs
        
        def get_new_diff_bits(x, current_row):
            """Compute bitset of new differences if we add x to current_row."""
            bits = 0
            for a in current_row:
                d = x - a
                if d > 0:
                    bits |= (1 << d)
            return bits
        
        def check_conflict(x, current_row, used_bits):
            """Check if adding x creates any conflict."""
            for a in current_row:
                d = x - a
                if d > 0:
                    if (used_bits >> d) & 1:
                        return True
            return False
        
        def forward_check(x, current_row, used_bits, max_entry, remaining_positions):
            """
            Check if after placing x, we can still fill remaining_positions
            more elements without conflicts. Returns True if feasible.
            """
            if remaining_positions <= 0:
                return True
            
            # After placing x, the next element must be > x
            # We need remaining_positions elements in range (x, max_entry]
            if max_entry - x < remaining_positions:
                return False
            
            # Check if there are enough available differences
            # This is a weak check; we rely on backtracking for stronger pruning
            return True
        
        def build_row(current_row, used_bits, deadline):
            """
            Recursively build a row using backtracking with bitset checks.
            Uses heuristic: prefer values that use large differences (preserve small ones).
            """
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Generate candidate values
            # Heuristic: try larger values first to use "expensive" differences
            # This preserves small differences for other rows
            candidates = []
            for x in range(max_entry, last, -1):
                # Prune: need room for remaining-1 more elements
                if max_entry - x < remaining - 1:
                    continue
                
                # Check for conflicts
                conflict = False
                new_diff_bits = 0
                for a in current_row:
                    d = x - a
                    if d > 0:
                        if (used_bits >> d) & 1:
                            conflict = True
                            break
                        new_diff_bits |= (1 << d)
                
                if conflict:
                    continue
                
                # Score: prefer values that use larger differences
                # Count how many small differences (<=20) this would use
                small_diff_count = 0
                for i in range(1, 21):
                    if (new_diff_bits >> i) & 1:
                        small_diff_count += 1
                
                # Prefer fewer small diffs (more flexible)
                candidates.append((small_diff_count, -x, new_diff_bits))
            
            # Sort by score: fewer small diffs first, then larger x first
            candidates.sort()
            
            for small_diff_count, neg_x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                x = -neg_x
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
    
    # Strategy: try max_entry values from lb upward with time budgets
    max_search_range = min(greedy_max, lb + 50)
    
    for max_entry in range(lb, max_search_range + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5  # leave 5 sec safety
        
        if remaining_time < 1.0:
            break
        
        # Allocate time: more time for values closer to lb
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(20.0, remaining_time)
        elif distance_from_lb <= 5:
            budget = min(10.0, remaining_time)
        elif distance_from_lb <= 20:
            budget = min(5.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.5:
            break
        
        result = solve_csp(n_rows, k, max_entry, budget)
        if result is not None:
            best_rows = result
            best_max = max_entry
            break  # Found best possible for this max_entry
    
    # If we didn't find anything better, try a mid-point with more time
    if best_max == greedy_max and greedy_max > lb:
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        if remaining_time > 5.0:
            mid = (lb + greedy_max) // 2
            if mid > lb:
                budget = min(remaining_time, 15.0)
                result = solve_csp(n_rows, k, mid, budget)
                if result is not None:
                    best_rows = result
                    best_max = mid
    
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
