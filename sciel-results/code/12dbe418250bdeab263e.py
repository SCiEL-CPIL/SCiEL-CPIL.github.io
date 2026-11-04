"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random

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
    
    # Improved Greedy: Try to keep numbers small by prioritizing small diffs
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
    time_limit = 118.0  # Leave 2s for safety
    
    def solve_csp(n_rows, k, max_entry, timeout):
        """
        CSP solver using bitset difference tracking.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        
        # Precompute masks for fast checking? No, just use bit operations.
        
        def build_row(current_row, used_bits, deadline):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Heuristic: Try values that minimize the "cost" of used differences.
            # Small differences are more "expensive" because they are rare.
            # So we prefer candidates that use larger differences if possible,
            # or rather, we want to leave small differences for other rows.
            
            candidates = []
            # Start from last+1 up to max_entry
            # Pruning: We need 'remaining' more numbers, so the next number must be
            # at most max_entry - (remaining - 1)
            max_x = max_entry - (remaining - 1)
            
            for x in range(last + 1, max_x + 1):
                # Check conflicts and compute new diff bits
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
                
                # Score: Sum of the differences? Or count of small diffs?
                # Let's try to minimize the sum of the new differences to keep the row tight?
                # Or maximize? 
                # Actually, for the *global* problem, we want to distribute differences.
                # A good heuristic is to prefer x that creates differences that are
                # "less likely" to be needed elsewhere. Small differences are critical.
                # So we should AVOID small differences if possible.
                # Score = sum of (1/d) for d in new_diffs? 
                # Let's just use the sum of the differences. Lower sum = tighter row.
                # But tighter rows might use up small diffs.
                # Let's try sorting by the largest difference in the set (minimize it).
                max_diff_in_new = max(d for d in [x - a for a in current_row])
                candidates.append((max_diff_in_new, x, new_diff_bits))
            
            # Sort by smallest max_diff first (tightest fit)
            candidates.sort()
            
            for _, x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline):
                    return True
                
                current_row.pop()
            
            return False
        
        # Build rows one by one
        # Order of rows doesn't matter, but we can try to build the "hardest" row first?
        # For now, standard order.
        for _ in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline):
                rows.append(current_row)
                # Update used_bits
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
            else:
                return None
        
        return rows
    
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Search for better solutions
    # Try values from lb to greedy_max
    # Allocate time wisely.
    
    # If lb is very close to greedy_max, try lb first with high budget.
    # If gap is large, try a few points.
    
    search_values = []
    if greedy_max - lb <= 10:
        search_values = range(lb, greedy_max + 1)
    else:
        # Try lb, lb+1, lb+2, then some midpoints
        search_values = list(range(lb, min(lb + 3, greedy_max + 1)))
        # Add midpoints
        if greedy_max > lb + 10:
            mid1 = lb + (greedy_max - lb) // 4
            mid2 = lb + (greedy_max - lb) // 2
            mid3 = lb + 3 * (greedy_max - lb) // 4
            search_values.extend([mid1, mid2, mid3])
            search_values = sorted(set(search_values))
    
    for max_entry in search_values:
        if max_entry >= best_max:
            continue
            
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        
        if remaining_time < 0.5:
            break
        
        # Budget allocation: higher for lower max_entry
        # If max_entry is close to lb, give it more time.
        if max_entry <= lb + 2:
            budget = min(20.0, remaining_time)
        elif max_entry <= lb + 10:
            budget = min(10.0, remaining_time)
        else:
            budget = min(5.0, remaining_time)
        
        if budget < 0.1:
            break
        
        result = solve_csp(n_rows, k, max_entry, budget)
        if result is not None:
            best_rows = result
            best_max = max_entry
            # If we found a solution at lb, we can't do better
            if best_max == lb:
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
