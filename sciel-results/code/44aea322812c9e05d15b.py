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
    
    Uses a CSP approach with bitset-based difference tracking and MRV heuristic.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: M must be at least total_diffs_needed
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
    
    def solve_csp(n_rows, k, max_entry, timeout, seed=0):
        """
        CSP solver using bitset difference tracking and MRV heuristic.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        rng = random.Random(seed)
        
        # We'll build rows one by one, but use a smarter approach:
        # For each row, we choose elements sequentially with MRV-like heuristic.
        
        def can_place(x, current_row, used_bits):
            """Check if x can be added to current_row without conflicts."""
            for a in current_row:
                d = x - a
                if d <= 0:
                    return False
                if (used_bits >> d) & 1:
                    return False
            return True
        
        def get_diff_bits(x, current_row):
            """Get the bitset of new differences when adding x to current_row."""
            bits = 0
            for a in current_row:
                d = x - a
                bits |= (1 << d)
            return bits
        
        def build_row_with_heuristic(current_row, used_bits, deadline, rng):
            """
            Build a row using a heuristic that prefers values that leave
            more options for future elements.
            """
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
            
            # Generate candidates and score them
            candidates = []
            for x in range(min_x, max_x + 1):
                if not can_place(x, current_row, used_bits):
                    continue
                
                # Heuristic: prefer x that uses "larger" differences
                # because smaller differences are more "scarce" in the global pool
                # Also prefer x that doesn't create too many small differences
                diff_sum = 0
                min_diff = max_x
                for a in current_row:
                    d = x - a
                    diff_sum += d
                    if d < min_diff:
                        min_diff = d
                
                # Score: prefer larger min_diff (less conflict with small diffs)
                # and moderate diff_sum
                score = min_diff * 10 - abs(diff_sum - (max_x - last) * remaining // 2) * 0.1
                
                candidates.append((score, x))
            
            if not candidates:
                return False
            
            # Sort by score (descending), then add some randomness
            candidates.sort(key=lambda t: -t[0])
            
            # Take top candidates with some randomness
            top_n = min(len(candidates), max(5, len(candidates) // 4))
            top_candidates = [x for _, x in candidates[:top_n]]
            
            # Shuffle the top candidates for diversity
            rng.shuffle(top_candidates)
            
            for x in top_candidates:
                if time.time() > deadline:
                    return False
                
                new_diff_bits = get_diff_bits(x, current_row)
                new_used = used_bits | new_diff_bits
                
                current_row.append(x)
                if build_row_with_heuristic(current_row, new_used, deadline, rng):
                    return True
                current_row.pop()
            
            # If top candidates failed, try all candidates in random order
            all_candidates = [x for _, x in candidates]
            rng.shuffle(all_candidates)
            
            for x in all_candidates:
                if time.time() > deadline:
                    return False
                
                new_diff_bits = get_diff_bits(x, current_row)
                new_used = used_bits | new_diff_bits
                
                current_row.append(x)
                if build_row_with_heuristic(current_row, new_used, deadline, rng):
                    return True
                current_row.pop()
            
            return False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row_with_heuristic(current_row, used_bits, deadline, rng):
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
    # But be smart about which values to try
    
    search_upper_bound = greedy_max
    
    # First, try a few values near the lower bound with generous time
    # Then try values in between if we have time
    
    # Determine a good search range
    # For (4,7): lb=112, greedy~134, best known=134
    # We want to find the minimum M
    
    # Try values from lb to lb+20 first with more time each
    # Then if not found, try up to greedy_max
    
    # Phase 1: Try close to lb
    phase1_end = min(lb + 15, search_upper_bound)
    
    for max_entry in range(lb, phase1_end + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        
        if remaining_time < 0.5:
            break
        
        # Give more time to values closer to lb
        distance_from_lb = max_entry - lb
        if distance_from_lb <= 5:
            budget = min(8.0, remaining_time / (phase1_end - lb + 1))
        else:
            budget = min(4.0, remaining_time / (phase1_end - lb + 1))
        
        if budget < 0.1:
            continue
        
        # Try multiple restarts
        num_restarts = max(1, int(budget / 0.8))
        
        found = False
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            restart_budget = min(0.8, budget / (num_restarts - seed))
            
            result = solve_csp(n_rows, k, max_entry, restart_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
            # Found a solution for this max_entry
            # Since we're going up from lb, this is the best for this range
            # But maybe we can find a smaller one? No, we started at lb.
            # However, we might have skipped some values. Let's break.
            break
    
    # Phase 2: If not found in phase 1, try a wider range
    if best_max >= search_upper_bound:
        # Try some values in the middle range
        mid_start = lb + 20
        mid_end = min(search_upper_bound - 5, lb + 50)
        
        if mid_start < mid_end:
            # Sample some values in this range
            step = max(1, (mid_end - mid_start) // 10)
            for max_entry in range(mid_start, mid_end + 1, step):
                elapsed = time.time() - start_time
                remaining_time = time_limit - elapsed - 2
                
                if remaining_time < 0.5:
                    break
                
                budget = min(3.0, remaining_time / 10)
                
                if budget < 0.1:
                    continue
                
                num_restarts = max(1, int(budget / 0.5))
                
                found = False
                for seed in range(num_restarts):
                    if time.time() > start_time + time_limit - 2:
                        break
                    restart_budget = min(0.5, budget / (num_restarts - seed))
                    
                    result = solve_csp(n_rows, k, max_entry, restart_budget, seed=seed)
                    if result is not None:
                        if max_entry < best_max:
                            best_rows = result
                            best_max = max_entry
                        found = True
                        break
                
                if found:
                    # Continue searching for even smaller M
                    # But we're going up, so we should go back down?
                    # Actually, we should search more carefully near this value
                    pass
    
    # Phase 3: If we have time, try to improve the current best
    if best_max < search_upper_bound:
        # Try to find a solution with max_entry = best_max - 1, best_max - 2, etc.
        for max_entry in range(best_max - 1, max(lb - 1, best_max - 10), -1):
            if max_entry < lb:
                break
            
            elapsed = time.time() - start_time
            remaining_time = time_limit - elapsed - 2
            
            if remaining_time < 0.5:
                break
            
            budget = min(5.0, remaining_time / 10)
            
            if budget < 0.1:
                continue
            
            num_restarts = max(1, int(budget / 0.5))
            
            for seed in range(num_restarts):
                if time.time() > start_time + time_limit - 2:
                    break
                restart_budget = min(0.5, budget / (num_restarts - seed))
                
                result = solve_csp(n_rows, k, max_entry, restart_budget, seed=seed)
                if result is not None:
                    best_rows = result
                    best_max = max_entry
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
