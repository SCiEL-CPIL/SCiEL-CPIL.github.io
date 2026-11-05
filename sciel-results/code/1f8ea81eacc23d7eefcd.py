import sys
import time
import random
from collections import deque

def construct(n_rows, k):
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound on scope M
    # We need total_diffs_needed distinct integers in [1, M]
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
    time_limit = 115.0 # Safety margin for 120s limit
    
    # Bitset solver
    def solve_csp(n_rows, k, max_entry, timeout, seed=0):
        deadline = time.time() + timeout
        used_bits = 0
        rows = []
        rng = random.Random(seed)
        
        # Precompute: diffs needed for a row of length L (L elements)
        # If we have placed 'placed' elements, we need 'k+1-placed' more.
        # The new differences added by placing 'rem' more elements depend on their values.
        # But we can bound the *count* of new differences.
        # Each new element adds 'placed' new differences.
        # So if we place 'rem' more elements, we add sum_{i=1..rem} (placed + i - 1) differences?
        # No.
        # Let current row length be L.
        # Next element adds L differences.
        # Next adds L+1.
        # ...
        # Total new diffs = sum_{j=L}^{k} j = (k+1)(k+2)/2 - L(L+1)/2.
        # This is the exact number of new differences added by completing the row.
        
        def count_new_diffs_for_completion(current_len):
            # current_len is number of elements already in the row
            # remaining elements: k + 1 - current_len
            # Let r = k + 1 - current_len
            # The new elements will be at indices current_len, ..., k
            # Element at index i adds i new differences (distances to 0..i-1)
            # So sum from i=current_len to k of i
            # = (k(k+1)/2) - (current_len-1)(current_len)/2
            return (k * (k + 1)) // 2 - ((current_len - 1) * current_len) // 2
        
        def build_row(current_row, used_bits, deadline, row_idx):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Pruning 1: Can we fit 'remaining' elements in [last+1, max_entry]?
            # Min sum of gaps: 1 + 2 + ... + remaining = remaining(remaining+1)/2
            if last + remaining * (remaining + 1) // 2 > max_entry:
                return False
            
            # Pruning 2: Global difference count
            # Diffs used so far: popcount(used_bits)
            # Diffs needed to finish current row: count_new_diffs_for_completion(len(current_row))
            # Diffs needed for other rows: (n_rows - row_idx - 1) * diffs_per_row
            # Total needed: used + needed_current + needed_other
            # Available: max_entry
            # If available < total_needed, prune.
            
            # Note: popcount is expensive in Python if done naively.
            # But used_bits is an int. bin(used_bits).count('1') is fast enough for small M.
            # Or we can track count in a variable. Let's pass it.
            
            # For now, let's skip the expensive popcount in the inner loop if possible.
            # We can track `used_count` as a parameter.
            
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            candidates = []
            for x in range(min_x, max_x + 1):
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
                
                candidates.append((x, new_diff_bits))
            
            if not candidates:
                return False
            
            # Heuristic: Order candidates
            # 1. Prefer smaller x (compact)
            # 2. Prefer x that uses "small" differences?
            # Let's just shuffle to ensure exploration.
            rng.shuffle(candidates)
            
            for x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline, row_idx):
                    return True
                
                current_row.pop()
            
            return False
        
        # Add used_count to build_row signature?
        # Let's refactor build_row to accept used_count.
        
        def build_row_v2(current_row, used_bits, used_count, deadline, row_idx):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Pruning 1: Space
            if last + remaining * (remaining + 1) // 2 > max_entry:
                return False
            
            # Pruning 2: Difference Count
            needed_current = count_new_diffs_for_completion(len(current_row))
            needed_other = (n_rows - row_idx - 1) * diffs_per_row
            total_needed = used_count + needed_current + needed_other
            
            if total_needed > max_entry:
                return False
            
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            candidates = []
            for x in range(min_x, max_x + 1):
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
                
                candidates.append((x, new_diff_bits))
            
            if not candidates:
                return False
            
            rng.shuffle(candidates)
            
            for x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                new_count = used_count + len(current_row) # New element adds len(current_row) diffs
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row_v2(current_row, new_used, new_count, deadline, row_idx):
                    return True
                
                current_row.pop()
            
            return False

        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            # used_count is 0 initially
            if build_row_v2(current_row, used_bits, 0, deadline, row_idx):
                rows.append(current_row)
                # Update used_bits and used_count for next row
                # We can just re-calculate or update incrementally
                # Since build_row_v2 doesn't return the final used_bits, we need to capture it.
                # Let's modify build_row_v2 to return the final used_bits and used_count?
                # Or just re-calculate from the row.
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
                # Recalculate used_count
                used_count = bin(used_bits).count('1')
            else:
                return None
        
        return rows

    best_rows = greedy_rows
    best_max = greedy_max
    
    # Search
    for max_entry in range(lb, greedy_max + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        
        if remaining_time < 0.5:
            break
        
        # Allocate time
        # If we are close to lb, give more time.
        distance = max_entry - lb
        if distance == 0:
            budget = min(15.0, remaining_time)
        elif distance <= 5:
            budget = min(10.0, remaining_time)
        elif distance <= 10:
            budget = min(5.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.1:
            continue
        
        # Try multiple seeds
        num_restarts = max(1, int(budget / 0.5))
        found = False
        
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            restart_budget = budget / num_restarts
            result = solve_csp(n_rows, k, max_entry, restart_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
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
