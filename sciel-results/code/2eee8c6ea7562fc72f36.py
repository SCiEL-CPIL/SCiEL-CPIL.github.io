# EVOLVE-BLOCK-START
import sys
import time
import random

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Hybrid approach:
    1. Multi-start randomized greedy to get a valid baseline.
    2. Rebuild-based local search to tighten the maximum entry.
    3. Bitset-accelerated difference tracking for speed.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def time_up():
        return time.time() > deadline

    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed

    # Helper to compute all positive differences in a row as a set
    def get_diffs_set(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # Bitset helpers
    def diffs_to_bits(diffs):
        bits = 0
        for d in diffs:
            bits |= (1 << d)
        return bits

    def bits_to_set(bits):
        s = set()
        d = 1
        while bits:
            if bits & 1:
                s.add(d)
            bits >>= 1
            d += 1
        return s

    # 1. Multi-start Greedy with bitset tracking
    def greedy_construct(seed):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            local_bits = 0
            
            for _ in range(k):
                last = row[-1]
                valid_candidates = []
                
                # Search for valid candidates
                # We don't know the upper bound yet, so we search reasonably high
                # But we can stop if we find enough.
                # For speed, we just find the smallest valid one, or a few.
                # Let's find the smallest valid one first to ensure validity quickly.
                # Then maybe pick a random one from the top 5?
                
                # Optimization: 
                # We can iterate x from last+1 upwards.
                # Check conflicts.
                
                # To allow randomization, we collect a few valid ones.
                width = 5
                
                for x in range(last + 1, 100000): # Large enough upper bound for greedy
                    new_diffs = {x - a for a in row}
                    new_bits = diffs_to_bits(new_diffs)
                    if (new_bits & local_bits) == 0 and (new_bits & used_bits) == 0:
                        valid_candidates.append((x, new_bits))
                        if len(valid_candidates) >= width:
                            break
                
                if not valid_candidates:
                    return None # Failed to construct
                
                # Selection strategy
                # Pick randomly from the top `width` valid candidates.
                # This introduces diversity in the greedy path.
                idx = rng.randint(0, len(valid_candidates) - 1)
                x, new_bits = valid_candidates[idx]
                
                row.append(x)
                local_bits |= new_bits
                
            rows.append(row)
            used_bits |= local_bits
            
        return rows, used_bits

    # Initial best solution
    best_rows = None
    best_max = float('inf')
    current_rows = None
    current_used_bits = 0
    
    # Try a few seeds to find a good initial solution
    for seed in range(20):
        if time_up(): break
        res = greedy_construct(seed)
        if res:
            rows, used_bits = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                current_rows = rows
                current_used_bits = used_bits
    
    if best_rows is None:
        # Fallback: deterministic greedy
        res = greedy_construct(0)
        if res:
            best_rows, current_used_bits = res
            current_rows = best_rows
            best_max = get_max(best_rows)
        else:
            # Should not happen for small k, n_rows
            return []

    if best_max == lb:
        return best_rows

    # 2. Local Search with Rebuilds
    # We maintain a bitset of used differences.
    # We repeatedly pick a row, remove its differences, and try to rebuild it with a smaller max.
    # If we succeed, we accept. If not, we try to rebuild with the same max (reshuffle) to escape local minima.
    
    rng = random.Random(42)
    current_max = best_max

    def rebuild_row(k, forbidden_bits, rng, max_val_limit, bias=0.0):
        """
        Builds a row of size k+1 starting at 0, strictly increasing,
        avoiding forbidden_bits, with max entry <= max_val_limit.
        bias: probability to pick a random valid candidate from the first 10, else smallest.
        """
        row = [0]
        local_bits = 0
        
        for _ in range(k):
            last = row[-1]
            valid_candidates = []
            
            # Search for valid candidates
            # We must ensure x <= max_val_limit
            # Also, we need to leave room for remaining elements?
            # remaining_elements = k - len(row)
            # The next element must be at most max_val_limit - (remaining_elements - 1)
            
            remaining = k - len(row)
            max_x_allowed = max_val_limit - (remaining - 1)
            
            if last + 1 > max_x_allowed:
                return None
            
            for x in range(last + 1, max_x_allowed + 1):
                new_diffs = {x - a for a in row}
                new_bits = diffs_to_bits(new_diffs)
                if (new_bits & local_bits) == 0 and (new_bits & forbidden_bits) == 0:
                    valid_candidates.append((x, new_bits))
                    if len(valid_candidates) >= 10:
                        break
            
            if not valid_candidates:
                return None
            
            # Selection strategy
            if bias > 0 and rng.random() < bias:
                idx = rng.randint(0, len(valid_candidates) - 1)
                x, new_bits = valid_candidates[idx]
            else:
                x, new_bits = valid_candidates[0]
            
            row.append(x)
            local_bits |= new_bits
            
        return row, local_bits

    no_improve_count = 0
    max_no_improve = 100
    
    while not time_up():
        # Pick a row to rebuild
        i = rng.randint(0, n_rows - 1)
        
        old_row = current_rows[i]
        old_diffs = get_diffs_set(old_row)
        old_bits = diffs_to_bits(old_diffs)
        
        # Remove from used_bits
        # Since we're using a bitset, we need to clear the bits for this row
        # But we must ensure no other row uses those bits.
        # The problem statement says all differences are distinct across all rows.
        # So if we remove row i, the bits in old_bits are only used by row i.
        # So we can safely AND with ~old_bits.
        current_used_bits &= ~old_bits
        
        new_row = None
        new_row_bits = 0
        
        # Attempt 1: Improve (strictly smaller max for this row)
        # We want the global max to decrease.
        # If this row is not the one holding the max, we can't lower the global max by lowering this row's max below current_max.
        # But we can try to lower it to see if it helps future steps?
        # Actually, the standard approach is:
        # If max(old_row) == current_max, try to rebuild with max_val_limit = current_max - 1.
        # If max(old_row) < current_max, we can't improve global max directly, but we can reshuffle.
        
        if max(old_row) == current_max:
            target_max = current_max - 1
            if target_max > k: # Must be at least k
                res = rebuild_row(k, current_used_bits, rng, target_max, bias=0.5)
                if res:
                    new_row, new_row_bits = res
        
        # Attempt 2: Reshuffle (maintain max) to escape local minima
        if new_row is None:
            # Try to rebuild with the same max, but different structure
            # This helps if the current structure is "tight" but we need a different configuration
            # to allow another row to be improved later.
            res = rebuild_row(k, current_used_bits, rng, current_max, bias=0.8)
            if res:
                new_row, new_row_bits = res
        
        if new_row is None:
            # Failed to rebuild
            current_used_bits |= old_bits
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a different strategy or break
                # For now, just continue
                no_improve_count = 0
            continue
        
        # Accept the new row
        current_rows[i] = new_row
        current_used_bits |= new_row_bits
        
        new_max = get_max(current_rows)
        
        if new_max < current_max:
            current_max = new_max
            best_rows = [r[:] for r in current_rows]
            best_max = new_max
            no_improve_count = 0
            if best_max == lb:
                break
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # If we've been stuck for a long time, try to restart with a different seed or strategy
                # Or just continue
                no_improve_count = 0

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
        # Fallback to greedy
        res = greedy_construct(0)
        if res:
            best_rows = res[0]
        else:
            best_rows = [[0] * (k + 1)] * n_rows
    
    return best_rows

# EVOLVE-BLOCK-END
