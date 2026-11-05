"""
Strategy: Randomized Greedy Construction with Bitmask Acceleration and Restarts.

The previous approach used backtracking and Simulated Annealing. This approach
uses a highly optimized randomized greedy algorithm.
1. We represent the set of used differences as a Python integer bitmask.
   Since the lower bound on max entry is ~112 and best known is 134, we can
   assume max entry < 200. The bitmask fits easily in a Python int.
2. For each row, we construct it element by element.
3. At each step, instead of picking the absolute smallest valid number (which 
   leads to deterministic dead-ends), we pick from a random subset of the 
   smallest valid candidates. This introduces stochasticity to escape local traps.
4. We perform many independent restarts with different random seeds.
5. We keep the solution with the smallest maximum entry.
"""

import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Total differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Simple lower bound: We need to fit 'total_diffs' distinct positive integers 
    # into the range [1, max_entry]. So max_entry >= total_diffs.
    # Actually, the differences are $a_j - a_i$. The largest difference is $a_k - 0 = a_k$.
    # All differences must be in [1, a_k].
    # So we need $a_k \ge total\_diffs$.
    lower_bound = total_diffs
    
    best_rows = None
    best_max = float('inf')
    
    # Bitmask size: We expect max entry to be around 112-140. 
    # Let's allow up to 256 to be safe. 2^256 is huge but Python ints handle it.
    # However, checking bit operations is faster than set lookups.
    
    def construct_greedy_randomized(seed, max_limit):
        """
        Attempts to construct a valid set of rows with max entry < max_limit.
        Uses randomized greedy selection.
        Returns the rows if successful, None otherwise.
        """
        rng = random.Random(seed)
        used_diffs_mask = 0
        rows = []
        
        for r in range(n_rows):
            row = [0]
            current_max_in_row = 0
            
            for c in range(1, k + 1):
                # We need to pick the next number x > row[-1]
                # such that all new differences x - row[i] are not in used_diffs_mask
                
                # Estimate a reasonable range for x.
                # x must be > row[-1].
                # Also x - 0 = x must be < max_limit.
                # So x is in [row[-1] + 1, max_limit - 1]
                
                min_x = row[-1] + 1
                max_x = max_limit - 1
                
                if min_x > max_x:
                    return None
                
                # Collect valid candidates
                # To speed up, we don't scan ALL up to max_x.
                # We scan a window of candidates. If we find enough, we pick one.
                # If we don't find enough, we extend the window.
                
                candidates = []
                # Heuristic: The number of valid candidates is usually dense enough 
                # near the lower bound. Let's check up to 100 candidates or until 
                # we have 5 valid ones.
                
                check_limit = min(max_x, min_x + 200)
                
                for x in range(min_x, check_limit + 1):
                    # Check if x - 0 is used
                    # x is the difference with 0.
                    if (used_diffs_mask >> x) & 1:
                        continue
                    
                    # Check differences with other elements in row
                    valid = True
                    new_diffs_bits = 0
                    for a in row:
                        d = x - a
                        if (used_diffs_mask >> d) & 1:
                            valid = False
                            break
                        new_diffs_bits |= (1 << d)
                    
                    if valid:
                        candidates.append((x, new_diffs_bits))
                        if len(candidates) >= 5:
                            break
                
                if not candidates:
                    # If we didn't find any in the window, maybe we need to go higher?
                    # For now, if we hit the check_limit and found none, it's likely 
                    # a dead end for this greedy path.
                    return None
                
                # Randomly select from the top candidates
                # Weight towards smaller x to keep max entry low
                # But allow some exploration
                # Simple uniform choice among the first few found
                x, new_diffs_bits = candidates[rng.randint(0, len(candidates) - 1)]
                
                row.append(x)
                used_diffs_mask |= new_diffs_bits
                current_max_in_row = x
            
            rows.append(row)
            
        # Verify the whole thing just in case (though construction ensures validity 
        # if we don't have bugs in the mask logic)
        # The max entry is the max of the last elements (since rows are increasing)
        final_max = max(row[-1] for row in rows)
        
        return rows, final_max

    # Main loop: Try many seeds
    # We want to find a solution with max < best_max.
    # Initially best_max is inf.
    # We can set an initial max_limit based on lower bound + slack.
    # Let's start with max_limit = lower_bound + 50
    
    current_limit = lower_bound + 50
    
    # If we find a solution, we can tighten the limit.
    
    attempts = 0
    max_attempts = 5000  # Cap attempts to avoid infinite loop if time is not the issue
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Try to construct with current limit
        # We reduce the limit slightly as we find better solutions
        # Start with a generous limit, then tighten
        
        # Strategy: 
        # 1. Try to beat the current best.
        # 2. If no best, try to find any valid solution.
        
        if best_rows is None:
            limit_to_try = current_limit
        else:
            limit_to_try = best_max - 1
            
        if limit_to_try < lower_bound:
            # We've hit the theoretical lower bound, can't do better
            break
            
        rows, final_max = construct_greedy_randomized(attempts, limit_to_try)
        
        if rows is not None:
            if best_rows is None or final_max < best_max:
                best_rows = rows
                best_max = final_max
                
                # If we found a solution, we can try to tighten the limit for next iterations
                # The next search should aim for best_max - 1
                # We don't need to change current_limit variable explicitly if we 
                # just use best_max - 1 in the loop condition.
                
                # Optional: If we get very close to lower bound, stop?
                # The task says "best known is 134", lower bound 112.
                # If we hit 112, we are done.
                if best_max <= lower_bound:
                    break
    
    # Fallback: If no valid rows found (should be rare), use a simple deterministic greedy
    # that is guaranteed to produce a valid (though possibly large) set.
    if best_rows is None:
        # Deterministic greedy: always pick smallest valid x
        used_diffs_mask = 0
        rows = []
        for r in range(n_rows):
            row = [0]
            for c in range(1, k + 1):
                x = row[-1] + 1
                while True:
                    valid = True
                    new_diffs_bits = 0
                    for a in row:
                        d = x - a
                        if (used_diffs_mask >> d) & 1:
                            valid = False
                            break
                        new_diffs_bits |= (1 << d)
                    if valid:
                        break
                    x += 1
                row.append(x)
                used_diffs_mask |= new_diffs_bits
            rows.append(row)
        best_rows = rows
        best_max = max(row[-1] for row in rows)
        
    return best_rows

