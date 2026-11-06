import time
import random
import numpy as np

def _compute_block_sums(arr):
    """Compute all consecutive block sums using prefix sums. Returns a set."""
    if len(arr) == 0:
        return set()
    prefix = np.concatenate(([0], np.cumsum(arr)))
    sums = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n+1):
            sums.add(int(prefix[j] - prefix[i]))
    return sums

def _can_append_fast(values, x, existing_sums):
    """Check if appending x creates any duplicate block sum."""
    if not values:
        return True
    # New block sums are x + suffix_sums of values
    # Suffix sums: 0, values[-1], values[-1]+values[-2], ...
    # We need to check if any new sum is in existing_sums
    # And if new sums are distinct (they are, since values > 0)
    
    # Optimization: Just compute the new sums and check intersection
    # Suffix sums can be computed quickly
    suffix_sum = 0
    # We need to check x + 0, x + v[-1], x + v[-1] + v[-2], ...
    # Note: The existing sums include all subarrays of values.
    # The new sums are subarrays ending at x.
    
    # We can just generate the new sums and check membership
    # Number of new sums is len(values) + 1
    # This is fast.
    
    # To be extra safe and fast, we can precompute suffix sums of values
    # But values is a list, so we can just iterate.
    
    total = x
    if total in existing_sums:
        return False
    for a in reversed(values):
        total += a
        if total in existing_sums:
            return False
    return True

def _get_new_sums(values, x):
    """Get the new block sums when appending x to values."""
    total = x
    new_sums = {total}
    for a in reversed(values):
        total += a
        new_sums.add(total)
    return new_sums

def _greedy_min(start_val=1, max_val=256):
    """Greedy: pick smallest valid value at each step."""
    values = []
    sums = set()
    for x in range(start_val, max_val + 1):
        if values and x <= values[-1]:
            continue
        if _can_append_fast(values, x, sums):
            values.append(x)
            sums |= _get_new_sums(values[:-1], x)
    return values

def _local_search_extend(values, sums, deadline, max_iterations=10000):
    """
    Local search to extend the sequence.
    1. Try to append a new value.
    2. If stuck, try to replace an existing value with a larger one to 'make room'.
    3. Repeat.
    """
    best = list(values)
    best_sums = set(sums)
    best_len = len(best)
    
    # Helper to validate a sequence
    def _is_valid(seq):
        if not seq:
            return True
        # Check strictly increasing
        for i in range(len(seq)-1):
            if seq[i] >= seq[i+1]:
                return False
        # Check distinct sums
        s = _compute_block_sums(np.array(seq, dtype=np.int64))
        return len(s) == len(seq) * (len(seq) + 1) // 2

    # Phase 1: Simple extension
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(best[-1] + 1 if best else 1, 257):
            if _can_append_fast(best, x, best_sums):
                best.append(x)
                best_sums |= _get_new_sums(best[:-1], x)
                best_len = len(best)
                changed = True
                break

    # Phase 2: Local Search (Swaps)
    # We try to replace elements with larger values to shift the sum set,
    # potentially allowing for more elements to be appended.
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        improved = False
        
        # Try replacing each element with a larger one
        for i in range(len(best)):
            if time.time() > deadline:
                break
            
            old_val = best[i]
            # Determine range for new_val
            min_val = old_val + 1
            max_val = 256
            if i > 0:
                min_val = max(min_val, best[i-1] + 1)
            if i < len(best) - 1:
                max_val = min(max_val, best[i+1] - 1)
            
            if min_val > max_val:
                continue
                
            # Try a few random candidates or scan
            # Scanning is safer for finding the "best" swap
            # But we just want *any* valid swap that might help.
            # Actually, we want to swap to a value that is valid *and* allows extension.
            # This is hard to check directly.
            # Let's just find *any* valid swap, then try to extend.
            
            # To be efficient, let's try a few random values in range
            # Or just the smallest valid one?
            # Smallest valid one keeps the sequence "tight", which might be bad for extension.
            # Largest valid one pushes sums up, which might be good.
            
            # Let's try the largest valid value first, as it creates the most "room" at the low end?
            # No, pushing sums up creates room at the low end for new small sums?
            # But new elements must be larger than the last element.
            
            # Let's just try to find a valid replacement.
            # We'll iterate from old_val + 1 upwards.
            found_replacement = False
            for new_val in range(min_val, max_val + 1):
                trial = list(best)
                trial[i] = new_val
                
                # Quick check: is it strictly increasing? (Already handled by range)
                # Check distinct sums
                trial_sums = _compute_block_sums(np.array(trial, dtype=np.int64))
                if len(trial_sums) == len(trial) * (len(trial) + 1) // 2:
                    # Valid replacement
                    # Does this allow extension?
                    # We don't know yet. Let's accept it and try to extend in the next loop.
                    best = trial
                    best_sums = trial_sums
                    improved = True
                    found_replacement = True
                    break
            
            if improved:
                break
        
        if not improved:
            # If no replacement found, try to insert a new element?
            # Or break if we can't improve length.
            # Since we already tried simple extension in Phase 1, 
            # and replacements don't change length, we need to be careful.
            # If we can't replace, we are stuck at this length.
            # However, a replacement might enable a longer sequence later.
            # So we should continue searching for replacements even if length doesn't change immediately?
            # No, that's infinite loop.
            # We should only accept a replacement if it leads to a longer sequence eventually.
            # This requires lookahead.
            
            # Simpler: If no replacement found, break.
            break
            
    return best

def construct():
    deadline = time.time() + 85.0  # Safety margin
    
    # Phase 1: Greedy
    best = _greedy_min()
    best_len = len(best)
    
    # Phase 2: Local Search from Greedy
    if time.time() < deadline:
        sums = _compute_block_sums(np.array(best, dtype=np.int64))
        result = _local_search_extend(best, sums, deadline)
        if len(result) > best_len:
            best = result
            best_len = len(best)
    
    # Phase 3: Try different starting points or random restarts
    # The greedy might be suboptimal due to early choices.
    # We can try to "restart" the greedy from a certain point with a different choice.
    
    # Let's implement a simple backtracking search to see if we can beat the greedy result.
    # We use the greedy result as a lower bound.
    
    def _backtrack_search(current_seq, current_sums, min_val, deadline, best_len_ref):
        """
        Depth-first search to extend current_seq.
        Returns best sequence found.
        """
        best_found = list(current_seq)
        best_len_local = len(best_found)
        
        # Pruning
        if len(current_seq) + (256 - (current_seq[-1] if current_seq else 0)) <= best_len_ref:
            return best_found
        
        # Try to extend
        for x in range(min_val, 257):
            if time.time() > deadline:
                break
            if current_seq and x <= current_seq[-1]:
                continue
            
            if _can_append_fast(current_seq, x, current_sums):
                new_sums = current_sums | _get_new_sums(current_seq, x)
                new_seq = current_seq + [x]
                
                if len(new_seq) > best_len_local:
                    best_found = new_seq
                    best_len_local = len(new_seq)
                    # Update global best if needed? 
                    # We pass best_len_ref as a constant lower bound to prune.
                    # We should update the nonlocal best if we find something better.
                    # But for simplicity, we return the best found in this branch.
                
                # Recurse
                deeper_result = _backtrack_search(new_seq, new_sums, x + 1, deadline, best_len_ref)
                if len(deeper_result) > len(best_found):
                    best_found = deeper_result
                    best_len_local = len(deeper_result)
                
                # If we found a sequence longer than the initial best, we can prune other branches
                # that can't beat it.
        
        return best_found

    if time.time() < deadline:
        # Start backtracking from the greedy result
        # But we need to be careful: the greedy result is a "leaf".
        # We want to search for *better* sequences.
        # We can try to backtrack from the *start* with pruning based on the greedy length.
        
        # However, full backtracking from start is expensive.
        # Let's try to extend the greedy result further.
        # If greedy is stuck, backtracking from the end might help by changing previous choices.
        
        # Let's try a limited backtracking from the end of the greedy sequence.
        # We will try to replace the last k elements.
        
        # For now, let's just run a short backtracking from the start with aggressive pruning.
        # Pruning: If current_length + (256 - last_val) < best_len, stop.
        
        # To make it faster, we only explore if we have time.
        # We'll use the greedy result as the initial best.
        
        # Let's try to run backtracking from a few different "prefixes" of the greedy sequence.
        # Or just from the start.
        
        # Given time constraints, let's just do a quick DFS from the start with pruning.
        # If it finds something better, great.
        
        # We'll use a global variable to track the best found in this phase.
        global_best = best
        global_best_len = best_len
        
        def _dfs(vals, sums_set, min_v, depth):
            nonlocal global_best, global_best_len
            if time.time() > deadline:
                return
            
            # Prune
            max_possible_len = len(vals) + (256 - (vals[-1] if vals else 0))
            if max_possible_len <= global_best_len:
                return
            
            if len(vals) > global_best_len:
                global_best = list(vals)
                global_best_len = len(global_best)
                
            # Try to extend
            # Optimization: Only try a few candidates per level to avoid explosion
            # We'll try all candidates, but break if time is up.
            for x in range(min_v, 257):
                if time.time() > deadline:
                    return
                if vals and x <= vals[-1]:
                    continue
                
                if _can_append_fast(vals, x, sums_set):
                    new_sums = sums_set | _get_new_sums(vals, x)
                    vals.append(x)
                    _dfs(vals, new_sums, x + 1, depth + 1)
                    vals.pop()
                    
                    # If we found a new best, we might prune more aggressively
                    # But the loop condition handles it.

        # Start DFS from empty
        # To speed up, we can seed it with the first few elements of the greedy sequence.
        # Let's just start from empty, but the pruning will be strong if best_len is high.
        
        # If greedy found length 9, best_len is 9.
        # Pruning: len(vals) + (256 - last) <= 9.
        # This will prune a lot.
        
        _dfs([], set(), 1, 0)
        
        if len(global_best) > best_len:
            best = global_best
            best_len = len(best)
            
    # Final validation
    if best:
        arr = np.array(best, dtype=np.int64)
        all_sums = _compute_block_sums(arr)
        expected = len(best) * (len(best) + 1) // 2
        if len(all_sums) != expected:
            # Fallback
            best = _greedy_min()
            
    return best
