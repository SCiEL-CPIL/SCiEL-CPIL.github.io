import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    # Leave 10s buffer for safety
    deadline = start_time + 45.0
    
    def is_time_up():
        return time.time() > deadline

    # 1. Greedy Baseline
    def greedy():
        values = []
        sums = set()
        suffix_sums = [] # Sums of consecutive blocks ending at the last element
        
        for x in range(1, 257):
            # New sums if we append x
            # The new blocks are: [x], [prev_last, x], [prev_prev, prev_last, x], etc.
            # These correspond to x + suffix_sums
            # But suffix_sums currently holds sums ending at values[-1].
            # So new_sums = [x] + [x + s for s in suffix_sums]
            
            new_sums = [x]
            for s in suffix_sums:
                new_sums.append(x + s)
            
            # Check collision
            if sums.intersection(new_sums):
                continue
            
            # Accept x
            values.append(x)
            sums.update(new_sums)
            # Update suffix_sums
            # New suffix sums are exactly new_sums
            suffix_sums = new_sums
            
        return values, sums, suffix_sums

    base_seq, base_sums, base_suffix = greedy()
    best_seq = base_seq
    base_len = len(base_seq)
    
    # 2. Backtracking Search
    # We want to find a sequence of length L > base_len
    
    global_timeout_flag = False

    def dfs(seq, sums, suffix_sums, last, target_len):
        nonlocal global_timeout_flag
        
        if global_timeout_flag:
            return None
            
        if is_time_up():
            global_timeout_flag = True
            return None
            
        if len(seq) == target_len:
            return seq[:]
        
        # Pruning: Not enough numbers left
        remaining_slots = target_len - len(seq)
        available_numbers = 256 - last
        if available_numbers < remaining_slots:
            return None
            
        # Try next numbers
        # Heuristic: Smaller numbers first?
        # In some cases, larger numbers might be better to space out sums?
        # But we are bounded by 256.
        # Let's try increasing order.
        
        for x in range(last + 1, 257):
            # Calculate new sums
            new_sums = [x]
            for s in suffix_sums:
                new_sums.append(x + s)
            
            # Check collision
            if sums.intersection(new_sums):
                continue
            
            # Update state
            old_sums_len = len(sums)
            sums.update(new_sums)
            
            # Recurse
            result = dfs(seq + [x], sums, new_sums, x, target_len)
            
            if result is not None:
                return result
            
            # Backtrack
            # Remove new_sums from sums
            # Since we know they were not in sums before, we can just remove them
            for s in new_sums:
                sums.remove(s)
                
            if global_timeout_flag:
                return None

    # Try to find longer sequences
    # We start from base_len + 1
    # We cap the search depth or time per length?
    # Global deadline is the main constraint.
    
    for target_len in range(base_len + 1, base_len + 20): # Arbitrary upper bound
        if global_timeout_flag:
            break
            
        # Reset flag? No, if we hit deadline, we stop everything.
        # But we need to distinguish "exhausted" from "timeout" to break early if no solution exists.
        # If dfs returns None and global_timeout_flag is False, it means search space exhausted.
        
        # Initialize DFS from empty state?
        # Or from greedy state?
        # Starting from empty is safer to find *any* sequence of that length.
        # However, the search space is huge.
        # Is it better to start from the greedy sequence and try to "modify" or "extend"?
        # Greedy sequence is a valid prefix.
        # But we might need to change earlier elements to allow a longer total sequence.
        # So we must search from the root (empty sequence).
        
        result = dfs([], set(), [], 0, target_len)
        
        if result is not None:
            best_seq = result
            # Continue to next length
        else:
            if global_timeout_flag:
                break
            else:
                # Search space exhausted, no solution of this length exists.
                # Since valid lengths are contiguous [1, N_max], and we found none for L,
                # N_max < L. So we can stop.
                break

    return best_seq
