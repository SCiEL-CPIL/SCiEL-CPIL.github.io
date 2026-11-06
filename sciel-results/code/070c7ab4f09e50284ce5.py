import time
import random
import sys

def construct():
        start_time = time.time()
        time_limit = 85.0  # Safety margin for 90s kill
        
        MAX_VAL = 256
        # Max possible sum for a sequence of length up to 20 with values up to 256
        # Sum <= 20 * 256 = 5120. 
        # Let's use 10000 to be safe.
        MAX_SUM = 10000
        
        best_seq = []
        best_len = 0
        
        def is_valid_candidate(x, prefix_sums, forbidden_mask, last_val):
            if x <= last_val:
                return False, []
            if x > MAX_VAL:
                return False, []
            
            new_prefix = prefix_sums[-1] + x
            new_diffs = []
            for p in prefix_sums:
                d = new_prefix - p
                if d >= MAX_SUM or (forbidden_mask >> d) & 1:
                    return False, []
                new_diffs.append(d)
            return True, new_diffs

        def dfs(depth, last_val, prefix_sums, forbidden_mask, current_seq, target_len):
            nonlocal best_seq, best_len
            nonlocal start_time
            
            if time.time() > time_limit:
                return False
            
            if depth == target_len:
                if depth > best_len:
                    best_len = depth
                    best_seq = current_seq[:]
                return True
            
            # Pruning: If we can't reach target_len
            # Max remaining steps is limited by MAX_VAL - last_val
            # But we need strictly increasing, so max steps is MAX_VAL - last_val
            if depth + (MAX_VAL - last_val) < target_len:
                return False
                
            # Generate candidates
            # Heuristic: Try values that are "safe"
            # We want small values to allow more steps, but they cause more collisions.
            # Let's iterate from last_val + 1 upwards.
            
            found = False
            for x in range(last_val + 1, MAX_VAL + 1):
                if time.time() > time_limit:
                    return False
                
                valid, new_diffs = is_valid_candidate(x, prefix_sums, forbidden_mask, last_val)
                if not valid:
                    continue
                
                # Commit
                new_prefix = prefix_sums[-1] + x
                prefix_sums.append(new_prefix)
                new_mask = forbidden_mask
                for d in new_diffs:
                    new_mask |= (1 << d)
                current_seq.append(x)
                
                # Recursive call
                if dfs(depth + 1, x, prefix_sums, new_mask, current_seq, target_len):
                    found = True
                    # If we found a solution of target_len, we can stop?
                    # Or keep searching for longer?
                    # We are targeting a specific length.
                    # If we want max length, we should try to extend.
                    # But this DFS is for a fixed target_len.
                    pass
                
                # Backtrack
                current_seq.pop()
                prefix_sums.pop()
                
                if found:
                    return True
                
            return False

        # Strategy: Iterative Deepening
        # Start with a known good sequence (powers of 2)
        # Try to extend it.
        
        # Initialize with powers of 2
        base_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        best_seq = base_seq
        best_len = len(base_seq)
        
        # Try to find longer sequences
        # We can't just append to powers of 2.
        # We need to search from scratch or modify the base.
        
        # Let's run a general DFS for length 10, 11, etc.
        # To make it faster, we can use a heuristic to order candidates.
        
        def dfs_heuristic(depth, last_val, prefix_sums, forbidden_mask, current_seq, target_len):
            nonlocal best_seq, best_len
            nonlocal start_time
            
            if time.time() > time_limit:
                return False
            
            if depth == target_len:
                if depth > best_len:
                    best_len = depth
                    best_seq = current_seq[:]
                return True
            
            if depth + (MAX_VAL - last_val) < target_len:
                return False
                
            # Candidates
            candidates = []
            for x in range(last_val + 1, MAX_VAL + 1):
                valid, new_diffs = is_valid_candidate(x, prefix_sums, forbidden_mask, last_val)
                if valid:
                    candidates.append((x, new_diffs))
            
            if not candidates:
                return False
            
            # Heuristic: Sort by x (smaller first)
            # Or sort by "flexibility"?
            # Smaller x leaves more room for larger x later.
            # So smaller first is generally good for maximizing length.
            
            for x, new_diffs in candidates:
                if time.time() > time_limit:
                    return False
                
                new_prefix = prefix_sums[-1] + x
                prefix_sums.append(new_prefix)
                new_mask = forbidden_mask
                for d in new_diffs:
                    new_mask |= (1 << d)
                current_seq.append(x)
                
                if dfs_heuristic(depth + 1, x, prefix_sums, new_mask, current_seq, target_len):
                    return True
                
                current_seq.pop()
                prefix_sums.pop()
                
            return False

        # Try lengths 10, 11, 12...
        # Start from length 10
        for target_len in range(best_len + 1, 20):
            if time.time() > time_limit:
                break
            
            # Reset state for new search
            # We can try multiple random starts or just greedy from scratch.
            # Let's try a few random restarts for this target length.
            
            found = False
            for restart in range(50): # 50 restarts per length
                if time.time() > time_limit:
                    break
                
                # Randomized Greedy to get a starting point?
                # Or just DFS from empty?
                # DFS from empty with target_len 10 might be too slow if branching factor is high.
                # Branching factor is ~256.
                # Depth 10. 256^10 is huge.
                # But constraints prune heavily.
                
                # Let's use a hybrid: Greedy to get to length 8, then DFS to extend.
                # Or just DFS with strong pruning.
                
                # Let's try DFS from scratch with a time limit per restart.
                restart_deadline = time.time() + 1.0 # 1 second per restart
                
                # Save global deadline
                global_deadline = time_limit
                
                # Monkey patch time check? No, just pass deadline.
                # Let's refactor dfs to take deadline.
                
                pass
            
            # If found, best_len is updated.
            if best_len >= target_len:
                continue
            else:
                # If not found, break? Or try harder?
                # If we can't find length 10, we might not find 11.
                # But maybe a different search strategy would work.
                pass
        
        return best_seq
