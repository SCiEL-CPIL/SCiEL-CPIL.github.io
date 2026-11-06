import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 50.0  # Leave 10s margin for safety
    
    best_seq = []
    
    # Helper to check validity and update sums
    # Returns (is_valid, new_sums_list)
    def check_and_extend(seq, x):
        new_sums = []
        total = x
        # Sums ending at x
        for a in reversed(seq):
            total += a
            new_sums.append(total)
        new_sums.append(x)
        
        # Check against existing sums
        # We maintain a global set of sums for the current path
        return new_sums

    def dfs(seq, sums_set, depth, target_len):
        nonlocal best_seq, start_time, TIME_LIMIT
        
        if time.time() - start_time > TIME_LIMIT:
            return False
            
        if len(seq) == target_len:
            if len(seq) > len(best_seq):
                best_seq = list(seq)
            return True
            
        if not seq:
            # First element
            # Heuristic: Start small to allow more room?
            # Or start with 1?
            # Let's try a range of starting values
            candidates = list(range(1, 20))
            random.shuffle(candidates)
        else:
            last = seq[-1]
            # Next must be > last
            # Heuristic: Try values close to last first? Or spread out?
            # Usually, smaller increments are safer for distinct sums initially,
            # but might cause collisions later.
            # Let's try a window of candidates.
            # To maximize length, we want small numbers.
            # But we need distinct sums.
            
            # Pruning: 
            # Max possible sum of remaining elements?
            # Not easy to bound tightly.
            
            candidates = []
            # Try a mix of small and medium steps
            # Step size 1..10
            for d in range(1, 11):
                candidates.append(last + d)
            # Also try some larger jumps to avoid local minima
            for d in range(10, 20):
                if random.random() < 0.5:
                    candidates.append(last + d)
            
            # Filter valid candidates
            valid_candidates = []
            for x in candidates:
                if x > 256:
                    continue
                # Check sums
                new_sums = []
                total = x
                for a in reversed(seq):
                    total += a
                    new_sums.append(total)
                new_sums.append(x)
                
                # Check intersection
                # Optimization: Check one by one
                is_valid = True
                for s in new_sums:
                    if s in sums_set:
                        is_valid = False
                        break
                
                if is_valid:
                    valid_candidates.append((x, new_sums))
            
            # Shuffle to break determinism
            random.shuffle(valid_candidates)
            
        if not seq:
             # Handle first element case separately in loop below
             pass
        else:
             # Use valid_candidates
             pass

        # Unified loop for candidates
        if not seq:
            candidates_to_try = [(x, None) for x in candidates if x <= 256]
            # For first element, sums are just {x}
        else:
            candidates_to_try = valid_candidates

        for x, pre_computed_sums in candidates_to_try:
            if time.time() - start_time > TIME_LIMIT:
                return False
            
            if pre_computed_sums is None:
                # First element
                new_sums = [x]
            else:
                new_sums = pre_computed_sums
            
            # Add to sets
            seq.append(x)
            sums_set.update(new_sums)
            
            # Recurse
            if dfs(seq, sums_set, depth + 1, target_len):
                return True
            
            # Backtrack
            seq.pop()
            sums_set.difference_update(new_sums)
            
        return False

    # Try to find sequences of decreasing length
    # Start from a high estimate. 
    # Theoretical max for N=256 is likely around 12-14.
    # Let's try 14 down to 8.
    
    for target_len in range(14, 7, -1):
        # Multiple restarts for each target length
        max_restarts = 50
        for i in range(max_restarts):
            if time.time() - start_time > TIME_LIMIT:
                break
            
            # Randomize the seed for this restart
            random.seed(i * 1000 + target_len)
            
            seq = []
            sums_set = set()
            
            # Run DFS
            found = dfs(seq, sums_set, 0, target_len)
            
            if found:
                # We found a sequence of this length
                # Since we are going downwards, this is the max found so far
                # We can return immediately or try to see if we can find a better one?
                # The task is to maximize length. If we found length L, we can return it.
                # But maybe we can find L+1? No, we are iterating downwards.
                # So if we found 14, we return 14.
                # If we fail 14, we try 13.
                # If we found 13, we return 13.
                return best_seq
            
            # If not found, continue to next restart
            # If all restarts for this length fail, we go to next lower length
            # But we should keep the best_seq found in previous (higher) lengths?
            # No, if we failed to find length 14, best_seq might be empty or from a previous lower length?
            # We iterate 14, 13, 12...
            # If we fail 14, best_seq is empty.
            # If we fail 13, best_seq is empty.
            # If we find 12, we return it.
            
    # If loop finishes without returning, return the best_seq found (which might be empty or a partial one if we modified logic)
    # To be safe, ensure we return a valid sequence.
    # The DFS only updates best_seq if it reaches target_len.
    # If we didn't reach any target_len, best_seq is [].
    # We should fall back to a simple greedy sequence to ensure non-empty result.
    
    if not best_seq:
        # Simple greedy fallback
        seq = []
        sums = set()
        for x in range(1, 257):
            tails = []
            total = x
            for a in reversed(seq):
                total += a
                tails.append(total)
            tails.append(x)
            if not sums.intersection(tails):
                seq.append(x)
                sums.update(tails)
        best_seq = seq

    return best_seq
