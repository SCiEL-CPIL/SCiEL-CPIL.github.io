import time
import sys
import random

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    # Leave 5s buffer for safety against 90s limit
    deadline = start_time + 85.0
    
    def is_time_up():
        return time.time() > deadline

    best_seq = []
    best_len = 0
    
    def dfs_opt(prefix_sums, diffs, last_val, current_seq, rng):
        nonlocal best_seq, best_len
        
        if is_time_up():
            return
            
        # Pruning: Max possible additional elements
        if len(current_seq) + (256 - last_val) <= best_len:
            return

        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = list(current_seq)
            
        # Generate candidates
        candidates = list(range(last_val + 1, 257))
        
        # Shuffle to explore different branches
        # Bias towards smaller? 
        # If we shuffle uniformly, we might pick large x early and waste space.
        # Let's try: sort by (x, random) ? No.
        # Let's try: 50% chance to pick smallest, else random?
        # Or just random.
        rng.shuffle(candidates)
        
        for x in candidates:
            if is_time_up():
                return
            
            new_prefix = prefix_sums[-1] + x
            
            # Check collisions
            conflict = False
            for p in prefix_sums:
                if (new_prefix - p) in diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Add new differences
            for p in prefix_sums:
                diffs.add(new_prefix - p)
            
            prefix_sums.append(new_prefix)
            current_seq.append(x)
            
            dfs_opt(prefix_sums, diffs, x, current_seq, rng)
            
            # Backtrack
            current_seq.pop()
            prefix_sums.pop()
            for p in prefix_sums:
                diffs.discard(new_prefix - p)
                
            if is_time_up():
                return

    # Try a few different starting strategies
    # 1. Start with 1, random order
    # 2. Start with 1, ascending order (greedy-like but with backtracking)
    
    # Let's run a few iterations with different seeds
    for i in range(10): # Max 10 restarts, but time limited
        if is_time_up():
            break
        
        rng = random.Random(12345 + i)
        
        # Start with 1
        # If we start with 1, we fix a_1=1.
        # Is it optimal to start with 1?
        # Probably, as it minimizes the prefix sum growth.
        
        dfs_opt([0], set(), 0, [], rng)
        
        # If we found a very long sequence, we might stop?
        # No, we want to maximize.
        
    return best_seq
