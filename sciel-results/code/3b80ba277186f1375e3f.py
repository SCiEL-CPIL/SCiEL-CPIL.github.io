import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    best_seq = []
    best_len = 0

    # Greedy baseline: Always pick smallest valid next number
    def greedy_smallest():
        seq = []
        sums = set()
        for x in range(1, 257):
            # Check if x can be appended
            # New sums: x, x+suffix_sum
            new_sums = {x}
            s = 0
            for v in reversed(seq):
                s += v
                new_sums.add(x + s)
            if not sums.intersection(new_sums):
                seq.append(x)
                sums.update(new_sums)
        return seq

    best_seq = greedy_smallest()
    best_len = len(best_seq)

    # Backtracking search to find longer sequences
    # Strategy: DFS with pruning.
    
    def dfs_backtrack(current_seq, current_sums):
        nonlocal best_seq, best_len
        
        if len(current_seq) > best_len:
            best_seq = list(current_seq)
            best_len = len(current_seq)
            
        # Check time
        if time.time() > deadline:
            return

        # Try to add next element
        start_val = current_seq[-1] + 1 if current_seq else 1
        
        for x in range(start_val, 257):
            # Calculate new sums
            new_sums = set(current_sums)
            new_sums.add(x)
            s = 0
            for v in reversed(current_seq):
                s += v
                new_sums.add(x + s)
            
            if len(new_sums) == len(current_sums) + len(current_seq) + 1: # No collisions
                # Recurse
                dfs_backtrack(current_seq + [x], new_sums)
                
                if time.time() > deadline:
                    return
                
                # Pruning: If we know we can't beat best_len with remaining numbers
                # Remaining numbers greater than x: 256 - x
                # Max additional length: 256 - x
                # Current length: len(current_seq) + 1
                # If (len(current_seq) + 1 + (256 - x)) <= best_len, we can break
                if (len(current_seq) + 1 + (256 - x)) <= best_len:
                    break

    # Run deterministic DFS first
    dfs_backtrack([], set())
    
    # Run a few randomized DFS if time permits
    # To allow randomization in DFS, we need to shuffle candidates
    def dfs_randomized(current_seq, current_sums, rng):
        nonlocal best_seq, best_len
        
        if len(current_seq) > best_len:
            best_seq = list(current_seq)
            best_len = len(current_seq)
            
        if time.time() > deadline:
            return

        start_val = current_seq[-1] + 1 if current_seq else 1
        
        candidates = list(range(start_val, 257))
        rng.shuffle(candidates)
        
        for x in candidates:
            new_sums = set(current_sums)
            new_sums.add(x)
            s = 0
            for v in reversed(current_seq):
                s += v
                new_sums.add(x + s)
            
            if len(new_sums) == len(current_sums) + len(current_seq) + 1:
                dfs_randomized(current_seq + [x], new_sums, rng)
                
                if time.time() > deadline:
                    return
                
                # Pruning
                if (len(current_seq) + 1 + (256 - x)) <= best_len:
                    break

    rng = random.Random(42)
    while time.time() < deadline - 5:
        dfs_randomized([], set(), rng)
        rng = random.Random(rng.randint(0, 10000))

    return best_seq
