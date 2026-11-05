import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Try to find the longest sequence
    # Upper bound on length: 
    # Max value 256. 
    # Sequence 1, 2, ..., n. Sums are 1..n(n+1)/2. 
    # But we can choose larger values.
    # Let's guess a reasonable max length. 
    # With values up to 256, we can likely fit around 15-25 elements.
    # Let's try from 30 down to 1.
    
    for L in range(30, 0, -1):
        if time.time() > deadline:
            break
        seq = find_sequence(L, deadline)
        if seq is not None:
            return seq
            
    # Fallback: return a known short sequence if search fails
    return [1, 2]

def find_sequence(L, deadline):
    # Backtracking search for a sequence of length L
    # current_seq: list of chosen numbers
    # sums: set of all consecutive sums
    
    def dfs(current_seq, sums, last_val):
        if time.time() > deadline:
            return None
        
        if len(current_seq) == L:
            return current_seq[:]
        
        # Determine range for next value
        # Must be > last_val and <= 256
        # Also, we need to fit (L - len(current_seq)) more numbers.
        # Min next is last_val + 1.
        # Max next is 256 - (L - len(current_seq) - 1) to leave room for increasing sequence.
        
        current_len = len(current_seq)
        remaining = L - current_len
        
        # Heuristic: Try smaller values first? Or larger?
        # Smaller values keep sums smaller, potentially avoiding large sum collisions?
        # Actually, distinctness is the key.
        # Let's iterate candidates.
        
        # Optimization: 
        # If we pick x, new sums are x, x + a[-1], x + a[-1] + a[-2], ...
        
        min_candidate = last_val + 1
        # We must leave enough room for the remaining 'remaining' numbers to be strictly increasing up to 256.
        # a_{k+1} < a_{k+2} < ... < a_L <= 256
        # So a_{k+1} <= 256 - (remaining - 1)
        max_candidate = 256 - (remaining - 1)
        
        if min_candidate > max_candidate:
            return None
            
        # Try candidates
        # Randomizing the order can help escape bad branches, but deterministic is better for reproducibility and often speed if pruned well.
        # Let's try deterministic ascending first.
        
        for x in range(min_candidate, max_candidate + 1):
            if time.time() > deadline:
                return None
                
            # Calculate new sums
            new_sums = []
            total = 0
            # Sums ending at x
            for a in reversed(current_seq):
                total += a
                new_sums.append(total)
            new_sums.append(x) # The sum consisting of just x
            
            # Check for duplicates within new_sums
            if len(new_sums) != len(set(new_sums)):
                continue
                
            # Check for duplicates with existing sums
            # Note: new_sums are all distinct from each other.
            # We just need to check if any are in `sums`
            if any(s in sums for s in new_sums):
                continue
                
            # If valid, recurse
            new_sums_set = sums.union(new_sums)
            res = dfs(current_seq + [x], new_sums_set, x)
            if res is not None:
                return res
                
        return None

    # Initial call
    # last_val = 0, sums = set()
    return dfs([], set(), 0)
