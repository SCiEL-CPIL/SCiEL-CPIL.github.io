import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s limit

    def get_subarray_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid_insertion(seq, x, k, sums_set):
        """
        Check if inserting x at index k in seq results in all distinct subarray sums.
        seq is sorted.
        """
        n = len(seq)
        
        # New subarrays are those that include x.
        # Left parts: seq[i..k-1] for i in 0..k-1 (and empty)
        # Right parts: seq[k..j] for j in k..n-1 (and empty)
        
        # Precompute left sums
        left_sums = [0]
        s = 0
        for i in range(k - 1, -1, -1):
            s += seq[i]
            left_sums.append(s)
            
        # Precompute right sums
        right_sums = [0]
        s = 0
        for j in range(k, n):
            s += seq[j]
            right_sums.append(s)
            
        # Check new sums
        new_sums = set()
        for l in left_sums:
            for r in right_sums:
                ns = l + x + r
                if ns in sums_set:
                    return False
                if ns in new_sums:
                    return False
                new_sums.add(ns)
                
        return True

    def greedy_init():
        vals = []
        sums_set = set()
        for x in range(1, 257):
            if is_valid_insertion(vals, x, len(vals), sums_set):
                vals.append(x)
                # Update sums_set incrementally for efficiency in init
                # New sums are x, x+vals[-1], ...
                s = x
                sums_set.add(s)
                for v in reversed(vals[:-1]):
                    s += v
                    sums_set.add(s)
        return vals, sums_set

    # 1. Get initial greedy solution
    best_seq, best_sums = greedy_init()
    best_len = len(best_seq)
    
    current_seq = best_seq.copy()
    current_sums = best_sums.copy()
    
    def try_improve(seq, sums_set):
        """
        Attempts to insert elements into seq. Returns (new_seq, new_sums, changed).
        Greedily inserts the first valid element found.
        """
        improved = True
        while improved:
            improved = False
            n = len(seq)
            # Try all positions k from 0 to n
            for k in range(n + 1):
                # Determine valid range for x
                low = seq[k-1] + 1 if k > 0 else 1
                high = seq[k] - 1 if k < n else 256
                
                if low > high:
                    continue
                    
                # Try all x in [low, high]
                # Iterate from low to high to prefer smaller numbers (keeps sums small)
                for x in range(low, high + 1):
                    if is_valid_insertion(seq, x, k, sums_set):
                        # Insert x
                        new_seq = seq[:k] + [x] + seq[k:]
                        # Recompute sums for safety
                        new_sums = get_subarray_sums(new_seq)
                        
                        return new_seq, new_sums, True
        return seq, sums_set, False

    # Main Loop
    while time.time() - start_time < time_limit:
        # Try to improve current state
        new_seq, new_sums, changed = try_improve(current_seq, current_sums)
        
        if changed:
            if len(new_seq) > best_len:
                best_seq = new_seq
                best_sums = new_sums
                best_len = len(best_seq)
            current_seq = new_seq
            current_sums = new_sums
        else:
            # Stuck. Perform random kicks.
            # Try a few random deletions and re-improvements
            best_kick_seq = current_seq
            best_kick_sums = current_sums
            
            # Number of kicks based on remaining time
            kicks = 5
            for _ in range(kicks):
                if time.time() - start_time > time_limit:
                    break
                    
                kick_seq = current_seq.copy()
                if len(kick_seq) > 1:
                    idx_to_remove = random.randint(0, len(kick_seq) - 1)
                    kick_seq.pop(idx_to_remove)
                    kick_sums = get_subarray_sums(kick_seq)
                    
                    # Try to improve from this kicked state
                    new_kick_seq, new_kick_sums, changed_kick = try_improve(kick_seq, kick_sums)
                    
                    if len(new_kick_seq) > len(best_kick_seq):
                        best_kick_seq = new_kick_seq
                        best_kick_sums = new_kick_sums
                        
            # Update current state to the best kick result
            current_seq = best_kick_seq
            current_sums = best_kick_sums
            
            if len(current_seq) > best_len:
                best_seq = current_seq
                best_sums = current_sums
                best_len = len(best_seq)
                
    return best_seq

construct()
