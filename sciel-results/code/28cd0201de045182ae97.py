"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random

import time

# EVOLVE-BLOCK-START
def _check_valid(seq):
    """Check if all consecutive block sums in seq are distinct."""
    n = len(seq)
    if n == 0:
        return True
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _find_golomb_ruler_increasing_gaps(max_val=256, time_limit=55):
    """
    Finds the longest strictly increasing sequence of integers in [1, max_val]
    such that the partial sums form a Golomb ruler (all pairwise differences distinct).
    This is equivalent to the task requirement.
    Uses Depth-First Search with backtracking.
    """
    start_time = time.time()
    best_len = 0
    best_seq = []
    
    def dfs(a, p, diffs, current_len, last_a):
        nonlocal best_len, best_seq
        nonlocal start_time
        
        if time.time() - start_time > time_limit:
            return
            
        if current_len > best_len:
            best_len = current_len
            best_seq = a.copy()
            
        # Pruning: Upper bound on remaining length
        # Available numbers: last_a + 1 to max_val
        # Count = max_val - last_a
        # If current_len + count <= best_len, we can't beat the best, so prune.
        remaining_count = max_val - last_a
        if current_len + remaining_count <= best_len:
            return

        # Try next gap a_k
        # Heuristic: Try smaller values first to allow more steps
        for next_a in range(last_a + 1, max_val + 1):
            if time.time() - start_time > time_limit:
                return
                
            new_p = p[-1] + next_a
            
            # Check new differences
            # New diffs are new_p - p[i] for i in 0..len(p)-1
            valid = True
            new_diffs = []
            for i in range(len(p)):
                d = new_p - p[i]
                if d in diffs:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
                
            # Recurse
            a.append(next_a)
            p.append(new_p)
            diffs.update(new_diffs)
            
            dfs(a, p, diffs, current_len + 1, next_a)
            
            # Backtrack
            a.pop()
            p.pop()
            for d in new_diffs:
                diffs.discard(d)
                
    # Initial state
    initial_p = [0]
    initial_diffs = set()
    dfs([], initial_p, initial_diffs, 0, 0)
    
    return best_seq

def construct():
    # Use the new backtracking approach
    seq = _find_golomb_ruler_increasing_gaps(max_val=256, time_limit=55)
    
    # Fallback to greedy if backtracking fails or returns empty
    if not seq:
        values = []
        sums = set()
        for x in range(1, 257):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        seq = values
        
    return seq
# EVOLVE-BLOCK-END
