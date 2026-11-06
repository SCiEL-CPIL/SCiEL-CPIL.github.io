"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_DEADLINE = 85.0  # safety margin before 90s kill

def _is_valid_prefix(seq):
    """Check if the prefix is strictly increasing and has distinct block sums."""
    if not seq:
        return True
    # strictly increasing
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    # all in [1, 256]
    for v in seq:
        if v < 1 or v > 256:
            return False
    # distinct block sums
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build():
    """Original greedy strategy as a baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _backtrack_search(seq, sums, max_len=20):
    """
    Backtracking search to extend the sequence.
    seq: current sequence
    sums: set of current block sums
    max_len: target length to try to reach
    Returns the longest valid sequence found.
    """
    if len(seq) >= max_len:
        return seq[:]
    
    last_val = seq[-1] if seq else 0
    best = seq[:]
    
    # Try to add next element
    # Heuristic: try smaller values first to leave room for more elements
    # But also try larger values to avoid sum collisions
    # We'll try values from last_val+1 to 256
    # To make it smarter, we can prioritize values that create new sums
    # that are "far" from existing sums, but for now, let's just iterate
    
    # Optimization: precompute which values are viable
    # A value x is viable if:
    # 1. x > last_val
    # 2. x <= 256
    # 3. All new block sums (x, x+seq[-1], x+seq[-1]+seq[-2], ...) are not in sums
    
    for x in range(last_val + 1, 257):
        # Check new block sums
        new_sums = []
        s = x
        for a in reversed(seq):
            new_sums.append(s)
            s += a
        new_sums.append(x)
        
        # Check if any new sum is in existing sums
        if any(ns in sums for ns in new_sums):
            continue
        
        # Also check if new sums collide with each other
        if len(set(new_sums)) != len(new_sums):
            continue
        
        # Try this value
        new_seq = seq + [x]
        new_sums_set = sums | set(new_sums)
        
        # Recurse
        result = _backtrack_search(new_seq, new_sums_set, max_len)
        if len(result) > len(best):
            best = result[:]
        
        # If we found a solution of max_len, we can stop early
        if len(best) >= max_len:
            return best
    
    return best

def construct():
    start_time = time.time()
    
    # Start with greedy baseline
    best_seq = _greedy_build()
    best_len = len(best_seq)
    
    # Try to improve using backtracking
    # We'll try to extend the greedy sequence or find a better one
    # Since backtracking can be expensive, we'll use it carefully
    
    # First, try to extend the greedy sequence
    if time.time() < _DEADLINE:
        # Compute sums for greedy sequence
        sums = set()
        n = len(best_seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += best_seq[j]
                sums.add(s)
        
        # Try to extend
        extended = _backtrack_search(best_seq, sums, best_len + 5)
        if len(extended) > best_len:
            best_seq = extended
            best_len = len(best_seq)
    
    # If we still have time, try different starting points
    # We can try starting with different first elements
    if time.time() < _DEADLINE:
        for first_val in range(1, 20):
            if time.time() > _DEADLINE:
                break
            # Try to build a sequence starting with first_val
            # Use backtracking
            seq = [first_val]
            sums = {first_val}
            result = _backtrack_search(seq, sums, best_len + 1)
            if len(result) > best_len:
                best_seq = result
                best_len = len(best_seq)
    
    # Final safety check
    if not _is_valid_prefix(best_seq):
        # Fallback to greedy
        best_seq = _greedy_build()
    
    return best_seq
# EVOLVE-BLOCK-END
