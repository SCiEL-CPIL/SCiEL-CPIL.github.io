import time
import random
import bisect

def _is_valid_candidate(candidate):
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build(order):
    selected = []
    for x in order:
        candidate = selected + [x]
        candidate.sort()
        if _is_valid_candidate(candidate):
            selected = candidate
    return selected

def _try_insert(selected, x):
    """Try to insert x into sorted selected and check validity."""
    # Find insertion point
    pos = bisect.bisect_left(selected, x)
    if pos < len(selected) and selected[pos] == x:
        return None # Duplicate
    new_candidate = selected[:pos] + [x] + selected[pos:]
    if _is_valid_candidate(new_candidate):
        return new_candidate
    return None

def _local_search(selected, available, deadline):
    """Try to improve the sequence by inserting available numbers."""
    current = selected[:]
    available_set = set(available)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to insert each available number
        # Sort available to try smaller/larger first? 
        # Random order might be better to avoid local minima
        avail_list = list(available_set)
        random.shuffle(avail_list)
        for x in avail_list:
            if x in current:
                continue
            new_seq = _try_insert(current, x)
            if new_seq:
                current = new_seq
                available_set.remove(x)
                improved = True
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_seq = []
    
    # Baselines
    seq = _greedy_build(list(range(1, 257)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    seq = _greedy_build(list(range(256, 0, -1)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    # Random Restarts with Local Search
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Generate a random order, biased towards larger numbers sometimes
        # Strategy 1: Pure random
        order = list(range(1, 257))
        rng.shuffle(order)
        
        # Strategy 2: Sorted descending (often better for sums)
        # We already did this, but maybe mix?
        # Let's stick to random shuffle for diversity
        
        seq = _greedy_build(order)
        
        # Local Search Phase
        if len(seq) > len(best_seq):
            best_seq = seq
            
        # Try to improve the current sequence (or best_seq?)
        # It's better to improve the current sequence and then compare
        available = set(range(1, 257)) - set(seq)
        improved_seq = _local_search(seq, available, deadline)
        
        if len(improved_seq) > len(best_seq):
            best_seq = improved_seq
            
        # Early termination if we found a very long sequence?
        # Max possible length is 256, but constraints are tight.
        
    return best_seq
