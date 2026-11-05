"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random

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

def _greedy_forward():
    """Greedy: try values 1..256, append if valid."""
    values = []
    sums = set()
    for x in range(1, 257):
        # New consecutive block sums when appending x
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _greedy_reverse():
    """Greedy: try values 256..1, add to set if valid when sorted."""
    S = []
    for x in range(256, 0, -1):
        candidate = sorted(S + [x])
        if _check_valid(candidate):
            S.append(x)
    return sorted(S)

def _greedy_random(order_seed=None):
    """Greedy with random ordering of candidates."""
    if order_seed is None:
        order_seed = random.randint(0, 2**31 - 1)
    rng = random.Random(order_seed)
    candidates = list(range(1, 257))
    rng.shuffle(candidates)
    S = []
    for x in candidates:
        candidate = sorted(S + [x])
        if _check_valid(candidate):
            S.append(x)
    return sorted(S)

import time

def _backtrack_search(time_limit=50):
    """
    Depth-first search with heuristic ordering and time limit.
    Tries to build a sequence by adding one element at a time.
    Heuristic: Prefer larger numbers (descending order).
    """
    start_time = time.time()
    best_seq = []
    
    def dfs(seq, sums, depth):
        nonlocal best_seq, start_time
        
        # Time check
        if time.time() - start_time > time_limit:
            return
            
        # Update best
        if len(seq) > len(best_seq):
            best_seq = seq[:]
            
        # Heuristic: If we have a very long sequence, we can prune
        # deeper searches if we are close to the theoretical max?
        # For now, just continue.
        
        last_val = seq[-1] if seq else 0
        
        # Try candidates in descending order
        # Optimization: Skip candidates that are too close to last_val?
        # No, just try them.
        
        for x in range(256, last_val, -1):
            if x in seq:
                continue
                
            # Check validity
            new_sums = set()
            s = x
            if s in sums:
                continue
            new_sums.add(s)
            
            valid = True
            for a in reversed(seq):
                s += a
                if s in sums:
                    valid = False
                    break
                new_sums.add(s)
                
            if valid:
                new_seq = seq + [x]
                new_sums_set = sums.union(new_sums)
                dfs(new_seq, new_sums_set, depth + 1)
                
                # Pruning: If we found a sequence of length >= 20, 
                # we might want to stop exploring this branch?
                # No, we want the MAX length.
                
    dfs([], set(), 0)
    return best_seq

def construct():
    best = []
    
    # Strategy 1: Forward greedy (original)
    seq1 = _greedy_forward()
    if len(seq1) > len(best):
        best = seq1
    
    # Strategy 2: Reverse greedy
    seq2 = _greedy_reverse()
    if len(seq2) > len(best):
        best = seq2
    
    # Strategy 3: Multiple random restarts
    for seed in range(50):
        seq3 = _greedy_random(order_seed=seed)
        if len(seq3) > len(best):
            best = seq3
    
    # Strategy 4: Try a few more random seeds with different rng
    for seed in range(50, 150):
        seq4 = _greedy_random(order_seed=seed * 7919 + 13)
        if len(seq4) > len(best):
            best = seq4
            
    # Strategy 5: Backtracking search
    seq5 = _backtrack_search(time_limit=50)
    if len(seq5) > len(best):
        best = seq5
        
    return best
# EVOLVE-BLOCK-END
