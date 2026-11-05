"""Improved program: multiple greedy strategies to maximize sequence length."""
import time
from itertools import combinations

def _try_extend(values, sums, x):
    """Check if x can be appended to values without sum collision. Returns new sums if valid."""
    tails = []
    total = x
    for a in reversed(values):
        total += a
        tails.append(total)
    if sums.intersection(tails):
        return None
    new_sums = sums.copy()
    new_sums.update(tails)
    return new_sums

def _greedy_forward(values, sums, start, end, step):
    """Greedy: iterate from start to end in given step, add if possible."""
    vals = list(values)
    s = set(sums)
    for x in range(start, end, step):
        ns = _try_extend(vals, s, x)
        if ns is not None:
            vals.append(x)
            s = ns
    return vals, s

def _greedy_reverse(values, sums, start, end, step):
    """Greedy: iterate from end down to start in given step, add if possible."""
    vals = list(values)
    s = set(sums)
    for x in range(end, start - 1, step):
        ns = _try_extend(vals, s, x)
        if ns is not None:
            vals.append(x)
            s = ns
    return vals, s

def _greedy_largest_first(values, sums, candidates):
    """Try candidates in descending order, add if possible."""
    vals = list(values)
    s = set(sums)
    for x in sorted(candidates, reverse=True):
        ns = _try_extend(vals, s, x)
        if ns is not None:
            vals.append(x)
            s = ns
    return vals, s

def _greedy_smallest_first(values, sums, candidates):
    """Try candidates in ascending order, add if possible."""
    vals = list(values)
    s = set(sums)
    for x in sorted(candidates):
        ns = _try_extend(vals, s, x)
        if ns is not None:
            vals.append(x)
            s = ns
    return vals, s

def _validate(seq):
    """Verify sequence is valid: strictly increasing, in [1,256], all consecutive block sums distinct."""
    if not seq:
        return True
    if any(x < 1 or x > 256 for x in seq):
        return False
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i + 1]:
            return False
    sums = set()
    for i in range(len(seq)):
        s = 0
        for j in range(i, len(seq)):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe under 90s limit
    
    best_seq = []
    best_len = 0
    
    # Strategy 1: Original greedy (smallest first, 1 to 256)
    vals, sums = [], set()
    for x in range(1, 257):
        ns = _try_extend(vals, sums, x)
        if ns is not None:
            vals.append(x)
            sums = ns
    if len(vals) > best_len:
        best_seq = vals
        best_len = len(vals)
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 2: Largest first (256 down to 1)
    vals, sums = [], set()
    for x in range(256, 0, -1):
        ns = _try_extend(vals, sums, x)
        if ns is not None:
            vals.append(x)
            sums = ns
    if len(vals) > best_len:
        best_seq = vals
        best_len = len(vals)
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 3: Start with a medium value, then fill
    # Try different starting points
    for start_val in [64, 128, 192]:
        vals, sums = [start_val], {start_val}
        # Add smaller values first (descending from start_val-1)
        for x in range(start_val - 1, 0, -1):
            ns = _try_extend(vals, sums, x)
            if ns is not None:
                vals.append(x)
                sums = ns
        # Add larger values (ascending from start_val+1)
        for x in range(start_val + 1, 257):
            ns = _try_extend(vals, sums, x)
            if ns is not None:
                vals.append(x)
                sums = ns
        if len(vals) > best_len:
            best_seq = vals
            best_len = len(vals)
        
        if time.time() > deadline:
            return best_seq
    
    # Strategy 4: Greedy with step patterns
    # Try step 1 but starting from different offsets
    for offset in [0, 1, 2, 3, 4, 5]:
        vals, sums = [], set()
        for x in range(offset + 1, 257):
            ns = _try_extend(vals, sums, x)
            if ns is not None:
                vals.append(x)
                sums = ns
        if len(vals) > best_len:
            best_seq = vals
            best_len = len(vals)
        
        if time.time() > deadline:
            return best_seq
    
    # Strategy 5: Try with some initial seed values
    # Seed with a few well-spaced values, then fill
    seed_sets = [
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 3, 7, 15, 31, 63, 127, 255],
        [10, 20, 40, 80, 160],
        [16, 32, 64, 128, 192],
    ]
    
    for seed in seed_sets:
        if time.time() > deadline:
            return best_seq
        
        vals = list(seed)
        sums = set()
        for i in range(len(vals)):
            s = 0
            for j in range(i, len(vals)):
                s += vals[j]
                sums.add(s)
        
        # Fill remaining values
        remaining = [x for x in range(1, 257) if x not in seed]
        for x in remaining:
            ns = _try_extend(vals, sums, x)
            if ns is not None:
                vals.append(x)
                sums = ns
        
        if len(vals) > best_len:
            best_seq = vals
            best_len = len(vals)
    
    # Final validation
    if not _validate(best_seq):
        # Fallback to original greedy
        vals, sums = [], set()
        for x in range(1, 257):
            ns = _try_extend(vals, sums, x)
            if ns is not None:
                vals.append(x)
                sums = ns
        best_seq = vals
    
    return best_seq
