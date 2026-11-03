"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import math
from itertools import permutations

# EVOLVE-BLOCK-START
def construct(n):
    """Return shortest superpermutation found for n symbols within time budget."""
    if n <= 2:
        return "121" if n == 2 else "1"
    
    start_time = time.time()
    TIME_LIMIT = 140  # Safety margin before 150s limit
    
    # For small n, use exact known solutions
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431214321342132431213243214321"
    
    # Build initial superpermutation using recursive construction
    word = build_initial_superperm(n)
    
    # Verify it's valid
    if not is_valid_superperm(word, n):
        return "".join(map(str, word))
    
    # For n >= 5, try to shorten using structural compression
    if n >= 5 and time.time() - start_time < TIME_LIMIT:
        word = compress_superperm(word, n, start_time, TIME_LIMIT)
    
    return "".join(map(str, word))

def build_initial_superperm(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def is_valid_superperm(word, n):
    if len(word) < n:
        return False
    try:
        arr = np.array(word, dtype=np.uint8)
        # Use sliding window view
        w = np.lib.stride_tricks.sliding_window_view(arr, n)
        # Check if each window is a permutation of 1..n
        # Sort each window and compare to [1, 2, ..., n]
        sorted_w = np.sort(w, axis=1)
        target = np.arange(1, n + 1, dtype=np.uint8)
        is_perm = np.all(sorted_w == target, axis=1)
        
        if not np.any(is_perm):
            return False
            
        # Get unique permutations
        # Extract only the permutation windows
        perm_windows = w[is_perm]
        
        # Count unique rows
        # For n=7, 5040 perms. 
        # np.unique is efficient for this size.
        unique_count = len(np.unique(perm_windows, axis=0))
        
        return unique_count == math.factorial(n)
    except Exception:
        # Fallback to pure python if numpy fails (shouldn't happen)
        seen = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i + n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen) == math.factorial(n)

def compress_superperm(word, n, start_time, time_limit):
    """Compress a superpermutation by removing redundant symbols."""
    current = list(word)
    target_count = math.factorial(n)
    
    # Greedy removal of single redundant symbols
    # We iterate until no more removals are possible or time runs out
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try removing from right to left
        for pos in range(len(current) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            if len(current) - 1 < n:
                continue
            
            # Check if removing current[pos] keeps validity
            # We can do a quick check: if the character is part of a unique permutation, it's likely critical
            # But we must verify.
            trial = current[:pos] + current[pos + 1:]
            
            if is_valid_superperm(trial, n):
                current = trial
                improved = True
                break
    
    # Second pass: try to remove larger chunks?
    # Given time constraints, single symbol removal is the most reliable.
    # We can try a few more passes if time allows.
    if time.time() - start_time < time_limit:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for pos in range(len(current) - 1, -1, -1):
                if time.time() - start_time > time_limit:
                    break
                if len(current) - 1 < n:
                    continue
                trial = current[:pos] + current[pos + 1:]
                if is_valid_superperm(trial, n):
                    current = trial
                    improved = True
                    break
                    
    return current
# EVOLVE-BLOCK-END
