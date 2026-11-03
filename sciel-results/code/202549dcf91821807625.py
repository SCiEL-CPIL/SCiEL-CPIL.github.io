import time
import sys
import itertools
import numpy as np
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin: evaluator kills at 150s, leave 15s buffer
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate standard recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        L = len(word)
        for i in range(L - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
        # Time check
        if time.time() > deadline - 10.0:
            break
    
    L = len(word)
    
    # 2. Refinement: Greedy trimming with fast permutation counting
    # This is the key improvement - use numpy for efficient counting
    
    # Map permutations to IDs for fast lookup
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {}
    for idx, p in enumerate(all_perms):
        perm_to_id[p] = idx
    
    num_perms = len(all_perms)
    
    # Initialize counts array
    counts = np.zeros(num_perms, dtype=np.int32)
    
    # Initialize window IDs
    # window_ids[s] is the ID of the permutation starting at s, or -1 if not a valid permutation
    num_windows = L - n + 1
    window_ids = np.full(num_windows, -1, dtype=np.int32)
    
    for s in range(num_windows):
        p = tuple(word[s:s + n])
        if p in perm_to_id:
            pid = perm_to_id[p]
            window_ids[s] = pid
            counts[pid] += 1
    
    # Verify we have all permutations
    if np.any(counts == 0):
        # Not a valid superpermutation, return as is
        return "".join(map(str, word))
    
    # Greedy removal: scan left to right, remove if safe
    i = 0
    while i < L:
        if time.time() > deadline:
            break
        
        # Identify affected windows for removal of word[i]
        # Old windows: start indices s such that window contains i
        # s <= i < s + n  =>  i - n + 1 <= s <= i
        # Also s must be in [0, L - n]
        
        s_min = max(0, i - n + 1)
        s_max = min(i, L - n)
        
        # Check if any affected window has count 1
        # If so, removing it would lose that permutation
        safe = True
        for s in range(s_min, s_max + 1):
            pid = window_ids[s]
            if pid != -1:
                if counts[pid] == 1:
                    safe = False
                    break
        
        if not safe:
            i += 1
            continue
        
        # Decrement counts for old windows
        for s in range(s_min, s_max + 1):
            pid = window_ids[s]
            if pid != -1:
                counts[pid] -= 1
        
        # Perform removal
        word.pop(i)
        L -= 1
        
        # Update window_ids for the affected region in the new string
        # New string has length L. Windows start from 0 to L - n.
        # Windows s < i - n + 1 are unchanged.
        # Windows s > i are shifted left: new window_ids[s] = old window_ids[s + 1]
        # Windows in [i - n + 1, i] need recomputation
        
        new_num_windows = L - n + 1
        
        # Shift window_ids: for s in [i + 1, new_num_windows - 1],
        # new window_ids[s] = old window_ids[s + 1]
        if i + 1 < new_num_windows:
            window_ids[i + 1:new_num_windows] = window_ids[i + 2:new_num_windows + 1]
        
        # Recompute windows for s in [i - n + 1, i] (clamped to [0, L - n])
        new_s_min = max(0, i - n + 1)
        new_s_max = min(i, L - n)
        
        for s in range(new_s_min, new_s_max + 1):
            p = tuple(word[s:s + n])
            if p in perm_to_id:
                pid = perm_to_id[p]
                window_ids[s] = pid
                counts[pid] += 1
            else:
                window_ids[s] = -1
        
        # Do not increment i, check the new element at i
    
    return "".join(map(str, word))
