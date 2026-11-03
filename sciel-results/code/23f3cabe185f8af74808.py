import time
import numpy as np
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin for 150s limit

    # 1. Generate standard superpermutation
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
            # Find max overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    L = len(word)
    
    # 2. Greedy Trimming with fast permutation counting
    if n >= 4:
        # Map permutations to IDs
        # For n=7, 5040 perms.
        # We can use a dictionary to map tuple -> id
        perm_to_id = {}
        count = 0
        # We only care about permutations of 1..n
        # Pre-generate all permutations of 1..n?
        # For n=7, 5040 is small.
        all_perms = list(permutations(range(1, n+1)))
        for p in all_perms:
            perm_to_id[p] = count
            count += 1
            
        # Initialize counts array
        counts = np.zeros(count, dtype=np.int32)
        
        # Initialize window IDs
        # window_ids[s] is the ID of the permutation starting at s
        # If the window is not a valid permutation (repeats), ID = -1
        window_ids = np.full(L - n + 1, -1, dtype=np.int32)
        
        for s in range(L - n + 1):
            p = tuple(word[s:s+n])
            if p in perm_to_id:
                pid = perm_to_id[p]
                window_ids[s] = pid
                counts[pid] += 1
        
        # Greedy removal
        i = 0
        while i < L:
            if time.time() > deadline:
                break
            
            # Identify affected windows for removal of word[i]
            # Old windows: start indices s such that window contains i
            # s <= i < s+n  =>  i-n+1 <= s <= i
            # Also s must be in [0, L-n]
            
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
            # The new string has length L.
            # Windows start from 0 to L-n.
            
            # Shift window_ids for s > i
            # window_ids[i+1 : L-n+1] = window_ids[i+2 : L-n+2]
            if i + 1 < L - n + 1:
                window_ids[i+1 : L-n+1] = window_ids[i+2 : L-n+2]
            
            # Now compute the new windows for s in [i-n+1, i] (clamped to [0, L-n])
            new_s_min = max(0, i - n + 1)
            new_s_max = min(i, L - n)
            
            for s in range(new_s_min, new_s_max + 1):
                p = tuple(word[s:s+n])
                if p in perm_to_id:
                    pid = perm_to_id[p]
                    window_ids[s] = pid
                    counts[pid] += 1
                else:
                    window_ids[s] = -1
            
            # We do not increment i, so we check the new element at i
            
    return "".join(map(str, word))
