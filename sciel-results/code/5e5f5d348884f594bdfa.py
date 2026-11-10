import time
import math
import itertools

def construct(n):
    start_time = time.time()
    # Safety margin: return before 120s
    deadline = 115.0
    
    # 1. Generate the classic superpermutation
    word = generate_classic(n)
    
    # Precompute the number of permutations
    target_count = math.factorial(n)
    
    # Build permutation index mapping for fast lookup
    perm_to_idx = {}
    for p in itertools.permutations(range(1, n + 1)):
        perm_to_idx[p] = len(perm_to_idx)
    
    # Build initial count array
    L = len(word)
    counts = [0] * target_count
    distinct_count = 0
    for s in range(L - n + 1):
        w = tuple(word[s:s+n])
        if w in perm_to_idx:
            idx = perm_to_idx[w]
            if counts[idx] == 0:
                distinct_count += 1
            counts[idx] += 1
    
    # 2. Greedy Reduction with Incremental Validity Checking
    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(word)
        for i in range(L):
            if time.time() > deadline:
                break
            
            # Try removing position i
            # Remove old windows containing position i
            removed = []
            s_start = max(0, i - n + 1)
            s_end = min(i, L - n)
            for s in range(s_start, s_end + 1):
                w = tuple(word[s:s+n])
                if w in perm_to_idx:
                    idx = perm_to_idx[w]
                    counts[idx] -= 1
                    if counts[idx] == 0:
                        distinct_count -= 1
                    removed.append((s, idx))
            
            # Add new windows spanning the gap
            added = []
            ns_start = max(0, i - n + 1)
            ns_end = min(i - 1, L - 1 - n)
            for s_new in range(ns_start, ns_end + 1):
                if s_new + n > i:
                    new_w = tuple(word[s_new:i] + word[i+1:s_new+n+1])
                    if new_w in perm_to_idx:
                        idx = perm_to_idx[new_w]
                        if counts[idx] == 0:
                            distinct_count += 1
                        counts[idx] += 1
                        added.append((s_new, idx))
            
            if distinct_count == target_count:
                # Valid! Commit removal
                word = word[:i] + word[i+1:]
                improved = True
                break  # Restart scan
            else:
                # Revert changes
                for (s, idx) in removed:
                    counts[idx] += 1
                    if counts[idx] == 1:
                        distinct_count += 1
                for (s_new, idx) in added:
                    counts[idx] -= 1
                    if counts[idx] == 0:
                        distinct_count -= 1
    
    # 3. Second phase: try removing pairs of adjacent characters
    if time.time() < deadline:
        pair_improved = True
        while pair_improved and time.time() < deadline:
            pair_improved = False
            L = len(word)
            for i in range(L - 1):
                if time.time() > deadline:
                    break
                
                # Try removing positions i and i+1
                # Remove old windows containing position i or i+1
                removed = []
                s_start = max(0, i - n + 1)
                s_end = min(i + 1, L - n)
                for s in range(s_start, s_end + 1):
                    w = tuple(word[s:s+n])
                    if w in perm_to_idx:
                        idx = perm_to_idx[w]
                        counts[idx] -= 1
                        if counts[idx] == 0:
                            distinct_count -= 1
                        removed.append((s, idx))
                
                # Add new windows spanning the gap (gap of 2)
                added = []
                # New word has length L-2. Gap is between new pos i-1 and new pos i+1
                # New windows spanning: s_new < i and s_new + n > i+1 (in new coords)
                # i.e., s_new <= i-1 and s_new + n >= i+2, so s_new >= i+2-n
                ns_start = max(0, i + 2 - n)
                ns_end = min(i - 1, L - 2 - n)
                for s_new in range(ns_start, ns_end + 1):
                    if s_new + n > i + 1:
                        new_w = tuple(word[s_new:i] + word[i+2:s_new+n+2])
                        if new_w in perm_to_idx:
                            idx = perm_to_idx[new_w]
                            if counts[idx] == 0:
                                distinct_count += 1
                            counts[idx] += 1
                            added.append((s_new, idx))
                
                if distinct_count == target_count:
                    word = word[:i] + word[i+2:]
                    pair_improved = True
                    break
                else:
                    for (s, idx) in removed:
                        counts[idx] += 1
                        if counts[idx] == 1:
                            distinct_count += 1
                    for (s_new, idx) in added:
                        counts[idx] -= 1
                        if counts[idx] == 0:
                            distinct_count -= 1
    
    return "".join(map(str, word))

def generate_classic(n):
    """
    Classic recursive construction of a superpermutation.
    Length is 1! + 2! + ... + n!
    """
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word
