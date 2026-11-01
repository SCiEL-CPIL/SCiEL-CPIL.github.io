import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431214321"

    # Phase 1: Build standard recursive superpermutation
    word = "1"
    for m in range(2, n + 1):
        if time.time() > deadline - 20:
            break
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = ""
        for p in perms:
            block = p + str(m) + p
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word.endswith(block[:o]):
                    max_overlap = o
                    break
            new_word += block[max_overlap:]
        word = new_word

    # Convert to list of integers
    arr = [int(c) for c in word]
    
    if time.time() > deadline - 10:
        return word

    # Phase 2: Optimization
    all_perms = set(permutations(range(1, n + 1)))
    
    def compute_counts(a):
        counts = {}
        for i in range(len(a) - n + 1):
            p = tuple(a[i:i + n])
            if p in all_perms:
                counts[p] = counts.get(p, 0) + 1
        return counts

    perm_count = compute_counts(arr)
    
    if len(perm_count) < len(all_perms):
        return word

    best_arr = arr[:]
    best_len = len(arr)
    
    # Random search for removals
    for step in range(50000):
        if time.time() > deadline - 5:
            break
        if len(arr) <= n:
            break
            
        i = random.randint(0, len(arr) - 1)
        
        start_j = max(0, i - n + 1)
        end_j = min(i, len(arr) - n)
        
        local_counts = {}
        for j in range(start_j, end_j + 1):
            p = tuple(arr[j:j + n])
            if p in all_perms:
                local_counts[p] = local_counts.get(p, 0) + 1
        
        valid = True
        for p, lc in local_counts.items():
            if perm_count.get(p, 0) == lc:
                valid = False
                break
        
        if valid:
            arr.pop(i)
            perm_count = compute_counts(arr)
            
            if len(arr) < best_len:
                best_len = len(arr)
                best_arr = arr[:]
                
    return ''.join(str(x) for x in best_arr)
