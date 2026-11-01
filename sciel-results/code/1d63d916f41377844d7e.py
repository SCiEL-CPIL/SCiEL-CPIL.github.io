import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

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
        if time.time() > deadline - 5:
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
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            new_word += block[max_overlap:]
        word = new_word

    # Convert to list of integers for efficient manipulation
    arr = [int(c) for c in word]
    L = len(arr)

    if time.time() > deadline - 10:
        return word

    # Phase 2: Greedy Removal to shorten
    from itertools import permutations
    all_perms = set(permutations(range(1, n + 1)))
    
    # Initialize perm_count
    perm_count = {}
    for i in range(L - n + 1):
        p = tuple(arr[i:i + n])
        if p in all_perms:
            perm_count[p] = perm_count.get(p, 0) + 1
            
    # Greedy removal loop
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Iterate backwards to handle index shifts
        for i in range(len(arr) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            # Determine affected windows (destroyed)
            # j in [i-n+1, i]
            lost_candidates = set()
            for j in range(max(0, i - n + 1), min(i, len(arr) - n) + 1):
                p = tuple(arr[j:j + n])
                if p in all_perms:
                    lost_candidates.add(p)
                    
            if not lost_candidates:
                # No permutations lost, safe to remove
                arr.pop(i)
                improved = True
                # Rebuild perm_count
                perm_count = {}
                for j in range(len(arr) - n + 1):
                    p = tuple(arr[j:j + n])
                    if p in all_perms:
                        perm_count[p] = perm_count.get(p, 0) + 1
                continue
            
            # Check if lost candidates are covered elsewhere
            safe = True
            for p in lost_candidates:
                # If count <= 1, it's definitely lost (since the window containing it is destroyed)
                # Wait, if count == 1, the only window is in the affected range, so it is lost.
                if perm_count.get(p, 0) <= 1:
                    safe = False
                    break
                
                # If count > 1, check if any window outside affected range has p
                found = False
                for j_prime in range(len(arr) - n + 1):
                    if j_prime >= i - n + 1 and j_prime <= i:
                        continue
                    if tuple(arr[j_prime:j_prime + n]) == p:
                        found = True
                        break
                if not found:
                    safe = False
                    break
            
            if safe:
                arr.pop(i)
                improved = True
                # Rebuild perm_count
                perm_count = {}
                for j in range(len(arr) - n + 1):
                    p = tuple(arr[j:j + n])
                    if p in all_perms:
                        perm_count[p] = perm_count.get(p, 0) + 1

    return ''.join(str(x) for x in arr)
