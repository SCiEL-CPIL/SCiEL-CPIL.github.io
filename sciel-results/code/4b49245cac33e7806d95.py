import time
from itertools import permutations

def construct(n):
    deadline = time.time() + 140  # 10s safety margin

    # Classic recursive construction of length 1! + 2! + ... + n!
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
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    if n <= 6:
        return "".join(map(str, word))

    # For n >= 7, apply local search to shorten the word
    all_perms = set(permutations(range(1, n + 1)))
    current = word[:]
    
    # Precompute initial windows
    seen = set()
    for i in range(len(current) - n + 1):
        seen.add(tuple(current[i:i + n]))
    
    # Verify classic construction is valid
    if not all_perms.issubset(seen):
        return "".join(map(str, current))

    # Fast local search: try deleting each character
    # We iterate multiple times until no more deletions are possible or time runs out
    improved = True
    n_local = n
    all_perms_local = all_perms
    start_time = time.time()
    
    while improved and time.time() - start_time < 130:  # Safety margin
        improved = False
        L = len(current)
        # Try deleting from left to right
        for i in range(L):
            if (i % 1000 == 0) and time.time() - start_time > 130:
                break
            
            # Remove current[i]
            saved = current.pop(i)
            
            # Rebuild seen set for the new word
            # Optimization: Use a list comprehension or generator if possible?
            # set comprehension is faster than loop with add
            new_L = L - 1
            if new_L < n_local:
                current.insert(i, saved)
                continue
                
            # Check if all perms are still present
            # We can break early if we find a missing perm?
            # all_perms.issubset(new_seen) is fast, but building new_seen is the cost.
            
            # Alternative: Check if any perm in all_perms is missing?
            # That's O(5040 * new_L) which is worse.
            
            # Just build the set.
            new_seen = set(tuple(current[j:j + n_local]) for j in range(new_L - n_local + 1))
            
            if all_perms_local.issubset(new_seen):
                improved = True
                # seen = new_seen # Not strictly needed for next pass if we rebuild, but good for tracking
                # Note: 'current' is already modified (popped)
                L = new_L
            else:
                current.insert(i, saved)
                # L remains unchanged

    return "".join(map(str, current))
