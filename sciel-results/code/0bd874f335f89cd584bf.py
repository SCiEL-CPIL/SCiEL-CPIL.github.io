import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
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
    
    # 2. Greedy Trimming
    # Only attempt trimming if n is small enough to be fast, or if we have time.
    # For n=7, L=5913. Trimming is O(L*n). Very fast.
    # For n=8, L=46233. Trimming is O(L*n). Still fast.
    
    if n >= 4:
        # Initialize permutation counts
        perm_counts = {}
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            if len(set(p)) == n: # Only count valid permutations (distinct elements)
                perm_counts[p] = perm_counts.get(p, 0) + 1
        
        # Check if we have all permutations
        total_perms = 1
        for i in range(2, n+1):
            total_perms *= i
        
        # If we don't have all, something went wrong in construction, but standard construction guarantees it.
        
        # Greedy removal
        i = 0
        while i < len(word):
            if time.time() > deadline:
                break
            
            # Try removing word[i]
            # Identify windows that will be removed
            # Windows starting at max(0, i-n+1) to min(i, len(word)-n)
            start_min = max(0, i - n + 1)
            start_max = min(i, len(word) - n)
            
            removed_perms = []
            for s in range(start_min, start_max + 1):
                p = tuple(word[s:s+n])
                if len(set(p)) == n:
                    removed_perms.append(p)
            
            # Identify windows that will be added/changed
            # After removal, the segment word[i+1:] shifts left.
            # New windows will be formed involving the gap.
            # The windows that were previously "after" i shift left by 1.
            # The windows that were "before" i stay the same?
            # Let's simulate the local change.
            
            # The new word is word[:i] + word[i+1:]
            # The affected indices are in the range [i-n+1, i+n-1] roughly.
            
            # Instead of complex simulation, we can just:
            # 1. Decrement counts for removed_perms
            # 2. Construct the new local word segment
            # 3. Increment counts for new_perms in the affected region
            # 4. Check if any count became 0
            
            # Optimization: Only proceed if no removed perm has count 1.
            # If a perm has count > 1, removing one instance is safe (temporarily).
            
            safe = True
            for p in removed_perms:
                if perm_counts[p] == 1:
                    safe = False
                    break
            
            if not safe:
                i += 1
                continue
            
            # Decrement counts
            for p in removed_perms:
                perm_counts[p] -= 1
            
            # Perform removal
            word.pop(i)
            
            # Calculate new windows in the affected region
            # The region of interest for new windows is around the gap.
            # Indices in new word:
            # The gap is at i.
            # Windows starting from max(0, i-n+1) to min(i, len(word)-n) in the NEW word
            # Note: len(word) decreased by 1.
            
            new_start_min = max(0, i - n + 1)
            new_start_max = min(i, len(word) - n)
            
            added_perms = []
            for s in range(new_start_min, new_start_max + 1):
                p = tuple(word[s:s+n])
                if len(set(p)) == n:
                    added_perms.append(p)
            
            # Increment counts
            for p in added_perms:
                perm_counts[p] = perm_counts.get(p, 0) + 1
            
            # Check validity
            # We only need to check if any perm count became 0.
            # Since we decremented removed_perms and incremented added_perms,
            # and we knew removed_perms had count >= 1,
            # we just need to ensure we didn't drop a perm to 0 that wasn't added back?
            # Actually, if a perm was in removed_perms and had count 1, we skipped.
            # So all removed_perms had count >= 2. After decrement, count >= 1.
            # So no perm becomes 0 from the removal side.
            # The added_perms just increase counts.
            # So the set of present permutations remains complete.
            
            # Thus, the removal is ALWAYS valid if no removed perm had count 1.
            # We do NOT increment i, so we check the new element at index i.
            
        # End greedy trimming
    
    return "".join(map(str, word))
