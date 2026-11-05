import time
import sys
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # 1. Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        # Time check
        if time.time() > deadline:
            # Fast path without optimization
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_check = min(len(new), len(block))
                overlap = 0
                for o in range(max_check, 0, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
            continue

        perms, seen = [], set()
        # Extract unique (m-1)-length windows with distinct symbols
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build blocks
        blocks = [list(p) + [m] + list(p) for p in perms]
        
        # Standard ordering is usually fine, but we can try to keep the current program's greedy
        # if it doesn't cost too much. For n=7, the number of blocks is 720.
        # Greedy O(N^2) is 720^2 ~ 500k ops, which is fast.
        if m <= 5 and len(blocks) > 2:
             # Simple greedy to try to improve overlap
             ordered_blocks = blocks
             # The current program has a _greedy_order function. 
             # I will implement a simple version inline to avoid dependency on potentially buggy external helpers 
             # or just use standard order which is known to be optimal for the recursive structure 
             # in terms of the theoretical bound, but local swaps might help.
             # Actually, the standard recursive construction is the one that achieves the bound.
             # Reordering blocks arbitrarily might break the property that all perms are included?
             # No, as long as we include all blocks, all perms are included.
             # The overlap determines the length.
             # So greedy ordering is safe.
             
             # Let's implement a simple greedy reorder
             remaining = list(range(len(blocks)))
             ordered_indices = [0]
             remaining.remove(0)
             while remaining:
                 last_block = blocks[ordered_indices[-1]]
                 best_idx = -1
                 best_overlap = -1
                 for i in remaining:
                     b = blocks[i]
                     max_ov = min(len(last_block), len(b))
                     for o in range(max_ov, 0, -1):
                         if last_block[-o:] == b[:o]:
                             if o > best_overlap:
                                 best_overlap = o
                                 best_idx = i
                             break
                 if best_idx == -1:
                     ordered_indices.extend(remaining)
                     break
                 ordered_indices.append(best_idx)
                 remaining.remove(best_idx)
             ordered_blocks = [blocks[i] for i in ordered_indices]
        else:
            ordered_blocks = blocks
            
        new = []
        for block in ordered_blocks:
            max_check = min(len(new), len(block))
            overlap = 0
            for o in range(max_check, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    # 2. Local Optimization: Remove redundant characters
    # We iterate through the string and try to remove each character.
    # If the resulting string is still a superpermutation, we keep the removal and restart.
    
    total_perms = int(math.factorial(n)) if n < 20 else 1 # n is small
    # Actually, for n=7, 5040.
    
    # Helper to check validity
    def is_valid_superperm(w, n, total_perms):
        if len(w) < n:
            return False
        perms = set()
        for i in range(len(w) - n + 1):
            perms.add(tuple(w[i:i+n]))
            if len(perms) == total_perms:
                return True
        return len(perms) == total_perms

    # Optimization loop
    # We use a list for w for efficient slicing? Slicing lists is O(N).
    # String slicing is also O(N).
    # Let's use a list.
    w = word
    
    # Initial check
    if not is_valid_superperm(w, n, total_perms):
        return "".join(map(str, w))

    # Pre-compute the set of all permutations to speed up check?
    # No, we just need to know if the set of windows is size total_perms.
    
    while True:
        if time.time() > deadline:
            break
            
        removed = False
        # Try to remove from left to right
        for i in range(len(w)):
            if time.time() > deadline:
                break
            
            # Skip if removing would make it too short
            if len(w) - 1 < n:
                break
                
            w_test = w[:i] + w[i+1:]
            
            # Check validity
            if is_valid_superperm(w_test, n, total_perms):
                w = w_test
                removed = True
                break # Restart from beginning
        
        if not removed:
            break

    return "".join(map(str, w))

# Need to import math
import math
