import time
import sys
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique (m-1)-permutations in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Maximal overlap with end of new
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Optimization: Remove redundant characters
    # Count permutations
    perm_counts = defaultdict(int)
    # Map permutation tuple to a unique ID or just use tuple as key
    # Tuples of length 7 are fine as keys
    
    for i in range(len(word) - n + 1):
        perm_counts[tuple(word[i:i+n])] += 1
    
    removed = True
    while removed and time.time() < deadline:
        removed = False
        # Scan from left to right
        for i in range(len(word)):
            if time.time() > deadline:
                break
            
            # Try removing word[i]
            # Identify affected windows
            # Windows starting at j such that j <= i <= j+n-1
            # j in [i-n+1, i]
            start_min = max(0, i - n + 1)
            start_max = min(i, len(word) - n)
            
            affected_perms = []
            valid_removal = True
            
            for j in range(start_min, start_max + 1):
                perm = tuple(word[j:j+n])
                if perm_counts[perm] == 1:
                    valid_removal = False
                    break
                affected_perms.append(perm)
            
            if valid_removal:
                # Perform removal
                # Decrement counts for affected perms
                for perm in affected_perms:
                    perm_counts[perm] -= 1
                    if perm_counts[perm] == 0:
                        del perm_counts[perm]
                
                word.pop(i)
                removed = True
                # After removal, indices shift. 
                # We can continue scanning from i (which is now the next element) 
                # or restart. Restarting is safer but slower. 
                # Given few removals, let's just break and restart the while loop 
                # to ensure we don't miss redundancies created earlier.
                # Actually, breaking and restarting the outer while loop 
                # effectively does a full pass again.
                break
                
    return "".join(map(str, word))
