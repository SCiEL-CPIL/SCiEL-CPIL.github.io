import time
import itertools
from collections import Counter

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # 1. Generate classic superpermutation
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
            # Overlap logic: find max overlap between end of new and start of block
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new)-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if n <= 4:
        return "".join(map(str, word))

    # Precompute all permutations of 1..n
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts of each permutation in the current word
    perm_counts = {p: 0 for p in all_perms}
    
    # Initial count pass
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if w in perm_counts:
            perm_counts[w] += 1
            
    # Greedy deletion with single-pass optimization
    # We scan left to right. If we delete, we stay at the same index to check the new element.
    # If we don't delete, we move to the next.
    # We repeat passes until no changes or time limit.
    
    changed = True
    while changed and (time.time() < deadline):
        changed = False
        L = len(word)
        # We scan indices 0 to L-1
        # Note: L changes if we delete
        i = 0
        while i < len(word):
            if time.time() > deadline:
                return "".join(map(str, word))
            
            # Determine which windows are affected by removing position i
            # Windows starting at s include position i if s <= i <= s+n-1
            # => i-n+1 <= s <= i
            # Also s must be valid: 0 <= s <= len(word) - n
            
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            
            # Check if any of these windows is a permutation with count 1
            safe = True
            for s in range(s_min, s_max + 1):
                w = tuple(word[s:s+n])
                if w in perm_counts and perm_counts[w] == 1:
                    safe = False
                    break
            
            if safe:
                # Perform deletion
                # 1. Decrement counts of windows that are removed
                for s in range(s_min, s_max + 1):
                    w = tuple(word[s:s+n])
                    if w in perm_counts:
                        perm_counts[w] -= 1
                
                # 2. Remove the element
                word.pop(i)
                
                # 3. Increment counts of new windows formed by the shift
                # The deletion happens at index i.
                # New word length L' = L - 1.
                # Windows starting at s in new word:
                # If s < i: same as old word (but shifted? No, indices < i are unchanged).
                # If s >= i: this window is formed by elements from old word starting at s, 
                # except the element at i is gone.
                # The "new" windows are those starting at s in [i, i+n-1] in the new word?
                # Actually, windows starting at s < i are identical to before.
                # Windows starting at s >= i in the new word are formed by old word[s:i] + old word[i+1:s+n+1].
                # We need to increment the count for these new windows.
                
                # Range of new windows to add:
                # The windows that *changed* are those that now cover the gap at i.
                # In the new word, the gap is gone.
                # The windows affected are those starting at s such that the window [s, s+n-1] 
                # in the new word includes the position i (which is now occupied by old i+1).
                # Actually, it's easier to say:
                # Windows starting at s in [max(0, i-n+1), min(i, len(word)-n)] in the NEW word?
                # No.
                # Let's trace:
                # Old word: ... A B C D ... (B is at i)
                # New word: ... A C D ... (C is at i)
                # Windows in Old word containing B:
                #   s such that s <= i <= s+n-1.
                # Windows in New word containing C (at i):
                #   s such that s <= i <= s+n-1.
                # But the content is different.
                # The windows that are *new* (or changed) are those that include the position i in the new word.
                # These are s in [i-n+1, i].
                # However, we also have to consider that the window at s=i in the new word 
                # is different from the window at s=i in the old word?
                # Old window at i: word[i:i+n] = B...
                # New window at i: word[i:i+n] = C...
                # So yes, windows starting at s in [i-n+1, i] are changed.
                # Also, the window starting at s=i-1 in the new word?
                # Old window at i-1: word[i-1:i-1+n] = A B ...
                # New window at i-1: word[i-1:i-1+n] = A C ...
                # So yes, s in [i-n+1, i] covers all changed windows.
                
                new_s_min = max(0, i - n + 1)
                new_s_max = min(i, len(word) - n)
                
                if new_s_min <= new_s_max:
                    for s in range(new_s_min, new_s_max + 1):
                        w = tuple(word[s:s+n])
                        if w in perm_counts:
                            perm_counts[w] += 1
                
                changed = True
                # Do NOT increment i, so we check the new element at i again
            else:
                i += 1

    return "".join(map(str, word))
