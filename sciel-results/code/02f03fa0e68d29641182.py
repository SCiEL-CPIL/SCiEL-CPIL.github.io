import time
import itertools
from collections import Counter

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Base cases
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431213243121321421324132141231421324123214321" # Standard S4

    def standard_superperm(k):
        if k == 1:
            return [1]
        if k == 2:
            return [1, 2, 1]
        if k == 3:
            return [1, 2, 3, 1, 2, 1, 3, 2, 1]
        
        word = standard_superperm(k-1)
        perms, seen = [], set()
        for i in range(len(word) - k + 2):
            window = tuple(word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [k] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new

    # Generate standard superpermutation for n
    W = standard_superperm(n)
    
    # Convert to list of ints for easier manipulation
    W = list(W)
    
    # Precompute permutation counts
    # Note: W contains integers 1..n
    perm_count = Counter()
    for j in range(len(W) - n + 1):
        P = tuple(W[j:j+n])
        perm_count[P] += 1
        
    # Greedy removal
    # We try to remove characters one by one.
    # We scan from left to right. If a character is removable, we remove it and restart the scan.
    # This ensures we always work with the current state of the word.
    
    max_removals = 20 # Safety cap, expected ~8
    removals = 0
    
    while removals < max_removals and time.time() < deadline:
        found_removal = False
        current_len = len(W)
        
        for i in range(current_len):
            if time.time() > deadline:
                break
            
            # Check if W[i] is removable
            # Windows covering i start at j such that j <= i < j+n
            # => i-n+1 <= j <= i
            start_j = max(0, i - n + 1)
            end_j = min(i, current_len - n)
            
            removable = True
            for j in range(start_j, end_j + 1):
                P = tuple(W[j:j+n])
                if perm_count[P] == 1:
                    removable = False
                    break
            
            if removable:
                # Perform removal
                # Update perm_count for the windows being removed
                for j in range(start_j, end_j + 1):
                    P = tuple(W[j:j+n])
                    perm_count[P] -= 1
                    if perm_count[P] == 0:
                        del perm_count[P]
                
                # Remove character
                del W[i]
                removals += 1
                found_removal = True
                break # Restart scan from 0
        
        if not found_removal:
            break
            
    return "".join(map(str, W))
