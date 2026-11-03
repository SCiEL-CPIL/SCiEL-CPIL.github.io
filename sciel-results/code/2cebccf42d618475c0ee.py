import time
import random
from itertools import permutations
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Standard construction for n=4: 1234 1324 1432 2314 2431 3124 3412 4132 4312 4231
        # Length is 33.
        # Let's generate it properly.
        pass

    # Generate standard superpermutation
    def standard_superperm(n):
        if n == 1:
            return [1]
        # Use recursive construction
        # S(n) is built from S(n-1)
        # For each permutation p of 1..n-1 in S(n-1), we append n and then p again?
        # No, the standard construction is:
        # Start with S(1) = [1]
        # S(n) is formed by taking S(n-1), and for each permutation p of 1..n-1 that appears as a window in S(n-1) in order of first appearance:
        # append n, then append p.
        
        # Actually, the length is n! + (n-1)! + ... + 1! - (n-1) + 1?
        # Length = sum_{k=1}^n k! - (n-1) + 1? No.
        # Length = n! + (n-1)! + ... + 2! + 1.
        # For n=1: 1.
        # For n=2: 2! + 1 = 3. "121"
        # For n=3: 3! + 2! + 1 = 6+2+1=9. "123121321"
        # For n=4: 4! + 3! + 2! + 1 = 24+6+2+1=33.
        
        word = [1]
        for m in range(2, n + 1):
            perms = []
            seen = set()
            # Find all unique permutations of 1..m-1 in the current word
            for i in range(len(word) - (m - 1) + 1):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1:
                    if window not in seen:
                        seen.add(window)
                        perms.append(window)
            
            new_word = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Max overlap
                overlap = 0
                max_ov = min(len(new_word), len(block))
                for o in range(max_ov, -1, -1):
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            word = new_word
        return word

    word = standard_superperm(n)
    
    L = len(word)
    
    # Precompute permutation to ID mapping
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts and window_ids
    counts = [0] * num_perms
    window_ids = [0] * (L - n + 1)
    
    pid = perm_to_id.get
    
    for j in range(L - n + 1):
        t = tuple(word[j:j+n])
        id_val = pid(t, -1)
        if id_val != -1:
            counts[id_val] += 1
            window_ids[j] = id_val
        else:
            window_ids[j] = -1 # Should not happen for valid superperm

    # Greedy Deletion Loop
    # We will alternate between forward and backward passes
    # And use the optimized window_ids update
    
    def try_delete(i, word, L, n, window_ids, counts, pid, num_perms):
        # Check safety at i
        # Affected old windows: j in [max(0, i-n+1), min(i, L-n)]
        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)
        
        safe = True
        affected_old_ids = []
        
        for j in range(start_min, start_max + 1):
            id_val = window_ids[j]
            if id_val != -1:
                affected_old_ids.append(id_val)
                if counts[id_val] == 1:
                    safe = False
                    break
        
        if not safe:
            return False, L
            
        # Perform deletion
        for id_val in affected_old_ids:
            counts[id_val] -= 1
        
        word.pop(i)
        L -= 1
        
        # Update window_ids
        # New length L
        # New number of windows: L - n + 1
        # Old number of windows: L + 1 - n + 1 = L - n + 2
        
        # 1. Handle seam windows: j in [max(0, i-n+1), min(i-1, L-n)]
        seam_min = max(0, i - n + 1)
        seam_max = min(i - 1, L - n)
        
        # Update these specific windows
        for j in range(seam_min, seam_max + 1):
            # Compute new window ID
            t = tuple(word[j:j+n])
            new_id = pid(t, -1)
            if new_id != -1:
                counts[new_id] += 1
                window_ids[j] = new_id
            else:
                window_ids[j] = -1
        
        # 2. Shift the rest of window_ids
        if i <= L - n:
            window_ids[i : L - n + 1] = window_ids[i + 1 : L - n + 2]
            window_ids.pop()
        else:
            window_ids.pop()
            
        return True, L

    while time.time() < deadline:
        changed = False
        L = len(word)
        
        # Forward pass
        i = 0
        while i < L:
            if time.time() > deadline - 1:
                break
                
            success, new_L = try_delete(i, word, L, n, window_ids, counts, pid, num_perms)
            if success:
                L = new_L
                changed = True
            else:
                i += 1
                
        # If no changes in forward pass, try backward pass
        if not changed:
            i = L - 1
            while i >= 0:
                if time.time() > deadline - 1:
                    break
                    
                success, new_L = try_delete(i, word, L, n, window_ids, counts, pid, num_perms)
                if success:
                    L = new_L
                    changed = True
                else:
                    i -= 1
            
            if not changed:
                # If neither forward nor backward pass changes anything, we are at a local minimum.
                # We can break, as further single-deletion moves won't help.
                break

    return "".join(map(str, word))
