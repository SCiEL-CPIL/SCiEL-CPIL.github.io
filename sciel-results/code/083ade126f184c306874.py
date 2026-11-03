import time
import sys
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Classic Recursive Construction
    # Length is 1! + 2! + ... + n!
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        # in order of appearance
        perms = []
        seen = set()
        # word length is L. windows of size m-1
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation (distinct elements)
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Block: p[0]...p[m-2], m, p[0]...p[m-2]
            block = list(p) + [m] + list(p)
            
            # Find max overlap with new_word
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        # Time check
        if time.time() > deadline:
            break

    # For n <= 4, the recursive construction is optimal.
    # For n >= 5, we can try to shorten it.
    # The recursive construction for n=7 has length 5913.
    # The optimal is 5905. We can remove 8 characters.
    
    if n >= 5 and time.time() < deadline - 10.0:
        
        # Map permutations to IDs for O(1) count access
        all_perms = list(itertools.permutations(range(1, n+1)))
        perm_to_id = {p: i for i, p in enumerate(all_perms)}
        num_perms = len(all_perms)
        
        # Build initial counts
        counts = [0] * num_perms
        L = len(word)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            pid = perm_to_id.get(p, -1)
            if pid != -1:
                counts[pid] += 1
        
        # Verify validity
        if sum(1 for c in counts if c > 0) < num_perms:
             return "".join(map(str, word))

        # Local Search: Greedy Deletion
        current_word = word
        w = current_word  # local ref for speed
        
        def try_delete(i):
            """
            Attempts to delete current_word[i].
            Returns True if successful.
            """
            L = len(w)
            if L <= n:
                return False
            
            # 1. Identify old windows containing i
            k_start_old = max(0, i - n + 1)
            k_end_old = min(i, L - n)
            
            old_pids = []
            for k in range(k_start_old, k_end_old + 1):
                p = tuple(w[k:k+n])
                pid = perm_to_id.get(p, -1)
                if pid != -1:
                    old_pids.append(pid)
                    counts[pid] -= 1
            
            # 2. Identify new windows formed
            k_start_new = max(0, i - n + 1)
            k_end_new = min(i - 1, L - n - 1)
            
            new_pids = []
            for k in range(k_start_new, k_end_new + 1):
                # New window: w[k:i] + w[i+1:k+n+1]
                p = tuple(w[k:i] + w[i+1:k+n+1])
                pid = perm_to_id.get(p, -1)
                if pid != -1:
                    new_pids.append(pid)
                    counts[pid] += 1
            
            # 3. Check validity: no old perm drops to 0
            valid = True
            for pid in old_pids:
                if counts[pid] == 0:
                    valid = False
                    break
            
            if valid:
                del w[i]
                return True
            else:
                # Revert
                for pid in old_pids:
                    counts[pid] += 1
                for pid in new_pids:
                    counts[pid] -= 1
                return False

        # Multiple deletion passes until no more deletions or time runs out
        passes = 0
        while time.time() < deadline - 5.0:
            passes += 1
            i = 0
            made_deletion = False
            while i < len(w):
                if time.time() > deadline - 5.0:
                    break
                
                if try_delete(i):
                    i = max(0, i - 1)
                    made_deletion = True
                else:
                    i += 1
                
                if time.time() > deadline - 5.0:
                    break
            
            if not made_deletion:
                break

    if n >= 5:
        return "".join(map(str, current_word))
    else:
        return "".join(map(str, word))
