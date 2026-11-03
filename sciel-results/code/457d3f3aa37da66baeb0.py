import itertools
import random

def construct(n):
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
            # Find max overlap
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Local Search: Delete characters
    # Precompute all valid permutations
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    def get_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in all_perms:
                counts[p] = counts.get(p, 0) + 1
        return counts

    counts = get_counts(word)
    
    # Check if initial is valid
    if len(counts) != len(all_perms):
        return "".join(map(str, word))

    improved = True
    max_passes = 100 # Increase passes to allow for 2-deletions and restarts
    
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        
        # Phase 1: Try single deletions
        # We scan left to right. If we find one, we take it and restart.
        # To improve, we can randomize the scan order or try 2-deletions if stuck.
        
        found_single = False
        found_double = False
        
        # Try singles
        for i in range(len(word)):
            j_min = max(0, i - n + 1)
            j_max = min(i, len(word) - n)
            
            if j_min > j_max:
                continue
            
            valid_deletion = True
            lost_perms = []
            for j in range(j_min, j_max + 1):
                p = tuple(word[j:j+n])
                if p in all_perms:
                    lost_perms.append(p)
                    if counts[p] == 1:
                        valid_deletion = False
                        break
            
            if valid_deletion:
                # Execute deletion
                for p in lost_perms:
                    counts[p] -= 1
                new_word = word[:i] + word[i+1:]
                
                # Update counts for new windows
                j_min_new = max(0, i - n + 1)
                j_max_new = min(i - 1, len(new_word) - n)
                for j in range(j_min_new, j_max_new + 1):
                    p = tuple(new_word[j:j+n])
                    if p in all_perms:
                        counts[p] = counts.get(p, 0) + 1
                        
                word = new_word
                improved = True
                found_single = True
                break # Restart scan
        
        if found_single:
            continue # Next pass
        
        # If no single deletion found, try double deletions
        # Scan for i, i+1
        for i in range(len(word) - 1):
            # Check if deleting i and i+1 is valid
            # Windows affected: those covering i or i+1
            j_min = max(0, i - n + 1)
            j_max = min(i + 1, len(word) - n)
            
            if j_min > j_max:
                continue
            
            valid_deletion = True
            lost_perms = []
            for j in range(j_min, j_max + 1):
                p = tuple(word[j:j+n])
                if p in all_perms:
                    lost_perms.append(p)
                    if counts[p] == 1:
                        valid_deletion = False
                        break
            
            if valid_deletion:
                # Execute double deletion
                for p in lost_perms:
                    counts[p] -= 1
                new_word = word[:i] + word[i+2:]
                
                # Update counts for new windows
                # Junction is at i-1 and i+1 (now adjacent)
                j_min_new = max(0, i - n + 1)
                j_max_new = min(i - 1, len(new_word) - n)
                for j in range(j_min_new, j_max_new + 1):
                    p = tuple(new_word[j:j+n])
                    if p in all_perms:
                        counts[p] = counts.get(p, 0) + 1
                        
                word = new_word
                improved = True
                found_double = True
                break # Restart scan
        
        if found_double:
            continue # Next pass

        # If no single or double deletion found, we are stuck.
        # Try a random swap to escape local minimum?
        # For now, just break.
        break

    return "".join(map(str, word))
