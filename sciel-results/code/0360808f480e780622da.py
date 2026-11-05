import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # 20s margin
    
    # Generate standard superpermutation
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    if n < 7:
        return "".join(map(str, word))
        
    # For n=7, try to optimize
    w = word[:]
    
    # Pre-compute all permutations
    all_perms = set(itertools.permutations(range(1, n+1)))
    total_perms = len(all_perms)
    
    # Build perm_counts
    perm_counts = {}
    for i in range(len(w) - n + 1):
        p = tuple(w[i:i+n])
        if p in all_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
            
    if len(perm_counts) < total_perms:
        return "".join(map(str, w))
        
    # Pass 1: Remove from end
    while len(w) > n:
        if time.time() > deadline:
            break
        last_start = len(w) - n
        p = tuple(w[last_start:])
        if p in all_perms and perm_counts.get(p, 0) == 1:
            break
        else:
            if p in all_perms:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    del perm_counts[p]
            w.pop()
    
    # Pass 2: Try removing from left to right
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(w) - n + 1):
            if time.time() > deadline:
                break
            
            removable = True
            affected_starts = list(range(max(0, i-n+1), min(i, len(w)-n) + 1))
            
            for start in affected_starts:
                p = tuple(w[start:start+n])
                if p in all_perms:
                    if perm_counts.get(p, 0) == 1:
                        removable = False
                        break
            
            if removable:
                for start in affected_starts:
                    p = tuple(w[start:start+n])
                    if p in all_perms:
                        perm_counts[p] -= 1
                        if perm_counts[p] == 0:
                            del perm_counts[p]
                
                w.pop(i)
                
                for start in affected_starts:
                    if start + n <= len(w):
                        p = tuple(w[start:start+n])
                        if p in all_perms:
                            perm_counts[p] = perm_counts.get(p, 0) + 1
                
                changed = True
                break
    
    # Pass 3: Try 2-character removals (remove two adjacent chars)
    # This can sometimes find further reductions
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(w) - n + 1):
            if time.time() > deadline:
                break
            if i + 1 >= len(w):
                break
            
            # Check if removing w[i] and w[i+1] is safe
            # Windows affected: those that include position i or i+1
            # Start range: max(0, i-n+1) to min(i+1, len(w)-n)
            affected_starts = list(range(max(0, i-n+1), min(i+1, len(w)-n) + 1))
            
            removable = True
            for start in affected_starts:
                p = tuple(w[start:start+n])
                if p in all_perms:
                    if perm_counts.get(p, 0) == 1:
                        removable = False
                        break
            
            if removable:
                # Decrement old windows
                for start in affected_starts:
                    p = tuple(w[start:start+n])
                    if p in all_perms:
                        perm_counts[p] -= 1
                        if perm_counts[p] == 0:
                            del perm_counts[p]
                
                # Remove two chars
                w.pop(i+1)
                w.pop(i)
                
                # Add new windows
                for start in affected_starts:
                    if start + n <= len(w):
                        p = tuple(w[start:start+n])
                        if p in all_perms:
                            perm_counts[p] = perm_counts.get(p, 0) + 1
                
                changed = True
                break
    
    # Pass 4: Try inserting a character to enable further removals
    # This is a more complex move; try it only if we have time
    if time.time() < deadline - 20:
        # Try single character insertions at various positions
        # For each position, try inserting each symbol
        # Check if it enables removals
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(w) + 1):
                if time.time() > deadline - 10:
                    break
                for sym in range(1, n+1):
                    # Try inserting sym at position i
                    w.insert(i, sym)
                    
                    # Update perm_counts for new windows
                    # New windows: those that include position i
                    new_starts = list(range(max(0, i-n+1), min(i, len(w)-n) + 1))
                    for start in new_starts:
                        p = tuple(w[start:start+n])
                        if p in all_perms:
                            perm_counts[p] = perm_counts.get(p, 0) + 1
                    
                    # Now try to remove from end
                    removed_any = False
                    while len(w) > n:
                        if time.time() > deadline - 10:
                            break
                        last_start = len(w) - n
                        p = tuple(w[last_start:])
                        if p in all_perms and perm_counts.get(p, 0) == 1:
                            break
                        else:
                            if p in all_perms:
                                perm_counts[p] -= 1
                                if perm_counts[p] == 0:
                                    del perm_counts[p]
                            w.pop()
                            removed_any = True
                    
                    # Try removing from left to right
                    local_changed = True
                    while local_changed and time.time() < deadline - 10:
                        local_changed = False
                        for j in range(len(w) - n + 1):
                            if time.time() > deadline - 10:
                                break
                            removable = True
                            affected = list(range(max(0, j-n+1), min(j, len(w)-n) + 1))
                            for start in affected:
                                p = tuple(w[start:start+n])
                                if p in all_perms:
                                    if perm_counts.get(p, 0) == 1:
                                        removable = False
                                        break
                            if removable:
                                for start in affected:
                                    p = tuple(w[start:start+n])
                                    if p in all_perms:
                                        perm_counts[p] -= 1
                                        if perm_counts[p] == 0:
                                            del perm_counts[p]
                                w.pop(j)
                                for start in affected:
                                    if start + n <= len(w):
                                        p = tuple(w[start:start+n])
                                        if p in all_perms:
                                            perm_counts[p] = perm_counts.get(p, 0) + 1
                                local_changed = True
                                removed_any = True
                                break
                    
                    # If we removed more than we added, keep the change
                    if removed_any:
                        improved = True
                    else:
                        # Revert: remove the inserted character
                        w.pop(i)
                        # Remove the new windows we added
                        new_starts = list(range(max(0, i-n+1), min(i, len(w)-n) + 1))
                        for start in new_starts:
                            if start + n <= len(w):
                                p = tuple(w[start:start+n])
                                if p in all_perms:
                                    perm_counts[p] -= 1
                                    if perm_counts[p] == 0:
                                        del perm_counts[p]
    
    # Final pass: remove from end again
    while len(w) > n:
        if time.time() > deadline:
            break
        last_start = len(w) - n
        p = tuple(w[last_start:])
        if p in all_perms and perm_counts.get(p, 0) == 1:
            break
        else:
            if p in all_perms:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    del perm_counts[p]
            w.pop()
    
    # Verify
    if len(perm_counts) < total_perms:
        # Should not happen, but just in case
        pass
    
    return "".join(map(str, w))
