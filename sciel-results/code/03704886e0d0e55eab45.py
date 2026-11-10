import itertools
import time

def construct(n):
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
            # Find overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # If n is small, standard is optimal
    if n < 7:
        return "".join(map(str, word))
    
    if n > 7:
        # For n > 7, standard construction is a safe fallback
        # Local search might be too slow or complex
        return "".join(map(str, word))
    
    # n == 7 case
    # Precompute permutation IDs
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(perms)}
    num_perms = len(perms)
    
    counts = [0] * num_perms
    essential = set()
    
    # Helper to update counts for a window
    def add_window(w):
        pid = perm_to_id.get(w)
        if pid is not None:
            counts[pid] += 1
            if counts[pid] == 1:
                essential.add(pid)
                
    def remove_window(w):
        pid = perm_to_id.get(w)
        if pid is not None:
            counts[pid] -= 1
            if counts[pid] == 1:
                essential.add(pid)
            elif counts[pid] == 0:
                essential.discard(pid)
            else:
                # count > 1, so it was not essential, and remains not essential
                # unless it was 2 and becomes 1?
                # If counts[pid] was 2, essential did NOT contain it.
                # Now counts[pid] is 1. It SHOULD be in essential.
                # So:
                pass 
            # Correct logic:
            # Before: count = C
            # If C == 1: essential contains pid.
            # If C > 1: essential does not contain pid.
            # After: count = C - 1
            # If C - 1 == 1 (i.e., C == 2): essential should ADD pid.
            # If C - 1 == 0 (i.e., C == 1): essential should REMOVE pid. (But this means invalid state if we didn't check)
            # If C - 1 > 1: no change in essential status.
            
            # Let's rewrite remove_window properly
            pass

    # Let's just implement the update logic inline or in a cleaner helper
    def update_remove(w):
        pid = perm_to_id.get(w)
        if pid is not None:
            old_count = counts[pid]
            counts[pid] = old_count - 1
            if old_count == 1:
                # Becomes 0. Should not happen if we checked validity.
                # But if it does, essential had it, now it's gone.
                essential.discard(pid)
            elif old_count == 2:
                # Becomes 1. It was not essential, now it is.
                essential.add(pid)
            # else: old_count > 2, becomes > 1. No change.

    def update_add(w):
        pid = perm_to_id.get(w)
        if pid is not None:
            old_count = counts[pid]
            counts[pid] = old_count + 1
            if old_count == 1:
                # Becomes 2. It was essential, now it is not.
                essential.discard(pid)
            # else: old_count == 0 (becomes 1, add to essential) or > 1 (no change)
            if old_count == 0:
                essential.add(pid)

    # Initialize counts
    L = len(word)
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        update_add(w)
        
    # Local Search
    start_time = time.time()
    timeout = 120.0
    
    # We will try to delete characters
    # We do passes until no more deletions are found or timeout
    while True:
        if time.time() - start_time > timeout:
            break
            
        found_deletion = False
        i = 0
        while i < len(word):
            if time.time() - start_time > timeout:
                break
                
            # Check if word[i] can be deleted
            # Windows containing i: start s from max(0, i-n+1) to min(i, L-n)
            valid = True
            windows_to_remove = []
            for s in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
                w = tuple(word[s:s+n])
                pid = perm_to_id.get(w)
                if pid is not None:
                    if counts[pid] == 1:
                        valid = False
                        break
                    windows_to_remove.append(w)
            
            if valid:
                # Perform deletion
                # 1. Remove old windows
                for w in windows_to_remove:
                    update_remove(w)
                
                # 2. Delete word[i]
                word.pop(i)
                
                # 3. Add new windows
                # New length L' = L - 1
                # Junction is at index i-1 in new string (if i>0)
                # New windows are those that cover the junction
                # Starts s_new from max(0, i-n) to min(i-1, L'-n)
                # Note: if i==0, junction is at start? No, if i==0, we removed first char.
                # New windows are just the first n-1 windows?
                # Actually, if we remove index 0, the new windows are the old windows shifted?
                # No, the new windows are formed by the new adjacency.
                # If i=0, the new first window is word[0:n] (which was word[1:n+1] before? No).
                # Let's just re-evaluate windows that *could* be new.
                # A window is new if it contains the new adjacency at (i-1, i) in the new string.
                # If i==0, there is no left neighbor. The "junction" is the start.
                # The windows starting at 0 in the new string are new.
                # Specifically, windows starting at s such that s <= i-1 and s+n-1 >= i-1.
                # i.e., s in [max(0, i-n+1), i-1] ?
                # Let's check indices.
                # Old: ... A B C ... (B is at i)
                # New: ... A C ...
                # New adjacency is A-C.
                # Windows containing A-C:
                # If A is at i-1 in new string, C is at i.
                # Windows starting at s such that s <= i-1 and s+n-1 >= i.
                # s <= i-1
                # s >= i - n + 1
                # Also s must be valid start: 0 <= s <= L'-n.
                
                L_new = len(word)
                if L_new >= n:
                    start_range = max(0, i - n + 1)
                    end_range = min(i - 1, L_new - n)
                    if i == 0:
                        # If we removed the first element, the "junction" is effectively the start.
                        # The windows starting at 0 are new.
                        # start_range = 0
                        # end_range = min(-1, ...) -> -1. Loop won't run?
                        # But the window starting at 0 IS new.
                        # If i=0, the new window is word[0:n].
                        # This window was NOT in the old string (old string had word[0:n] which included B).
                        # So we need to add it.
                        # The condition s <= i-1 fails for i=0 (s <= -1).
                        # So we handle i=0 separately or adjust range.
                        # If i=0, the only new window is at s=0?
                        # Actually, if i=0, the windows starting at 0, 1, ..., n-1 are affected?
                        # No, only the window starting at 0 changes its content?
                        # Old window at 0: word[0:n] (includes B)
                        # New window at 0: word[0:n] (does not include B, starts with C)
                        # Old window at 1: word[1:n+1] (includes B)
                        # New window at 1: word[1:n+1] (does not include B)
                        # Wait, the windows that *didn't* include B are unchanged.
                        # The windows that *did* include B are removed.
                        # The *only* new windows are those that bridge the gap.
                        # If i=0, the gap is at the beginning.
                        # The new window at 0 bridges the "start" and the rest.
                        # It is a new window.
                        # Are there others?
                        # The window at 1 in new string is old window at 2? No.
                        # New string: C D E ...
                        # Old string: B C D E ...
                        # New window at 0: C D E ...
                        # This is new.
                        # New window at 1: D E F ...
                        # This was old window at 2? No, old window at 2 was D E F... (if B was at 0, C at 1, D at 2).
                        # Old window at 2: D E F...
                        # So new window at 1 is same as old window at 2.
                        # So only the window at 0 is new?
                        # Let's check.
                        # Old windows containing B (at 0): s=0.
                        # Removed: old window 0.
                        # New windows:
                        # The new string is shorter.
                        # The window at 0 in new string is new.
                        # The window at 1 in new string corresponds to old window at 2.
                        # So only s=0 is new.
                        
                        # General formula:
                        # New windows start at s such that the window covers the junction.
                        # Junction is between index i-1 and i in new string.
                        # If i=0, there is no index -1.
                        # The "junction" is the start.
                        # The window starting at 0 covers the start.
                        # So s=0.
                        
                        # Let's use the range:
                        # s from max(0, i-n+1) to min(i, L_new-n) ??
                        # If i=0: max(0, -n+1)=0. min(0, L_new-n)=0.
                        # So s=0. Correct.
                        
                        # If i>0:
                        # Junction at i-1, i.
                        # s from max(0, i-n+1) to min(i-1, L_new-n).
                        # Wait, if i=1. Junction at 0, 1.
                        # s from max(0, 1-n+1) to min(0, L_new-n).
                        # s=0.
                        # Window at 0 covers 0 and 1. Correct.
                        
                        # So the range is:
                        # start = max(0, i - n + 1)
                        # end = min(i - 1, L_new - n)
                        # BUT if i == 0, end should be 0?
                        # If i=0, min(-1, ...) is -1. Loop doesn't run.
                        # So we need to handle i=0 specially or adjust end.
                        # If i=0, the new window is at 0.
                        # So end should be max(end, 0) if i==0?
                        # Actually, if i=0, the window at 0 is new.
                        # My previous logic: "s <= i-1" fails for i=0.
                        # But the window at 0 *does* cover the junction (start).
                        # So for i=0, we add window at 0.
                        
                        pass
                    
                    if i == 0:
                        # Add window at 0
                        if 0 <= L_new - n:
                            w_new = tuple(word[0:n])
                            update_add(w_new)
                    else:
                        start_s = max(0, i - n + 1)
                        end_s = min(i - 1, L_new - n)
                        for s in range(start_s, end_s + 1):
                            w_new = tuple(word[s:s+n])
                            update_add(w_new)
                            
                found_deletion = True
                # Do not increment i, check same position in new string
            else:
                i += 1
                
        if not found_deletion:
            break

    return "".join(map(str, word))
