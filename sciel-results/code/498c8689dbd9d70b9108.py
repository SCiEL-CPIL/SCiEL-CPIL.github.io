import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
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
            # Find max overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Optimization via greedy deletion
    # Convert word to list of ints for easier manipulation
    # We represent permutations as integers for fast counting
    # Encoding: base n. p[0]*n^(n-1) + ... + p[n-1]
    
    def perm_to_int(p):
        val = 0
        for x in p:
            val = val * n + (x - 1) # 0-indexed symbols
        return val

    # Build initial counts
    counts = {}
    L = len(word)
    if L < n:
        return "".join(map(str, word))
        
    # Precompute window ints
    window_ints = [0] * (L - n + 1)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        w_int = perm_to_int(p)
        window_ints[i] = w_int
        counts[w_int] = counts.get(w_int, 0) + 1
        
    # Check if all 5040 permutations are present (sanity check)
    # import math
    # if len(counts) < math.factorial(n):
    #     # Should not happen with standard construction
    #     pass

    # Greedy deletion loop
    # We will iterate through the word and try to delete characters.
    # To handle index shifts, we restart the scan after each successful deletion.
    # This is O(L^2) worst case, but L is ~6000, so 36M ops, which is fine.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        L = len(word)
        if L <= n:
            break
            
        # Rebuild window_ints if we want to be precise, or just update dynamically?
        # Dynamic update is complex. Rebuilding is O(L).
        # Since we restart the scan after deletion, we can rebuild window_ints at the start of each pass?
        # No, we can just calculate the affected windows on the fly.
        
        for i in range(L):
            if time.time() > deadline:
                break
                
            # Identify windows in current word that cover index i
            # Windows start at j such that j <= i <= j+n-1
            # j_min = max(0, i-n+1)
            # j_max = min(L-n, i)
            
            j_min = max(0, i - n + 1)
            j_max = min(L - n, i)
            
            # Store affected window ints to revert if necessary
            affected = []
            valid = True
            
            # 1. Decrement counts of existing windows covering i
            for j in range(j_min, j_max + 1):
                # Calculate window int at j
                # To avoid recalculating entire window, we can use precomputed if we had them, 
                # but we don't maintain them globally in this loop easily without rebuilding.
                # Let's just calculate it. n is small (7).
                p = tuple(word[j:j+n])
                w_int = perm_to_int(p)
                affected.append(w_int)
                counts[w_int] -= 1
                if counts[w_int] == 0:
                    valid = False
                    # We don't need to revert yet if we break, but we must revert before continuing
                    break
            
            if not valid:
                # Revert decrements
                for w_int in affected:
                    counts[w_int] += 1
                continue
            
            # 2. Check new bridging windows
            # If we delete word[i], the new word is word[:i] + word[i+1:]
            # New windows that are "different" are those that span the gap.
            # In the new word, these windows start at indices j_new such that the window covers the deletion point.
            # The deletion point is at index i in old word. In new word, the gap is between i-1 and i (if i>0) or at start.
            # Actually, let's just simulate the new windows locally.
            
            new_affected = []
            # The new windows that change are those that would have included word[i] in the old word, 
            # but now bridge the gap.
            # In the new word, the characters are shifted.
            # Let's identify which windows in the NEW word are new.
            # A window in the new word starting at j_new corresponds to old indices:
            # If j_new < i: it takes word[j_new ... i-1] and word[i+1 ... j_new+n]
            # If j_new >= i: it takes word[j_new+1 ... j_new+1+n] (which is old window j_new+1)
            
            # The windows that are NOT simply shifted old windows are those that span the deletion.
            # These are j_new in [i-n+1, i-1] (clamped to valid range in new word)
            
            new_L = L - 1
            # j_new ranges from max(0, i-n+1) to min(new_L-n, i-1) ?
            # Let's check boundaries.
            # Old index i is removed.
            # New index i is old index i+1.
            # A window in new word starting at j_new covers new indices [j_new, j_new+n-1].
            # This window is "new" if it includes at least one char from old [i+1..] and one from old [..i-1].
            # This happens if j_new < i and j_new+n-1 >= i (in old coords, the gap is at i).
            # In new coords, the gap is between i-1 and i.
            # So we need j_new <= i-1 and j_new+n-1 >= i.
            # => j_new >= i - n + 1.
            
            j_new_min = max(0, i - n + 1)
            j_new_max = min(new_L - n, i - 1)
            
            for j_new in range(j_new_min, j_new_max + 1):
                # Construct the new window
                # It consists of word[j_new : i] + word[i+1 : j_new+n]
                # Note: word[i] is excluded.
                # Length of first part: i - j_new
                # Length of second part: (j_new+n) - (i+1) = j_new + n - i - 1
                # Total length: i - j_new + j_new + n - i - 1 = n - 1. 
                # Wait, we need n characters.
                # The window in new word is of length n.
                # It takes characters from new word indices j_new to j_new+n-1.
                # Mapping to old word:
                # new[k] = old[k] for k < i
                # new[k] = old[k+1] for k >= i
                # So for k in [j_new, j_new+n-1]:
                # if k < i, val = old[k]
                # if k >= i, val = old[k+1]
                
                p_new = []
                for k in range(j_new, j_new + n):
                    if k < i:
                        p_new.append(word[k])
                    else:
                        p_new.append(word[k + 1])
                
                w_int_new = perm_to_int(p_new)
                new_affected.append(w_int_new)
                counts[w_int_new] = counts.get(w_int_new, 0) + 1
            
            # 3. Check if any count became 0
            # We decremented old, incremented new.
            # We need to ensure no count is 0.
            # We already checked old counts during decrement (valid=True implies they were >0 before decrement, so >=0 after).
            # But wait, if a count was 1, it becomes 0. That's allowed only if it's not a *required* permutation?
            # No, ALL permutations must be present. So count must be >= 1.
            # My check `if counts[w_int] == 0: valid = False` handled the decrement step.
            # However, if a window was removed (count 1->0) but is re-added by a new bridging window (count 0->1), it's fine.
            # But if a window was removed and NOT re-added, it's invalid.
            # My decrement loop breaks immediately if count hits 0.
            # This is incorrect logic for the "re-added" case.
            # Example: Window A is covered by i. Count(A)=1. Window B is new bridging. B == A.
            # Decrement A -> 0. Valid=False. Break.
            # But if we had continued, we would increment B (which is A) -> 1. Final count 1. Valid.
            
            # Correction: We must apply all decrements and increments, THEN check if any count is 0.
            
            # Let's re-structure the check.
            
            # Revert previous decrements
            for w_int in affected:
                counts[w_int] += 1
            
            # Apply decrements
            for j in range(j_min, j_max + 1):
                p = tuple(word[j:j+n])
                w_int = perm_to_int(p)
                counts[w_int] -= 1
            
            # Apply increments
            for j_new in range(j_new_min, j_new_max + 1):
                p_new = []
                for k in range(j_new, j_new + n):
                    if k < i:
                        p_new.append(word[k])
                    else:
                        p_new.append(word[k + 1])
                w_int_new = perm_to_int(p_new)
                counts[w_int_new] = counts.get(w_int_new, 0) + 1
            
            # Check validity
            # We only need to check the windows we touched.
            # If any touched window has count 0, it's invalid.
            # Note: A window might have count 0 if it was removed and not re-added.
            # Or if it was removed twice? No, we only remove one instance.
            
            is_valid = True
            # Check old affected
            for j in range(j_min, j_max + 1):
                p = tuple(word[j:j+n])
                w_int = perm_to_int(p)
                if counts[w_int] == 0:
                    is_valid = False
                    break
            
            if is_valid:
                # Check new affected
                for j_new in range(j_new_min, j_new_max + 1):
                    p_new = []
                    for k in range(j_new, j_new + n):
                        if k < i:
                            p_new.append(word[k])
                    # Wait, p_new construction in check loop is redundant. 
                    # We just need the w_ints.
                    # Let's store w_ints in a list during the increment phase.
                    pass
                    
            # This is getting messy. Let's simplify.
            # Keep a list of all touched w_ints.
            
            # --- Revised Deletion Check ---
            
            # 1. Collect old windows to remove
            old_w_ints = []
            for j in range(j_min, j_max + 1):
                p = tuple(word[j:j+n])
                w_int = perm_to_int(p)
                old_w_ints.append(w_int)
            
            # 2. Collect new windows to add
            new_w_ints = []
            for j_new in range(j_new_min, j_new_max + 1):
                p_new = []
                for k in range(j_new, j_new + n):
                    if k < i:
                        p_new.append(word[k])
                    else:
                        p_new.append(word[k + 1])
                w_int_new = perm_to_int(p_new)
                new_w_ints.append(w_int_new)
            
            # 3. Apply changes
            for w_int in old_w_ints:
                counts[w_int] -= 1
            for w_int in new_w_ints:
                counts[w_int] = counts.get(w_int, 0) + 1
            
            # 4. Validate
            valid = True
            for w_int in old_w_ints:
                if counts[w_int] < 1:
                    valid = False
                    break
            if valid:
                for w_int in new_w_ints:
                    if counts[w_int] < 1: # Should always be >=1 since we incremented
                        valid = False
                        break
            # Actually, if a count was 1 and we decremented it, it's 0. If we incremented it back, it's 1.
            # If we decremented it and did NOT increment it, it's 0.
            # So checking old_w_ints for < 1 is sufficient?
            # What if a new window has count 0? Impossible, we just incremented it.
            # What if an untouched window has count 0? It was >=1 before, we didn't touch it, so it's still >=1.
            # So we only need to check old_w_ints.
            
            if valid:
                # Perform deletion
                word.pop(i)
                changed = True
                # Restart scan
                break
            else:
                # Revert
                for w_int in old_w_ints:
                    counts[w_int] += 1
                for w_int in new_w_ints:
                    counts[w_int] -= 1
                    if counts[w_int] == 0:
                        del counts[w_int] # Optional cleanup

    return "".join(map(str, word))
