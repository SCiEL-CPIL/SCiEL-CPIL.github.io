import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    # 2. Local search to shorten
    if n >= 3 and time.time() < deadline:
        word = fast_shorten(word, n, deadline)
        
    return "".join(map(str, word))

def fast_shorten(word, n, deadline):
    L = len(word)
    if L < n:
        return word
    
    # Precompute all permutations? No, just use tuples.
    # We need to track counts of each permutation.
    # Since n is small (7), we can map permutations to integers for faster hashing/lookup if needed, 
    # but tuples of ints are fine in Python for n=7.
    
    # Initial counts
    counts = {}
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        counts[p] = counts.get(p, 0) + 1
        
    # Check if all n! permutations are present
    total_perms = 1
    for i in range(1, n+1):
        total_perms *= i
        
    # Helper to check if all counts >= 1
    # This is O(n!) which is 5040. Too slow to call for every removal.
    # Instead, we track the number of permutations with count == 0.
    # If zero_count == 0, it's valid.
    
    zero_count = sum(1 for c in counts.values() if c == 0)
    # Wait, counts only has keys for present permutations.
    # We need to know if ANY of the n! permutations is missing.
    # It's better to have a dictionary of size n! initialized to 0?
    # 5040 is small.
    
    # Let's use a different approach:
    # Keep a set of "missing" permutations.
    # Initially, missing = all_perms - present_perms.
    # If missing is empty, we are good.
    # When we remove a window, if its count drops to 0, it becomes missing.
    # When we create a window, if it was missing, it becomes present.
    
    all_perms = set(permutations(range(1, n+1)))
    present = set(counts.keys())
    missing = all_perms - present
    
    if len(missing) > 0:
        # The classic construction should cover all, but just in case
        pass

    # Greedy deletion
    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(word)
        # Iterate from right to left to avoid index shifting issues?
        # Or just iterate and if removed, restart or adjust index.
        # Iterating right to left is safer for in-place modification logic if we were doing it in place, 
        # but we are building a new list or modifying the list and adjusting L.
        # Let's modify the list in place.
        
        for i in range(L - 1, -1, -1):
            if time.time() >= deadline:
                break
            
            # Identify affected windows
            # Start indices k such that k <= i < k+n
            # k_min = max(0, i - n + 1)
            # k_max = min(i, L - n)
            
            k_min = max(0, i - n + 1)
            k_max = min(i, L - n)
            
            if k_min > k_max:
                continue # No windows affected? Shouldn't happen if L>=n and i in range
            
            # Check if removal is valid
            # We need to simulate the counts change.
            # If any count drops to 0 and it's a required perm, it's invalid.
            # Actually, if a count drops to 0, that perm becomes missing.
            # If a new window is created, it might fill a missing slot.
            
            # Let's just do the counts update and check.
            # Since n is small, we can just update a temporary dict or list of changes.
            
            valid = True
            changes = [] # (perm, delta)
            
            for k in range(k_min, k_max + 1):
                old_p = tuple(word[k:k+n])
                # Decrement old_p
                if old_p in counts:
                    counts[old_p] -= 1
                    if counts[old_p] == 0:
                        # It becomes missing
                        missing.add(old_p)
                        changes.append((old_p, -1))
                else:
                    # Should not happen if counts is consistent
                    pass
                
                # Create new window if k < L - n
                if k < L - n:
                    # New window is word[k:i] + word[i+1:i+n+1]
                    # Note: word[i+1:i+n+1] is valid because i < L-1 (loop bound)
                    # and i+n+1 <= L? 
                    # i <= L-n. i+n+1 <= L+1.
                    # If i = L-n, k_max = L-n. k < L-n is false.
                    # If i < L-n, then i+n+1 <= L. 
                    # Wait, if i = L-n-1, i+n+1 = L. word[L] is out of bounds.
                    # Let's check indices carefully.
                    
                    # Word indices 0..L-1.
                    # Window k..k+n-1.
                    # Remove i.
                    # New window at k consists of:
                    # word[k]..word[i-1] and word[i+1]..word[i+n]
                    # This requires i+n <= L-1 => i <= L-n-1.
                    # If i = L-n, then i+n = L, so word[i+n] is out of bounds.
                    # But if i = L-n, then k_max = L-n.
                    # For k = L-n, the window is the last one.
                    # Removing i=L-n affects the last window.
                    # The new word has length L-1. The last window starts at L-1-n.
                    # So the window at k=L-n in the old word is destroyed.
                    # Is a new window created at k=L-n?
                    # The new word's windows start at 0..L-1-n.
                    # So k=L-n is NOT a valid start index in the new word.
                    # So no new window is created for k=L-n.
                    
                    # Condition for new window: k <= L-1-n
                    # i.e., k < L-n
                    
                    if k < L - n:
                        new_p_tuple = word[k:i] + word[i+1:i+n+1]
                        # Check if i+n+1 <= L
                        # i <= L-n-1 => i+n+1 <= L. Correct.
                        new_p = tuple(new_p_tuple)
                        
                        if new_p in counts:
                            counts[new_p] += 1
                            if counts[new_p] == 1:
                                # It was missing, now present
                                if new_p in missing:
                                    missing.remove(new_p)
                        else:
                            counts[new_p] = 1
                            if new_p in missing:
                                missing.remove(new_p)
                        changes.append((new_p, 1))
            
            # Check validity
            if len(missing) == 0:
                # Valid removal
                # Perform removal
                del word[i]
                L -= 1
                improved = True
            else:
                # Rollback changes
                for p, delta in changes:
                    counts[p] -= delta
                    if delta == -1:
                        # We decremented it to 0, so it was added to missing.
                        # Rollback: increment to 1, remove from missing
                        if counts[p] == 1:
                            missing.discard(p)
                    else:
                        # We incremented it.
                        # Rollback: decrement.
                        # If it was 0 before increment, it was in missing.
                        # Now it's back to 0? No, if it was in missing, count was 0.
                        # We incremented to 1. Rollback to 0.
                        # So it should go back to missing.
                        if counts[p] == 0:
                            missing.add(p)
                # Note: The logic for rollback of 'missing' set is tricky.
                # It's safer to not modify 'counts' and 'missing' until we know it's valid.
                
    return word

# The rollback logic above is messy. Let's refactor to calculate validity without modifying state, 
# or use a simpler check.
