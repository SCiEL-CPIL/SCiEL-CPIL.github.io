import time
import itertools
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n <= 4:
        return _standard_construct(n)
    
    if n == 5:
        # For n=5, standard is optimal-ish
        return _standard_construct(5)
    
    if n == 6:
        # Standard construction for n=6 is length 195
        # Greedy deletion can improve it
        return _improve_superperm(6, deadline - start_time)
        
    if n == 7:
        # Standard construction for n=7 is length 5913
        # We attempt greedy deletion to reduce length
        return _improve_superperm(7, deadline - start_time)
        
    # Fallback for other n
    return _standard_construct(n)

def _improve_superperm(n, budget_seconds):
    """
    Starts with the standard superpermutation for n, and applies greedy deletions.
    """
    word = _standard_list(n)
    L = len(word)
    
    # Precompute permutation counts
    perm_count = {}
    for i in range(L - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            perm_count[window] = perm_count.get(window, 0) + 1
            
    # We will attempt to delete symbols from right to left.
    # If a deletion is safe, we perform it and update counts.
    
    # To make it faster, we don't do multiple passes immediately. 
    # One pass from right to left is often sufficient to find many deletions.
    # If we delete at index i, indices < i are unchanged.
    # So a single right-to-left pass is valid and linear in L (with local checks).
    
    # Let's do a right-to-left pass.
    # We also track a "last_safe" index to avoid redundant checks if we want to be smart, 
    # but simple right-to-left is O(L * n * log L) or similar which is fine for 5913.
    
    # Actually, checking validity of deletion at i requires checking O(n) windows.
    # Each window check is O(n). So O(n^2) per index.
    # Total O(L * n^2).
    # For L=5913, n=7, this is ~300,000 ops. Very fast.
    
    # However, if we delete, we must update counts.
    # Updating counts for affected windows (O(n) of them) is fast.
    
    # Let's do a loop.
    # We scan i from L-1 down to 0.
    # If we delete at i, L decreases by 1.
    # We continue scanning downwards.
    
    # One issue: after deleting at i, the "context" for i-1 changes slightly?
    # No, indices < i are unchanged. So scanning i-1 is safe.
    
    # We can keep going until no more deletions.
    # But a single pass might miss opportunities where deleting at i+1 first made i deletable?
    # No, deletion only removes chars. If i wasn't deletable before i+1 was deleted, 
    # could it be now?
    # Deleting i+1 shifts left. The windows affected by i are the same set of positions relative to i.
    # However, the *content* of windows might change if they overlap with i+1?
    # Windows containing i do NOT contain i+1 necessarily? 
    # Window at i-1: [i-1, i]... contains i, not i+1?
    # Wait, window at i-1 covers indices i-1..i-1+n-1 = i-1..i+n-2.
    # Does it contain i+1? Yes, if n>=3.
    # So deleting i+1 DOES affect the content of windows that contain i.
    # Therefore, a symbol that was NOT deletable might become deletable after a later deletion?
    # Actually, deleting a symbol usually makes it HARDER to delete neighbors because it reduces overlaps?
    # Or easier?
    # If we delete i+1, we might break a critical permutation that made i non-deletable?
    # Or we might form a new duplicate that allows deletion?
    # It's complex. Multiple passes are safer.
    
    max_passes = 20
    improved = True
    pass_count = 0
    
    while improved and pass_count < max_passes:
        pass_count += 1
        if time.time() > start_time + budget_seconds - 5.0:
            break
            
        improved = False
        L = len(word)
        
        # Scan from right to left
        for i in range(L - 1, -1, -1):
            if time.time() > start_time + budget_seconds - 5.0:
                break
                
            # Check if deleting word[i] is safe
            
            # Affected windows in current word:
            # Start k such that k <= i <= k+n-1
            # start_k = max(0, i - n + 1)
            # end_k = min(i, L - n)
            
            start_k = max(0, i - n + 1)
            end_k = min(i, L - n)
            
            if start_k > end_k:
                # No windows contain i? 
                # This happens if i > L-n-1? 
                # If i = L-1, end_k = L-1, start_k = L-n. 
                # Since L-n <= L-1, start_k <= end_k is always true for valid i in range.
                continue
                
            # Collect old windows and their counts
            old_windows = []
            safe = True
            
            # We need to know if any of these windows is "critical" (count == 1)
            # AND if it still exists after deletion.
            
            # Let's simulate the deletion locally.
            # New segment from start_k to end_k+n-1
            # Old: word[start_k : end_k+n+1] (wait, end_k is last start index, so last char is end_k+n-1)
            # So old segment is word[start_k : end_k+n]
            
            # But we don't need the whole segment if we just check the specific windows.
            
            # Step 1: Check if any critical window disappears.
            # Critical window: one with count 1.
            # If we delete i, does this window still exist?
            # It exists if it can be formed by other chars.
            # Actually, simpler:
            # 1. List all unique critical perms in affected windows.
            # 2. Construct new segment.
            # 3. Check if all critical perms are present in new segment.
            
            critical_perms = set()
            for k in range(start_k, end_k + 1):
                win = tuple(word[k:k+n])
                if len(set(win)) == n:
                    if perm_count[win] == 1:
                        critical_perms.add(win)
                        
            if not critical_perms:
                # No critical perms, deletion is SAFE (regarding coverage)
                safe = True
            else:
                # Simulate new segment
                # Old indices involved: start_k to end_k+n-1
                # New segment excludes index i
                new_seg = word[start_k:i] + word[i+1:end_k+n+1]
                
                # Check if all critical perms are in new_seg
                # Since new_seg length is small (approx n + n - 1), we can check substrings
                # Or just slide window over new_seg
                
                found_all = True
                for p in critical_perms:
                    if p not in _sliding_windows(new_seg, n):
                        found_all = False
                        break
                safe = found_all
                
            if safe:
                # Perform Deletion
                # Update perm_count
                # Decrement old windows
                for k in range(start_k, end_k + 1):
                    old_win = tuple(word[k:k+n])
                    if len(set(old_win)) == n:
                        perm_count[old_win] -= 1
                        if perm_count[old_win] == 0:
                            del perm_count[old_win]
                            
                word.pop(i)
                L -= 1
                improved = True
                
                # Increment new windows
                # Windows at start_k..end_k in the NEW word
                # Note: In the new word, indices are shifted for j > i.
                # But start_k <= i. 
                # If k < i, window k is shifted right? No, k is an index.
                # The window starting at index k in the NEW word corresponds to:
                # If k < i: it ends at k+n-1. If k+n-1 < i, it's unchanged?
                # Actually, we deleted at i.
                # New word indices:
                # j < i: same.
                # j >= i: new word[j] = old word[j+1].
                
                # We need to recount the windows that CHANGED.
                # The windows that include the deleted position i are those k where k <= i <= k+n-1.
                # These are exactly k in [start_k, end_k].
                # In the new word, do these windows have the same index?
                # Yes, the start index k is still k.
                # So we just need to recount windows starting at k in [start_k, end_k].
                
                for k in range(start_k, end_k + 1):
                    # Ensure k is a valid start index in new word
                    if k < L - n + 1:
                        new_win = tuple(word[k:k+n])
                        if len(set(new_win)) == n:
                            perm_count[new_win] = perm_count.get(new_win, 0) + 1
            else:
                # Cannot delete
                pass
                
    return "".join(map(str, word))

def _sliding_windows(segment, n):
    """Generator of windows in segment."""
    L = len(segment)
    for i in range(L - n + 1):
        yield tuple(segment[i:i+n])

def _standard_construct(n):
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _standard_list(n):
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word
