import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n != 7:
        return _standard_construct(n)
    
    # 1. Generate standard superpermutation for n=7
    word = _standard_list(n)
    
    # Precompute permutation counts
    perm_count = {}
    L = len(word)
    for i in range(L - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            perm_count[window] = perm_count.get(window, 0) + 1
            
    # Local Search
    best_len = L
    best_word = word[:]
    
    # We will try to improve the word.
    # We use a "burn-in" phase of deterministic deletion, then random swaps/deletions.
    
    # Helper to check if a deletion at index i is valid
    def can_delete(word, perm_count, i, n):
        L = len(word)
        if i < 0 or i >= L:
            return False
        start_k = max(0, i - n + 1)
        end_k = min(i, L - n)
        
        critical = False
        # Check if any window containing i has a unique permutation
        for k in range(start_k, end_k + 1):
            window = tuple(word[k:k+n])
            if len(set(window)) == n:
                if perm_count[window] == 1:
                    # Check if this permutation appears in the new configuration
                    # New window at k would be word[k:i] + word[i+1:k+n+1]
                    # We need to see if this new window is the same permutation? No, it's a different window.
                    # The permutation `window` is lost if it doesn't appear anywhere else.
                    # It appears in windows start_k..end_k.
                    # If we delete i, the windows change.
                    # The specific permutation `window` (which included word[i]) is gone from this window position.
                    # Does it appear in any OTHER window?
                    # perm_count[window] == 1 means it ONLY appears here.
                    # So if we delete i, this permutation is lost.
                    # Unless... the new window formed at k happens to be the SAME permutation?
                    # No, the new window is different (length n, different elements).
                    # Wait, if the permutation is e.g. (1,2,3,4,5,6,7) and we delete the '4',
                    # the new window is (1,2,3,5,6,7,8) which is not a permutation of 1..7.
                    # So yes, if perm_count[window] == 1, and window contains i,
                    # deleting i will remove this unique occurrence.
                    return False
        
        return True

    # Helper to perform deletion
    def perform_delete(word, perm_count, i, n):
        L = len(word)
        start_k = max(0, i - n + 1)
        end_k = min(i, L - n)
        
        # Decrement old windows
        for k in range(start_k, end_k + 1):
            window = tuple(word[k:k+n])
            if len(set(window)) == n:
                perm_count[window] -= 1
                if perm_count[window] == 0:
                    del perm_count[window]
        
        # Delete
        word.pop(i)
        
        # Increment new windows
        # The new windows are at indices start_k..end_k in the new word
        # Note: the range of k is the same, but the content is shifted.
        for k in range(start_k, end_k + 1):
            new_win = tuple(word[k:k+n])
            if len(set(new_win)) == n:
                perm_count[new_win] = perm_count.get(new_win, 0) + 1

    # Helper to check if a swap at i, i+1 is valid
    def can_swap(word, perm_count, i, n):
        L = len(word)
        if i < 0 or i + 1 >= L:
            return False
        # Affected windows: those containing i or i+1
        # k such that k <= i <= k+n-1  OR  k <= i+1 <= k+n-1
        # => i-n+1 <= k <= i  OR  i+1-n+1 <= k <= i+1
        # => i-n+1 <= k <= i+1
        start_k = max(0, i - n + 1)
        end_k = min(i + 1, L - n)
        
        # Collect old windows and their counts
        old_windows = []
        for k in range(start_k, end_k + 1):
            window = tuple(word[k:k+n])
            if len(set(window)) == n:
                old_windows.append((k, window))
                
        # Simulate swap
        # We need to check if any old window with count 1 is lost.
        # A window is "lost" if it doesn't appear in the new set of windows at these positions.
        # Actually, the window object itself might change to a different permutation.
        # If an old window W had count 1, and in the new configuration,
        # the windows at these positions are W1, W2, ..., Wm.
        # If W is not in {W1..Wm}, then W is lost (since it was unique).
        
        new_windows_set = set()
        for k in range(start_k, end_k + 1):
            # Construct new window at k after swapping i, i+1
            # This is tricky to do without actually swapping.
            # Let's just swap, check, and swap back if invalid.
            pass
            
        # Optimization: Just swap, check validity, swap back if invalid.
        word[i], word[i+1] = word[i+1], word[i]
        
        # Check if any permutation is missing
        # We need to know if the set of permutations in the new local windows
        # covers all the permutations that were in the old local windows.
        # Specifically, if any old window had count 1, it MUST appear in the new local windows.
        
        critical = False
        new_local_perms = set()
        for k in range(start_k, end_k + 1):
            new_win = tuple(word[k:k+n])
            if len(set(new_win)) == n:
                new_local_perms.add(new_win)
                
        for k, window in old_windows:
            if perm_count[window] == 1:
                if window not in new_local_perms:
                    critical = True
                    break
                    
        if critical:
            # Swap back
            word[i], word[i+1] = word[i+1], word[i]
            return False
        else:
            # Swap is valid, update counts
            # Decrement old
            for k, window in old_windows:
                perm_count[window] -= 1
                if perm_count[window] == 0:
                    del perm_count[window]
            # Increment new
            for k in range(start_k, end_k + 1):
                new_win = tuple(word[k:k+n])
                if len(set(new_win)) == n:
                    perm_count[new_win] = perm_count.get(new_win, 0) + 1
            return True

    # Main Loop
    while time.time() < deadline - 5.0:
        improved = False
        L = len(word)
        
        # Pass 1: Deterministic Deletion (Right to Left)
        for i in range(L - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            if can_delete(word, perm_count, i, n):
                perform_delete(word, perm_count, i, n)
                L -= 1
                improved = True
                # Update best if better
                if L < best_len:
                    best_len = L
                    best_word = word[:]
        
        # Pass 2: Random Swaps
        # Try a number of random swaps
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            i = random.randint(0, len(word) - 2)
            if can_swap(word, perm_count, i, n):
                improved = True
                if len(word) < best_len:
                    best_len = len(word)
                    best_word = word[:]
                    
        if not improved:
            # If no improvement in swaps, try random deletions?
            # Or break?
            # Let's try a few random deletions
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                i = random.randint(0, len(word) - 1)
                if can_delete(word, perm_count, i, n):
                    perform_delete(word, perm_count, i, n)
                    improved = True
                    if len(word) < best_len:
                        best_len = len(word)
                        best_word = word[:]
                        
            if not improved:
                break
                
    return "".join(map(str, best_word))

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
