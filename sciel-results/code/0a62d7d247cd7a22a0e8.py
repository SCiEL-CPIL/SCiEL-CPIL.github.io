import time
from collections import defaultdict

def construct(n):
    """
    Constructs a superpermutation using the classic recursive method,
    then applies a deletion heuristic to reduce the length.
    """
    start_time = time.time()
    
    # 1. Generate initial superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
        
        if time.time() - start_time > 110:
            break

    if n == 1:
        return "1"

    # 2. Deletion Heuristic
    # Initialize permutation counts
    perm_counts = defaultdict(int)
    for i in range(len(word) - n + 1):
        perm_counts[tuple(word[i:i+n])] += 1
        
    # We want to remove characters.
    # We iterate until no more deletions are possible or time is up.
    # To avoid infinite loops or excessive time, we limit the number of full passes.
    # Given the small gap, a few passes should suffice.
    
    max_passes = 100
    pass_count = 0
    while pass_count < max_passes:
        pass_count += 1
        if time.time() - start_time > 115:
            break
            
        deleted = False
        # We scan from left to right. If we delete, we stay at the same index (which now holds the next char) 
        # or restart? Restarting is safer but slower. 
        # Since deletions are few, restarting is fine.
        i = 0
        while i < len(word):
            if time.time() - start_time > 115:
                break
            
            # Try to delete word[i]
            # Affected old windows: start j in [max(0, i-n+1), min(i, len(word)-n)]
            start_j = max(0, i - n + 1)
            end_j = min(i, len(word) - n)
            
            # Store old windows to restore if invalid
            old_windows = []
            valid = True
            
            for j in range(start_j, end_j + 1):
                w = tuple(word[j:j+n])
                old_windows.append(w)
                perm_counts[w] -= 1
                if perm_counts[w] == 0:
                    valid = False
                    break
            
            if valid:
                # Check if we need to add new windows
                # New word length is len(word) - 1
                # New windows affected: start j in [max(0, i-n+1), min(i-1, len(word)-n-1)]
                # Note: indices in new_word correspond to old_word indices shifted by -1 for j > i
                # But we are just checking validity.
                # If valid, we commit the deletion.
                
                # Perform deletion
                word.pop(i)
                
                # Update counts for new windows
                new_start_j = max(0, i - n + 1)
                new_end_j = min(i - 1, len(word) - n)
                
                for j in range(new_start_j, new_end_j + 1):
                    w = tuple(word[j:j+n])
                    perm_counts[w] += 1
                    
                deleted = True
                # Stay at i to check the new element at i (which was i+1 before)
                # Or restart? Let's stay at i.
            else:
                # Revert counts
                for w in old_windows:
                    perm_counts[w] += 1
                i += 1
        
        if not deleted:
            break
            
    return "".join(map(str, word))
