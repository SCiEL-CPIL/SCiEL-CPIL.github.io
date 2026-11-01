import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 115.0  # Safety margin

    # 1. Generate initial superpermutation (Recursive)
    # This yields length sum(k! for k in 1..n)
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique permutations of length m-1 from the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap with the current end of 'new'
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n <= 4:
        return "".join(map(str, word))

    # 2. Optimized Local Search Deletion
    
    # Precompute permutation ranks for O(1) lookup
    # Map tuple of digits to rank 0..n!-1
    perm_map = {}
    perms = list(permutations(range(1, n+1)))
    for i, p in enumerate(perms):
        perm_map[p] = i
    
    total_perms = len(perms)
    
    def get_rank(w_tuple):
        try:
            return perm_map[w_tuple]
        except KeyError:
            return -1

    L = len(word)
    counts = [0] * total_perms
    window_ranks = [-1] * L
    
    # Initialize counts and window_ranks
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        rank = get_rank(w)
        window_ranks[i] = rank
        if rank != -1:
            counts[rank] += 1

    # Deletion Loop
    # We scan left to right. If we delete, we don't advance i.
    # We use a flag to track if we made any progress in a full pass.
    # If no progress in a full pass, we break.
    
    while time.time() < deadline:
        deleted_any = False
        i = 0
        while i < L - n + 1:
            # Check if we can delete word[i]
            # Old windows involved: start at i-n+1 to i
            # New windows involved: start at i-n+1 to i-1 (in the new string)
            
            start_w = max(0, i - n + 1)
            end_w = min(i, L - n)
            
            # 1. Collect valid new ranks
            # The new windows are formed by removing word[i]
            # Window at index j (in new string) corresponds to word[j:i] + word[i+1:j+n+1]
            # This window exists in the old string if j < i and j+n+1 <= L
            new_valid_ranks = []
            for j in range(start_w, i):
                if j + n + 1 > L:
                    continue
                # Construct the new window
                # Optimization: word[j:i] + word[i+1:j+n+1]
                new_w = word[j:i] + word[i+1:j+n+1]
                rank = get_rank(tuple(new_w))
                if rank != -1:
                    new_valid_ranks.append(rank)
            
            # 2. Check if deletion is safe
            can_delete = True
            for w in range(start_w, end_w + 1):
                rank = window_ranks[w]
                if rank != -1 and counts[rank] == 1:
                    # This valid permutation is unique in the current string.
                    # It must be preserved in the new string.
                    # Since it's unique, it must appear in one of the new windows.
                    if rank not in new_valid_ranks:
                        can_delete = False
                        break
            
            if can_delete:
                # Perform deletion
                # Update counts for old valid windows
                for w in range(start_w, end_w + 1):
                    rank = window_ranks[w]
                    if rank != -1:
                        counts[rank] -= 1
                
                # Update counts for new valid windows
                for rank in new_valid_ranks:
                    counts[rank] += 1
                
                # Delete from word
                word.pop(i)
                L -= 1
                deleted_any = True
                
                # Update window_ranks
                # Truncate window_ranks to new length
                window_ranks = window_ranks[:L]
                
                # Recompute from start_w to L-n
                # Note: The windows before start_w are unchanged.
                # The windows from start_w to L-n might change.
                for k in range(start_w, L - n + 1):
                    w_tuple = tuple(word[k:k+n])
                    window_ranks[k] = get_rank(w_tuple)
                
                # Do not increment i, check the new element at i.
            else:
                i += 1
        
        if not deleted_any:
            break

    return "".join(map(str, word))
