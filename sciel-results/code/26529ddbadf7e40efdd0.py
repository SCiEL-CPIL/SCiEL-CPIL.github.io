import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 115.0  # Safety margin

    # 1. Generate initial superpermutation (Recursive)
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
            for o in range(max_ov, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n <= 4:
        return "".join(map(str, word))

    # 2. Optimized Local Search Deletion
    
    # Precompute permutation ranks
    perm_map = {}
    perms = list(permutations(range(1, n+1)))
    for i, p in enumerate(perms):
        perm_map[p] = i
    
    L = len(word)
    total_perms = len(perms)
    counts = [0] * total_perms
    window_ranks = [0] * L
    # We need to store -1 for invalid windows?
    # Or just handle it in the loop.
    # Let's store the rank, and a separate boolean?
    # Or just store the rank, and if invalid, store -1.
    
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        if w in perm_map:
            rank = perm_map[w]
            window_ranks[i] = rank
            counts[rank] += 1
        else:
            window_ranks[i] = -1

    # Deletion Loop
    while time.time() < deadline:
        deleted_any = False
        i = 0
        while i < L - n + 1:
            # Check if we can delete word[i]
            start_w = max(0, i - n + 1)
            end_w = min(i, L - n)
            
            # Collect new ranks (only valid ones)
            new_ranks = []
            new_ranks_set = set()
            for j in range(start_w, i):
                if j + n + 1 > L:
                    continue
                new_w = word[j:i] + word[i+1:j+n+1]
                tw = tuple(new_w)
                if tw in perm_map:
                    rank = perm_map[tw]
                    new_ranks.append(rank)
                    new_ranks_set.add(rank)
            
            can_delete = True
            # Check old windows
            for w in range(start_w, end_w + 1):
                rank = window_ranks[w]
                if rank != -1 and counts[rank] == 1:
                    # This permutation is unique. It must appear in new_ranks_set.
                    if rank not in new_ranks_set:
                        can_delete = False
                        break
            
            if can_delete:
                # Perform deletion
                # Decrement old counts
                for w in range(start_w, end_w + 1):
                    rank = window_ranks[w]
                    if rank != -1:
                        counts[rank] -= 1
                
                # Increment new counts
                for rank in new_ranks:
                    counts[rank] += 1
                
                # Delete from word
                word.pop(i)
                L -= 1
                deleted_any = True
                
                # Update window_ranks
                # Recompute from start_w to end
                for k in range(start_w, L - n + 1):
                    tw = tuple(word[k:k+n])
                    if tw in perm_map:
                        window_ranks[k] = perm_map[tw]
                    else:
                        window_ranks[k] = -1
                
                # Do not increment i
            else:
                i += 1
        
        if not deleted_any:
            break

    return "".join(map(str, word))
