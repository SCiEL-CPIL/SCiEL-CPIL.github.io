import itertools
import random
from collections import defaultdict

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
    
    # 2. Optimized Local Search: Delete characters
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Precompute window tuples for the current word
    def get_window_tuples(w):
        return [tuple(w[i:i+n]) for i in range(len(w) - n + 1)]
        
    w_tuples = get_window_tuples(word)
    counts = defaultdict(int)
    for t in w_tuples:
        if t in all_perms:
            counts[t] += 1
            
    # Check validity
    if len(counts) != len(all_perms):
        return "".join(map(str, word))

    improved = True
    max_passes = 20 # Increase passes to allow more optimization time
    
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        i = 0
        L = len(word)
        # We need to update w_tuples and counts as we go?
        # No, we can try deletions in a single pass. If we delete, we break and restart the pass.
        # This is safe but might be slow if we delete many.
        # However, deleting reduces L, so the next pass is faster.
        
        while i < L:
            # Try deleting word[i]
            # Affected windows in current word: j in [max(0, i-n+1), min(i, L-n)]
            j_min = max(0, i - n + 1)
            j_max = min(i, L - n)
            
            if j_min > j_max:
                i += 1
                continue
            
            lost_perms = []
            valid_deletion = True
            
            for j in range(j_min, j_max + 1):
                t = w_tuples[j]
                if t in all_perms:
                    lost_perms.append(t)
                    if counts[t] == 1:
                        valid_deletion = False
                        break
            
            if not valid_deletion:
                i += 1
                continue
            
            # Tentatively remove
            for t in lost_perms:
                counts[t] -= 1
            
            # Create new word
            new_word = word[:i] + word[i+1:]
            new_L = len(new_word)
            
            # Calculate new windows in the gap
            # The junction is between new_word[i-1] and new_word[i]
            # New windows start at j' in [max(0, i-n+1), min(i-1, new_L-n)]
            added_perms = []
            j_min_new = max(0, i - n + 1)
            j_max_new = min(i - 1, new_L - n)
            
            # We need to check if these new windows are permutations
            for j in range(j_min_new, j_max_new + 1):
                # Construct tuple from new_word
                # Note: new_word is a list. Slicing is O(n).
                t = tuple(new_word[j:j+n])
                if t in all_perms:
                    added_perms.append(t)
                    counts[t] = counts.get(t, 0) + 1
            
            # If we get here, the deletion is valid
            word = new_word
            L = new_L
            improved = True
            # Restart scan? Or continue?
            # Indices after i are shifted left by 1.
            # w_tuples is now invalid for indices >= i.
            # It is safest to restart the while i < L loop.
            break
            
    return "".join(map(str, word))
