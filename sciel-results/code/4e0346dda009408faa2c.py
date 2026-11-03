import itertools
import random

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
    
    # 2. Local Search: Delete characters
    # Precompute all valid permutations
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    def get_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in all_perms:
                counts[p] = counts.get(p, 0) + 1
        return counts

    counts = get_counts(word)
    
    # Check if initial is valid (it should be)
    if len(counts) != len(all_perms):
        # Fallback to classic if something went wrong
        return "".join(map(str, word))

    improved = True
    max_passes = 50 # Increased passes to allow more deletions
    
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        i = 0
        while i < len(word):
            # Try deleting word[i]
            j_min = max(0, i - n + 1)
            j_max = min(i, len(word) - n)
            
            if j_min > j_max:
                i += 1
                continue
            
            # Get perms in these windows
            lost_perms = []
            valid_deletion = True
            for j in range(j_min, j_max + 1):
                p = tuple(word[j:j+n])
                if p not in all_perms:
                    continue
                lost_perms.append(p)
                if counts[p] == 1:
                    valid_deletion = False
                    break
            
            if not valid_deletion:
                i += 1
                continue
            
            # Tentatively remove
            for p in lost_perms:
                counts[p] -= 1
            
            # Create new word
            new_word = word[:i] + word[i+1:]
            
            # Calculate new windows in the gap
            added_perms = []
            j_min_new = max(0, i - n + 1)
            j_max_new = min(i - 1, len(new_word) - n)
            
            for j in range(j_min_new, j_max_new + 1):
                p = tuple(new_word[j:j+n])
                if p in all_perms:
                    added_perms.append(p)
                    counts[p] = counts.get(p, 0) + 1
            
            word = new_word
            improved = True
            # Move back to check the new character at i (which was i+1)
            # and also potentially i-1 if we want to be thorough?
            # Just i is enough for greedy deletion.
            i = max(0, i - 1)
            
    return "".join(map(str, word))
