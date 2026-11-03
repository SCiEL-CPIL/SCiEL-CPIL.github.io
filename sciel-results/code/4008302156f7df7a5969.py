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
    max_passes = 10 # Safety limit
    
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        i = 0
        while i < len(word):
            # Try deleting word[i]
            # Identify windows covering i
            # Window j covers i if j <= i < j+n
            # j_min = max(0, i - n + 1)
            # j_max = min(i, len(word) - n)
            
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
                    # This window is not a permutation, so it doesn't contribute to counts
                    # But wait, if it's not a perm, it wasn't in counts.
                    # So deleting it doesn't affect counts.
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
            # New windows are at indices j' in new_word
            # The gap is around i.
            # The new windows that are different are those that span the deletion point.
            # In new_word, the deletion is at i.
            # Windows j' such that j' < i < j'+n in new_word?
            # No, the indices shift.
            # The new windows are at j' in [i-n+1, i-1] (in new_word coordinates)
            # Let's just compute the new perms for the affected range in new_word.
            # Affected range in new_word:
            # The deletion was at i.
            # The windows in new_word that are "new" are those that include the character at i-1 and i+1 (which are now adjacent).
            # These are windows starting at j' such that the window covers the junction.
            # Junction is between new_word[i-1] and new_word[i].
            # So windows starting at j' where j' <= i-1 < j'+n.
            # j' in [max(0, i-n+1), min(i-1, len(new_word)-n)]
            
            added_perms = []
            j_min_new = max(0, i - n + 1)
            j_max_new = min(i - 1, len(new_word) - n)
            
            for j in range(j_min_new, j_max_new + 1):
                p = tuple(new_word[j:j+n])
                if p in all_perms:
                    added_perms.append(p)
                    counts[p] = counts.get(p, 0) + 1
            
            # Check if all counts are still >= 1
            # We only modified counts for lost_perms and added_perms.
            # If any count in lost_perms became 0, it's invalid.
            # But we already checked counts[p] > 1 for lost_perms before decrementing.
            # So counts[p] >= 1 after decrement.
            # Added perms just increase counts.
            # So the word is valid!
            
            word = new_word
            improved = True
            # Do not increment i, restart scan? Or continue?
            # Continuing is risky because indices shifted.
            # But we can just restart the while loop for safety.
            break
            
    return "".join(map(str, word))
