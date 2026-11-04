import time
import random
from itertools import permutations

def standard_superpermutation(n):
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new_word = list(word)
        for p in perms:
            new_word.append(m)
            new_word.extend(p)
        word = new_word
    return word

def construct(n):
    start_time = time.time()
    timeout = 110
    
    if n <= 6:
        return "".join(map(str, standard_superpermutation(n)))
    
    if n == 7:
        word = standard_superpermutation(n)
        all_perms = set(permutations(range(1, n + 1)))
        perm_counts = {p: 0 for p in all_perms}
        
        # Initialize counts
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i+n])
            if w in perm_counts:
                perm_counts[w] += 1
        
        # Helper to get window
        def get_w(i):
            if i < 0 or i + n > len(word):
                return None
            return tuple(word[i:i+n])
        
        # Try deletions
        # We try to delete characters.
        # To be efficient, we prioritize characters that are part of many redundant windows.
        # But random is fine.
        
        # We will do a loop of random deletion attempts.
        # If successful, we keep it and continue.
        # We stop if we reach 5905 or time out.
        
        # To speed up, we can check if the character is "critical".
        # A character is critical if it is the only occurrence of some permutation.
        # We can maintain a list of "critical" indices?
        # Too complex.
        
        # Simple random deletion:
        # 1. Pick random index i.
        # 2. Check windows in [i-n+1, i].
        # 3. If any window has count 1, deletion is invalid.
        # 4. Else, delete, update counts, and accept.
        
        # Update logic:
        # Old windows: j in [i-n+1, i]
        # New windows: j in [i-n+1, i-1] (in new word)
        # Note: The new window at j (new) is the same as old window at j (old) for j < i-n+1? No.
        # The deletion shifts indices > i.
        
        # Let's implement the check.
        
        for _ in range(200000):
            if time.time() - start_time > timeout:
                break
            if len(word) <= 5905:
                break
            
            i = random.randint(0, len(word) - 1)
            
            # Check validity of deleting word[i]
            # Affected old windows: start index j such that j <= i <= j+n-1
            # j in [i-n+1, i]
            
            valid = True
            old_windows = []
            for j in range(max(0, i - n + 1), min(i + 1, len(word) - n + 1)):
                w = get_w(j)
                if w in perm_counts:
                    if perm_counts[w] == 1:
                        valid = False
                        break
                    old_windows.append((j, w))
            
            if not valid:
                continue
            
            # Perform deletion
            word.pop(i)
            
            # Update counts
            # Decrement old windows
            for j, w in old_windows:
                perm_counts[w] -= 1
            
            # Increment new windows
            # The windows that are "new" or "changed" are those that now span the gap.
            # In the new word, the windows starting at j in [i-n+1, i-1] are affected.
            # Specifically, the window starting at j in new word corresponds to:
            # If j < i: same as old window j? No, old window j was word[j:j+n].
            # New window j is word_new[j:j+n].
            # word_new[j] = word[j] for j < i.
            # word_new[j+n-1] = word[j+n] for j < i (since j+n-1 < i-1+n-1? No).
            
            # Actually, the set of windows in the new word that are different from the old word
            # are those that cover the deletion point.
            # Old windows covering i: [i-n+1, i]
            # New windows covering the gap (index i in new word is the char after deleted):
            # The windows starting at j in [i-n+1, i-1] in the new word.
            
            for j in range(max(0, i - n + 1), min(i, len(word) - n + 1)):
                w_new = get_w(j)
                if w_new in perm_counts:
                    perm_counts[w_new] += 1
            
            # Accept deletion
        
        return "".join(map(str, word))
    
    return "".join(map(str, standard_superpermutation(n)))
