import time
from itertools import permutations

def construct(n):
    deadline = time.time() + 140

    # 1. Generate standard superpermutation
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
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    if n <= 6:
        return "".join(map(str, word))

    # 2. Setup for local search
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if p in perm_to_id:
            counts[perm_to_id[p]] += 1
            
    # Check if valid initially
    if any(c == 0 for c in counts):
        # Should not happen for standard construction
        return "".join(map(str, word))

    # Local Search: Deletion
    # We will repeatedly try to delete characters
    # To avoid O(N^2) complexity, we use the fast count update
    
    def try_delete(i):
        """
        Attempts to delete word[i]. Returns True if successful.
        Modifies word and counts if successful.
        """
        L = len(word)
        if L == n: # Cannot delete if only one window
            return False
        
        # Affected windows in old word: start k in [i-n+1, i]
        # We need to decrement counts for these
        # And increment counts for new windows formed after deletion
        
        # Store old values to revert if fail
        old_counts = counts.copy() # This is O(5040), acceptable? 5040 * 6000 ops = 30M ops. Might be slow in Python.
        
        # Better: Just check validity without modifying global state first?
        # Or modify and revert.
        
        # Let's modify and revert.
        # 1. Decrement old windows
        old_windows = []
        for k in range(max(0, i - n + 1), min(i + 1, L - n + 1)):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] -= 1
                old_windows.append((k, pid))
        
        # Check if any count became negative? No, we are removing an existing window.
        # But we need to ensure we don't lose a permutation that is ONLY in these windows.
        # So if counts[pid] becomes 0, that permutation is lost.
        # We must check if the new windows restore it.
        
        # 2. Simulate deletion to find new windows
        # New word is word[:i] + word[i+1:]
        # New windows affected are those starting at k in [i-n+1, i] in the NEW word.
        # In the new word, index i is removed.
        # Windows starting at k < i:
        #   If k+n <= i: window is same as old word[k:k+n]. No change.
        #   If k+n > i: window is word[k:i] + word[i+1:k+n+1].
        # Windows starting at k >= i:
        #   In new word, these correspond to old word[k+1:k+1+n].
        
        # Actually, it's easier to just compute the new windows in the affected range.
        # Affected range in new word: start k in [max(0, i-n+1), min(i, L-n)]
        
        new_windows_pids = []
        for k in range(max(0, i - n + 1), min(i, L - n)): # k is start index in NEW word
            # The window is new_word[k:k+n]
            # new_word is word with i removed.
            # If k < i:
            #   new_word[k] = word[k]
            #   ...
            #   new_word[k + (i-k) - 1] = word[i-1]
            #   new_word[k + (i-k)] = word[i+1]
            # So the window consists of:
            # word[k : i] (length i-k)
            # word[i+1 : k+n+1] (length n - (i-k))
            
            if k < i:
                w1 = word[k:i]
                w2 = word[i+1:k+n+1]
                p = tuple(w1 + w2)
            else: # k >= i
                # new_word[k] = word[k+1]
                # window is word[k+1 : k+1+n]
                p = tuple(word[k+1:k+1+n])
                
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] += 1
                new_windows_pids.append(pid)
                
        # Check if all counts are >= 1
        valid = True
        for pid in range(num_perms):
            if counts[pid] == 0:
                valid = False
                break
                
        if valid:
            # Perform deletion
            word.pop(i)
            # Note: counts are already updated to reflect the new state
            return True
        else:
            # Revert counts
            counts[:] = old_counts
            return False

    # Iterate deletion passes
    improved = True
    while improved:
        improved = False
        # Left to right
        i = 0
        while i < len(word):
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_delete(i):
                improved = True
                # Do not increment i, check same position again (new character)
            else:
                i += 1
        # Right to left
        i = len(word) - 1
        while i >= 0:
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_delete(i):
                improved = True
            else:
                i -= 1

    return "".join(map(str, word))
