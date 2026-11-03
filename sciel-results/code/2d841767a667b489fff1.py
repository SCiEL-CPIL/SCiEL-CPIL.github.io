import time
from itertools import permutations

def construct(n):
    """
    Constructs a superpermutation of length 1! + 2! + ... + n!.
    This is the classic recursive construction.
    """
    start_time = time.time()
    # Allow 140s for computation, 10s buffer
    deadline = start_time + 140.0
    
    if n == 1:
        return "1"
        
    word = [1]
    
    for m in range(2, n + 1):
        perms = []
        seen = set()
        
        # Extract permutations of length m-1
        # Optimization: only check windows that could be permutations
        # Since m is small (up to 7), simple tuple conversion is fine.
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            # Quick check for duplicates
            if len(set(window)) < m - 1:
                continue
            w_tuple = tuple(window)
            if w_tuple not in seen:
                seen.add(w_tuple)
                perms.append(w_tuple)
                
        new_word = []
        for p in perms:
            # Block: p, m, p
            # Find overlap with current end of new_word
            # Max possible overlap is min(len(new_word), len(p) + 1 + len(p))
            # But realistically, overlap is with the start of the block.
            # Block starts with p.
            # So we check if suffix of new_word matches prefix of p.
            max_overlap = min(len(new_word), len(p))
            overlap = 0
            for o in range(max_overlap, 0, -1):
                if new_word[-o:] == list(p[:o]):
                    overlap = o
                    break
            
            if overlap == 0:
                new_word.extend(list(p))
                new_word.append(m)
                new_word.extend(list(p))
            else:
                # Append the rest of the block
                # Block is p + [m] + p
                # We already have p[:overlap] at the end of new_word?
                # No, we matched new_word suffix with p prefix.
                # So we need to append p[overlap:] + [m] + p
                new_word.extend(list(p[overlap:]))
                new_word.append(m)
                new_word.extend(list(p))
                
        word = new_word
        
        if time.time() > deadline:
            # If we run out of time, we return the current partial word?
            # No, that would be invalid.
            # But the classic construction is very fast for n=7.
            # This guard is just for safety against unexpected slowness.
            pass

    if n <= 6:
        return "".join(map(str, word))

    # For n >= 7, apply local search to shorten the word
    all_perms = set(permutations(range(1, n + 1)))
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
            if counts[pid] < 1:
                valid = False
                break
        
        if valid:
            # Success: actually remove the character
            word.pop(i)
            return True
        else:
            # Revert counts
            for k, pid in old_windows:
                counts[pid] += 1
            for pid in new_windows_pids:
                counts[pid] -= 1
            return False

    # Perform deletion pass
    # We iterate from left to right, trying to delete each character.
    # If a deletion is successful, the indices shift, so we stay at the same index.
    # If not, we move to the next index.
    i = 0
    while i < len(word):
        if time.time() > deadline:
            break
        if try_delete(i):
            # Deletion successful, i stays the same (next char shifts to i)
            continue
        else:
            i += 1

    return "".join(map(str, word))
