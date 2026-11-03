import time
from collections import Counter

def construct(n):
    """
    Constructs a superpermutation of length 1! + 2! + ... + n!.
    This is the classic recursive construction.
    Then applies a local search (deletion) to shorten it.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    if n == 1:
        return "1"
        
    word = [1]
    
    for m in range(2, n + 1):
        perms = []
        seen = set()
        
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) < m - 1:
                continue
            w_tuple = tuple(window)
            if w_tuple not in seen:
                seen.add(w_tuple)
                perms.append(w_tuple)
                
        new_word = []
        for p in perms:
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
                new_word.extend(list(p[overlap:]))
                new_word.append(m)
                new_word.extend(list(p))
                
        word = new_word
        
        if time.time() > deadline:
            break

    # Local Search: Try to delete characters
    if n >= 4: # Only try for larger n where improvement is possible/likely
        word_str = "".join(map(str, word))
        # We work with the list of ints for speed, or string?
        # List of ints is faster for slicing/tuple conversion?
        # String slicing is also fast.
        # Let's use list of ints.
        
        # Precompute initial counts
        # We need a function to get counts of all permutations in the list
        def get_counts(lst, n):
            counts = Counter()
            for i in range(len(lst) - n + 1):
                w = tuple(lst[i:i+n])
                counts[w] += 1
            return counts

        counts = get_counts(word, n)
        
        improved = True
        while improved:
            improved = False
            for i in range(len(word)):
                if time.time() > deadline:
                    return "".join(map(str, word))
                
                # Check validity of deleting word[i]
                # Removed windows: indices i-n+1 to i in current word
                # Added windows: indices i-n+1 to i-1 in new word (which correspond to i-n+1..i-1 shifted? No)
                
                # Let's identify the windows to remove
                removed_windows = []
                for k in range(max(0, i - n + 1), min(i + 1, len(word) - n + 1)):
                    # Window starts at k, ends at k+n-1
                    # Must contain i
                    # k <= i <= k+n-1
                    # i-n+1 <= k <= i
                    # Also k must be valid start: 0 <= k <= len(word)-n
                    w = tuple(word[k:k+n])
                    removed_windows.append(w)
                
                # Let's identify the windows to add
                # New word is word[:i] + word[i+1:]
                # Junction is between word[i-1] and word[i+1]
                # New windows crossing junction start at i-n+1 to i-1 in new word
                # In terms of old word indices:
                # New window starting at j (0 <= j < len(new_word)-n+1)
                # If j < i: new_word[j:j+n] = word[j:j+n] (if j+n-1 < i)
                # If j >= i: new_word[j:j+n] = word[j+1:j+1+n]
                
                # We only care about windows that change.
                # Windows entirely before i: unchanged.
                # Windows entirely after i: shifted, content same.
                # Windows crossing i: removed.
                # Windows crossing the new junction: added.
                
                added_windows = []
                # New junction is at index i-1 (last of left part) and i (first of right part)
                # Windows crossing this junction in new_word start at j such that:
                # j <= i-1 < j+n  => i-n < j <= i-1
                # So j ranges from max(0, i-n+1) to min(i-1, len(new_word)-n)
                # Note: len(new_word) = len(word) - 1
                for j in range(max(0, i - n + 1), min(i, len(word) - n)): # j goes up to i-1
                    # New window at j is word[j:i] + word[i+1:j+1+n] ?
                    # No.
                    # new_word[j] = word[j] if j < i else word[j+1]
                    # So new_word[j:j+n] consists of:
                    # word[j:i] (length i-j) and word[i+1:i+1+(n-(i-j))]
                    # Let's just construct the tuple
                    w_left = word[j:i]
                    w_right = word[i+1:i+1+(n - len(w_left))]
                    w = tuple(w_left + w_right)
                    added_windows.append(w)
                
                # Check if any removed window is unique and not in added
                valid = True
                for w in removed_windows:
                    if counts[w] == 1:
                        # Check if w is in added_windows
                        if w not in added_windows:
                            valid = False
                            break
                
                if valid:
                    # Perform deletion
                    # Update counts
                    for w in removed_windows:
                        counts[w] -= 1
                    for w in added_windows:
                        counts[w] += 1
                    
                    word.pop(i)
                    improved = True
                    break # Restart loop with new word

    return "".join(map(str, word))
