import time
import sys
from collections import Counter

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate standard recursive superpermutation
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
            max_check = min(len(new), len(block))
            for o in range(max_check, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    L = len(word)
    
    # 2. Initial counts
    # Map permutations to IDs? Tuples are fine for n=7.
    window_counts = Counter()
    for i in range(L - n + 1):
        window_counts[tuple(word[i:i+n])] += 1
    
    total_perms = 1
    for i in range(1, n+1):
        total_perms *= i
        
    # Count how many permutations are missing (count == 0)
    # We only track counts for present permutations.
    # missing = total_perms - len(window_counts)
    missing = total_perms - len(window_counts)
    
    if missing > 0:
        # Should not happen for recursive construction, but just in case
        pass

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(L):
            if time.time() > deadline:
                break
            
            # Simulate deletion of word[i]
            
            # 1. Remove windows in current word containing i
            removed_windows = []
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            for j in range(start_min, start_max + 1):
                w = tuple(word[j:j+n])
                removed_windows.append(w)
                window_counts[w] -= 1
                if window_counts[w] == 0:
                    del window_counts[w]
                    missing += 1
            
            # 2. Add new windows in new word crossing the join
            # New word length L' = L - 1
            # Join is at index i in new word (0-indexed, first char of right part)
            # New windows start at k in [max(0, i-n+1), min(i-1, L'-n)]
            new_windows = []
            L_prime = L - 1
            if L_prime >= n:
                new_start_min = max(0, i - n + 1)
                new_start_max = min(i - 1, L_prime - n)
                for k in range(new_start_min, new_start_max + 1):
                    # Construct window from new word
                    # New word is word[0:i] + word[i+1:L]
                    # Window k in new word:
                    # If k+n <= i: all in left part -> word[k:k+n]
                    # If k >= i: all in right part -> word[k+1:k+1+n]
                    # Else: crosses -> word[k:i] + word[i+1:i+1+(n-(i-k))]
                    if k + n <= i:
                        w = tuple(word[k:k+n])
                    elif k >= i:
                        w = tuple(word[k+1:k+1+n])
                    else:
                        # Crosses
                        left_len = i - k
                        right_len = n - left_len
                        w = tuple(word[k:i] + word[i+1:i+1+right_len])
                    new_windows.append(w)
                    if w in window_counts:
                        window_counts[w] += 1
                    else:
                        window_counts[w] = 1
                        missing -= 1

            # 3. Check validity
            if missing == 0:
                # Accept deletion
                del word[i]
                L -= 1
                improved = True
                break # Restart scan as indices changed
            else:
                # Revert
                for w in removed_windows:
                    if w in window_counts:
                        window_counts[w] += 1
                    else:
                        window_counts[w] = 1
                        missing -= 1
                for w in new_windows:
                    window_counts[w] -= 1
                    if window_counts[w] == 0:
                        del window_counts[w]
                        missing += 1

    return "".join(map(str, word))
