import time
import random
from collections import Counter

def construct(n):
    if n != 7:
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # n=7
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
        
    def get_counter(w):
        c = Counter()
        for i in range(len(w) - n + 1):
            c[tuple(w[i:i+n])] += 1
        return c

    current_word = list(word)
    current_counts = get_counter(current_word)
    best_word = list(current_word)
    best_len = len(current_word)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    while time.time() < deadline:
        if len(current_word) <= n:
            break
            
        k = random.randint(0, len(current_word) - 1)
        
        # Identify windows to update
        # Old windows affected: indices j such that window [j-n+1, j] contains k
        # j-n+1 <= k <= j  =>  j >= k and j <= k+n-1
        # Also j >= n-1 and j <= len(current_word)-1
        
        old_windows = []
        for j in range(max(n-1, k), min(len(current_word), k+n)):
            if j >= n-1:
                old_windows.append(tuple(current_word[j-n+1:j+1]))
        
        # Decrement counts for old windows
        for w in old_windows:
            current_counts[w] -= 1
            if current_counts[w] == 0:
                del current_counts[w]
        
        # Perform deletion
        removed_char = current_word.pop(k)
        
        # New windows affected: indices j in new word such that window contains k
        # Note: k is now the position of the character that was at k+1.
        # The windows covering position k in the new word are the same range of j.
        new_windows = []
        for j in range(max(n-1, k), min(len(current_word), k+n)):
            if j >= n-1:
                new_windows.append(tuple(current_word[j-n+1:j+1]))
        
        # Increment counts for new windows
        for w in new_windows:
            current_counts[w] += 1
            
        if len(current_counts) == 5040:
            best_word = list(current_word)
            best_len = len(current_word)
        else:
            # Revert
            current_word.insert(k, removed_char)
            
            # Undo increments
            for w in new_windows:
                current_counts[w] -= 1
                if current_counts[w] == 0:
                    del current_counts[w]
            
            # Undo decrements
            for w in old_windows:
                current_counts[w] += 1
                
    return "".join(map(str, best_word))
