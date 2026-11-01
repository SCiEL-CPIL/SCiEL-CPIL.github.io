import time
import itertools
import random
from collections import Counter

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Classic recursive construction of superpermutation
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For n <= 4, the classic construction is optimal, return immediately
    if n <= 4:
        return "".join(map(str, word))
    
    # Precompute all valid permutations
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        perm_counts[window] += 1
    
    # Greedy deletion with fast updates
    def try_delete(word, i, perm_counts, all_perms, n):
        """
        Checks if deleting word[i] is safe.
        Returns (is_safe, new_word, new_counts) if safe, else (False, None, None).
        """
        s_min = max(0, i - n + 1)
        s_max = min(i, len(word) - n)
        
        # Check safety
        for s in range(s_min, s_max + 1):
            window = tuple(word[s:s + n])
            if window in all_perms:
                if perm_counts[window] == 1:
                    return False, None, None
        
        # Safe to delete
        new_word = word[:i] + word[i+1:]
        
        # Update counts
        for s in range(s_min, s_max + 1):
            window = tuple(word[s:s + n])
            perm_counts[window] -= 1
        
        new_s_min = max(0, i - n + 1)
        new_s_max = min(i - 1, len(new_word) - n)
        
        for s in range(new_s_min, new_s_max + 1):
            window = tuple(new_word[s:s + n])
            perm_counts[window] += 1
            
        return True, new_word, perm_counts
    
    # Multiple passes with different strategies
    strategies = [
        "left_to_right",
        "right_to_left", 
        "random",
        "alternating"
    ]
    
    best_word = word[:]
    best_len = len(word)
    best_counts = perm_counts.copy()
    
    for strategy in strategies:
        if time.time() - start_time > deadline - 10:
            break
            
        current_word = best_word[:]
        current_counts = best_counts.copy()
        
        changed = True
        while changed and (time.time() - start_time) < deadline - 5:
            changed = False
            
            if strategy == "left_to_right":
                indices = range(len(current_word))
            elif strategy == "right_to_left":
                indices = range(len(current_word) - 1, -1, -1)
            elif strategy == "random":
                indices = list(range(len(current_word)))
                random.shuffle(indices)
            else:  # alternating
                indices = range(len(current_word))
                if changed:
                    indices = range(len(current_word) - 1, -1, -1)
            
            for i in indices:
                if time.time() - start_time > deadline - 5:
                    break
                
                if i >= len(current_word):
                    continue
                    
                safe, new_word, new_counts = try_delete(
                    current_word, i, current_counts, all_perms, n
                )
                if safe:
                    current_word = new_word
                    current_counts = new_counts
                    changed = True
                    if strategy in ["left_to_right", "right_to_left"]:
                        break  # Restart scan
                    elif strategy == "alternating":
                        break  # Restart scan
            
            if len(current_word) < best_len:
                best_word = current_word[:]
                best_len = len(current_word)
                best_counts = current_counts.copy()
    
    return "".join(map(str, best_word))
