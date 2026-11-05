import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
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
            # Max overlap
            overlap = 0
            # Optimization: check from max possible down
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # Map permutations to indices
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    
    def get_idx(window):
        return perm_to_idx.get(window, -1)
        
    # Count initial windows
    for i in range(len(word) - n + 1):
        idx = get_idx(tuple(word[i:i+n]))
        if idx != -1:
            counts[idx] += 1
            
    best_word = word[:]
    best_len = len(word)
    
    # Greedy deletion with correct compensation check
    # We will alternate between forward and backward scans
    
    improved = True
    direction = 1 # 1 for forward, -1 for backward
    while improved and time.time() < deadline:
        improved = False
        if direction == 1:
            indices = range(len(word))
        else:
            indices = range(len(word) - 1, -1, -1)
            
        i_iter = iter(indices)
        # We need to handle index shifting carefully.
        # If we remove an element, the list shrinks.
        # If going forward: removing i shifts i+1.. to left. We stay at i (which is now the old i+1).
        # If going backward: removing i shifts i+1.. to left. We move to i-1. The elements > i are already processed.
        
        current_i = 0 if direction == 1 else len(word) - 1
        # It's safer to use a while loop with manual index management
        
        i = current_i
        while (direction == 1 and i < len(word)) or (direction == -1 and i >= 0):
            if time.time() > deadline:
                break
            
            # Identify old windows containing i
            old_windows = []
            for k in range(n):
                start = i - k
                if 0 <= start <= len(word) - n:
                    old_windows.append(tuple(word[start:start+n]))
            
            # Identify new windows
            # New word length: len(word) - 1
            new_windows = []
            # Windows in new word that straddle i
            # Start indices in new word: i-n+1 ... i-1
            for k in range(1, n):
                start = i - k
                if 0 <= start <= len(word) - 1 - n:
                    w = []
                    for idx in range(start, start+n):
                        old_idx = idx if idx < i else idx + 1
                        w.append(word[old_idx])
                    new_windows.append(tuple(w))
            
            new_window_set = set(new_windows)
            
            valid = True
            for p in old_windows:
                idx = perm_to_idx.get(p, -1)
                if idx == -1:
                    continue
                if counts[idx] == 1:
                    if p not in new_window_set:
                        valid = False
                        break
            
            if not valid:
                # Move to next
                if direction == 1:
                    i += 1
                else:
                    i -= 1
                continue
            
            # Valid removal
            # Update counts
            # Decrement old
            for p in old_windows:
                idx = perm_to_idx.get(p, -1)
                if idx != -1:
                    counts[idx] -= 1
            
            # Increment new
            for p in new_windows:
                idx = perm_to_idx.get(p, -1)
                if idx != -1:
                    counts[idx] += 1
            
            # Remove from word
            word.pop(i)
            improved = True
            
            if len(word) < best_len:
                best_len = len(word)
                best_word = word[:]
                
            # Adjust i
            if direction == 1:
                # i stays same, because word[i] is now the old word[i+1]
                pass
            else:
                # i decreases by 1, because we want to process the new word[i-1] (which was old word[i-1])
                i -= 1
                
        # Toggle direction
        direction *= -1
        
    return "".join(map(str, best_word))
