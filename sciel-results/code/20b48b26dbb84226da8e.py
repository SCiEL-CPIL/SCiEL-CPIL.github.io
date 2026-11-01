import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    # 150s limit. Use 135s for construction + search to be safe.
    deadline = start_time + 135.0
    
    # For small n, return known optimal constructions
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231423124312"
    
    # Build the standard superpermutation via recursive insertion
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    if n <= 4:
        return "".join(map(str, word))
    
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i
        
    current_word = list(word)
    
    # Precompute window counts and positions
    def compute_window_data(w):
        counts = {}
        positions = {}  # window -> list of start indices
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                counts[window] = counts.get(window, 0) + 1
                if window not in positions:
                    positions[window] = []
                positions[window].append(i)
        return counts, positions

    perm_count, perm_positions = compute_window_data(current_word)
    
    # Main optimization loop with multiple strategies
    pass_count = 0
    max_passes = 200
    improved = True
    
    while improved and pass_count < max_passes and time.time() < deadline:
        improved = False
        pass_count += 1
        
        # Strategy 1: Remove positions in order of "removability" (windows with high count first)
        # Strategy 2: Random order
        # Strategy 3: Try from left to right, then right to left
        
        if pass_count % 3 == 1:
            indices = list(range(len(current_word)))
            random.shuffle(indices)
        elif pass_count % 3 == 2:
            indices = list(range(len(current_word)))
        else:
            indices = list(range(len(current_word)-1, -1, -1))
        
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(current_word):
                continue
            
            # Determine affected old windows
            old_start = max(0, i - n + 1)
            old_end = min(i, len(current_word) - n)
            
            affected_old = []
            for j in range(old_start, old_end + 1):
                window = tuple(current_word[j:j+n])
                if len(set(window)) == n:
                    affected_old.append((j, window))
            
            # Determine affected new windows
            affected_new = []
            new_start = max(0, i - n + 1)
            new_end = min(i - 1, len(current_word) - n - 1)
            
            for j in range(new_start, new_end + 1):
                new_window = tuple(current_word[j:i] + current_word[i+1:j+n+1])
                if len(set(new_window)) == n:
                    affected_new.append((j, new_window))
            
            # Check if removal is safe
            can_remove = True
            for j, window in affected_old:
                count = perm_count.get(window, 0)
                if count <= 1:
                    # Check if this exact window appears in affected_new
                    still_covered = False
                    for nj, nw in affected_new:
                        if nw == window:
                            still_covered = True
                            break
                    if not still_covered:
                        can_remove = False
                        break
            
            if not can_remove:
                continue
            
            # Perform removal
            removed_symbol = current_word[i]
            new_word = current_word[:i] + current_word[i+1:]
            
            # Update perm_count and perm_positions
            for j, window in affected_old:
                perm_count[window] -= 1
                if window in perm_positions:
                    perm_positions[window].remove(j)
                if perm_count[window] == 0:
                    del perm_count[window]
                    if window in perm_positions:
                        del perm_positions[window]
            
            for j, window in affected_new:
                perm_count[window] = perm_count.get(window, 0) + 1
                if window not in perm_positions:
                    perm_positions[window] = []
                # Adjust positions: windows starting after i shift by -1
                if window in perm_positions:
                    perm_positions[window].append(j)
            
            # Shift positions for windows that start after i
            for window in list(perm_positions.keys()):
                pos_list = perm_positions[window]
                new_pos_list = []
                for pos in pos_list:
                    if pos > i:
                        new_pos_list.append(pos - 1)
                    elif pos == i:
                        # This position was removed, but we already handled affected_old
                        continue
                    else:
                        new_pos_list.append(pos)
                perm_positions[window] = new_pos_list
            
            current_word = new_word
            improved = True

    # Final verification
    if time.time() < deadline - 2:
        seen = set()
        for i in range(len(current_word) - n + 1):
            window = tuple(current_word[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        if len(seen) != target_count:
            current_word = list(word)
    
    return "".join(map(str, current_word))
