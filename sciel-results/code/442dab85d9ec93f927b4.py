import time
from itertools import permutations
from collections import defaultdict

DEADLINE = time.time() + 140.0

def _build_recursive(n):
    """Classic recursive superpermutation of length 1!+2!+...+n!."""
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
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def construct(n):
    if n <= 1:
        return "1"
    
    start = time.time()
    word = _build_recursive(n)
    
    # Precompute all permutations
    all_perms = [tuple(p) for p in permutations(range(1, n + 1))]
    perm_hashes = set(all_perms)
    
    # Initialize counts and identify critical permutations (count == 1)
    counts = defaultdict(int)
    for i in range(len(word) - n + 1):
        counts[tuple(word[i:i+n])] += 1
        
    # Critical positions: positions where the window is the unique occurrence
    # of some permutation. If we delete a char from such a window, we lose it.
    # A deletion at position i is valid iff no critical window covers position i.
    
    critical_positions = set()
    for i in range(len(word) - n + 1):
        if counts[tuple(word[i:i+n])] == 1:
            for pos in range(i, i + n):
                critical_positions.add(pos)
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # Try deletions from left to right, skipping non-critical positions
        i = 0
        while i < len(word):
            if time.time() > DEADLINE:
                break
            
            # Skip positions that are not critical
            if i not in critical_positions:
                i += 1
                continue
            
            # This position is critical; try to delete it
            # Check if deletion is valid by examining affected windows
            start_j = max(0, i - n + 1)
            end_j = min(i, len(word) - n)
            
            # Collect affected windows (those that contain position i)
            affected = []
            for j in range(start_j, end_j + 1):
                affected.append(tuple(word[j:j+n]))
            
            # Compute new mixed windows after deletion
            new_mixed = []
            max_j_new = min(i - 1, len(word) - 1 - n)
            min_j_new = max(0, i - n + 1)
            
            for j_new in range(min_j_new, max_j_new + 1):
                left_part = word[j_new:i]
                right_part = word[i+1:j_new+n+1]
                new_window = left_part + right_part
                new_mixed.append(tuple(new_window))
            
            # Check validity: no permutation count should drop to 0
            valid = True
            temp_counts = {}
            for h in affected:
                if h in perm_hashes:
                    temp_counts[h] = counts.get(h, 0)
            
            for h in affected:
                if h in temp_counts:
                    temp_counts[h] -= 1
            
            for h in new_mixed:
                if h in perm_hashes:
                    temp_counts[h] = temp_counts.get(h, 0) + 1
            
            for h, c in temp_counts.items():
                if c <= 0:
                    valid = False
                    break
            
            if valid:
                # Commit deletion
                for h in affected:
                    if h in counts:
                        counts[h] -= 1
                        if counts[h] == 0:
                            del counts[h]
                for h in new_mixed:
                    counts[h] = counts.get(h, 0) + 1
                
                # Update critical positions: remove positions covered by old affected windows
                # that are no longer critical, add new critical positions
                for h in affected:
                    if counts.get(h, 0) == 0 or h not in counts:
                        for pos in range(start_j, start_j + n):
                            critical_positions.discard(pos)
                
                # Recompute critical positions for new_mixed windows
                for h in new_mixed:
                    if h in counts and counts[h] == 1:
                        for pos in range(min_j_new, min_j_new + n):
                            critical_positions.add(pos)
                
                # More precise: recompute critical positions around the deletion
                # Remove all critical positions in [start_j, end_j+n-1] and recompute
                # Actually, let's just recompute critical positions in the affected region
                # For simplicity, recompute all critical positions (expensive but correct)
                # Or better: only recompute in the region [max(0, start_j-n+1), min(len(word)-1, end_j+n)]
                
                # Recompute critical positions in the affected region
                region_start = max(0, start_j - n + 1)
                region_end = min(len(word) - n, end_j + n)
                
                # Clear critical positions in this region
                for pos in range(region_start, region_end + 1):
                    critical_positions.discard(pos)
                
                # Recompute
                for j in range(region_start, region_end + 1):
                    if j >= 0 and j <= len(word) - n:
                        w = tuple(word[j:j+n])
                        if w in counts and counts[w] == 1:
                            for pos in range(j, j + n):
                                critical_positions.add(pos)
                
                word = word[:i] + word[i+1:]
                improved = True
                break # Restart scan
            else:
                i += 1
    
    # If time remains, try a few random deletion passes to escape local minima
    if time.time() < DEADLINE - 5:
        import random
        random.seed(42)
        for _ in range(10):
            if time.time() > DEADLINE - 2:
                break
            improved = False
            while improved and time.time() < DEADLINE - 2:
                improved = False
                # Pick a random critical position to try deleting
                if not critical_positions:
                    break
                i = random.choice(list(critical_positions))
                
                start_j = max(0, i - n + 1)
                end_j = min(i, len(word) - n)
                
                affected = []
                for j in range(start_j, end_j + 1):
                    affected.append(tuple(word[j:j+n]))
                
                new_mixed = []
                max_j_new = min(i - 1, len(word) - 1 - n)
                min_j_new = max(0, i - n + 1)
                
                for j_new in range(min_j_new, max_j_new + 1):
                    left_part = word[j_new:i]
                    right_part = word[i+1:j_new+n+1]
                    new_window = left_part + right_part
                    new_mixed.append(tuple(new_window))
                
                valid = True
                temp_counts = {}
                for h in affected:
                    if h in perm_hashes:
                        temp_counts[h] = counts.get(h, 0)
                
                for h in affected:
                    if h in temp_counts:
                        temp_counts[h] -= 1
                
                for h in new_mixed:
                    if h in perm_hashes:
                        temp_counts[h] = temp_counts.get(h, 0) + 1
                
                for h, c in temp_counts.items():
                    if c <= 0:
                        valid = False
                        break
                
                if valid:
                    for h in affected:
                        if h in counts:
                            counts[h] -= 1
                            if counts[h] == 0:
                                del counts[h]
                    for h in new_mixed:
                        counts[h] = counts.get(h, 0) + 1
                    
                    # Update critical positions
                    region_start = max(0, start_j - n + 1)
                    region_end = min(len(word) - n, end_j + n)
                    for pos in range(region_start, region_end + 1):
                        critical_positions.discard(pos)
                    for j in range(region_start, region_end + 1):
                        if j >= 0 and j <= len(word) - n:
                            w = tuple(word[j:j+n])
                            if w in counts and counts[w] == 1:
                                for pos in range(j, j + n):
                                    critical_positions.add(pos)
                    
                    word = word[:i] + word[i+1:]
                    improved = True
    
    return "".join(map(str, word))
