import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s kill
    
    # For small n, return known optimal constructions
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Build the standard superpermutation via recursive insertion
    word = [1]
    for m in range(2, n + 1):
        # Find all distinct (m-1)-permutations in current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Insert m into each permutation
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap with current new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    # For n <= 4, the standard construction is optimal
    if n <= 4:
        return "".join(map(str, word))
    
    # Now do aggressive local search for n >= 5
    # Use incremental validity checking with window counting
    
    # Precompute all permutations for validity check
    all_perms = set(permutations(range(1, n+1)))
    
    # Helper: compute initial window counts
    def compute_window_counts(w):
        """Return dict mapping window tuple -> count of occurrences"""
        counts = {}
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:  # valid permutation
                window = tuple(window)
                counts[window] = counts.get(window, 0) + 1
        return counts
    
    # Check if all perms are covered
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return seen == all_perms
    
    # Incremental removal with validity tracking
    current_word = word
    current_len = len(current_word)
    
    # Track which permutations are covered and how many times
    perm_count = {}
    for i in range(len(current_word) - n + 1):
        window = tuple(current_word[i:i+n])
        if len(set(window)) == n:
            perm_count[window] = perm_count.get(window, 0) + 1
    
    covered_perms = set(perm_count.keys())
    
    # Try to remove symbols using greedy approach
    improved = True
    pass_count = 0
    max_passes = 20
    
    while improved and pass_count < max_passes and time.time() < deadline:
        improved = False
        pass_count += 1
        # Randomize scan order
        indices = list(range(len(current_word)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(current_word):
                continue
            
            # Determine which windows are affected by removing position i
            # Old windows that contain position i: start indices from max(0, i-n+1) to min(i, len(current_word)-n)
            old_start = max(0, i - n + 1)
            old_end = min(i, len(current_word) - n)
            
            affected_old = []
            for j in range(old_start, old_end + 1):
                window = tuple(current_word[j:j+n])
                if len(set(window)) == n:
                    affected_old.append((j, window))
            
            # New windows after removal: positions shift
            # New window starting at j (in new coords) where j < i and j+n-1 >= i
            # These are formed by concatenating current_word[j:i] + current_word[i+1:j+n]
            affected_new = []
            new_start = max(0, i - n + 1)
            new_end = min(i - 1, len(current_word) - n - 1)
            for j in range(new_start, new_end + 1):
                # New window at position j in new word
                new_window = tuple(current_word[j:i] + current_word[i+1:j+n])
                if len(set(new_window)) == n:
                    affected_new.append((j, new_window))
            
            # Check if removal is safe: all affected old windows must have count > 1
            # OR the affected new windows compensate
            can_remove = True
            for j, window in affected_old:
                if perm_count.get(window, 0) <= 1:
                    # This window would be lost; check if it's still covered elsewhere
                    # Actually, if count is 1 and we remove it, it's gone unless a new window has it
                    # But new windows are different positions, so we need to check if window appears in affected_new
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
            
            # Update perm_count
            for j, window in affected_old:
                perm_count[window] -= 1
                if perm_count[window] == 0:
                    del perm_count[window]
            
            for j, window in affected_new:
                perm_count[window] = perm_count.get(window, 0) + 1
            
            current_word = new_word
            current_len = len(current_word)
            improved = True
            
            # Quick sanity: ensure we still cover all perms
            if len(perm_count) < len(all_perms):
                # Revert - this shouldn't happen if logic is correct
                current_word = current_word[:i] + [removed_symbol] + current_word[i:]
                # Restore counts
                for j, window in affected_old:
                    perm_count[window] = perm_count.get(window, 0) + 1
                for j, window in affected_new:
                    perm_count[window] -= 1
                    if perm_count[window] == 0:
                        del perm_count[window]
                improved = False
                continue
    
    # Final verification and return
    if time.time() < deadline - 2:
        # Do a final verification pass
        if not is_valid(current_word):
            # Fallback to original
            current_word = word
    
    return "".join(map(str, current_word))
