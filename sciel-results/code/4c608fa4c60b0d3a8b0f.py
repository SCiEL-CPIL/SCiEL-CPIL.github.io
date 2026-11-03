import time
import random

def construct(n):
    if n <= 1:
        return "1"
    
    # 1. Construct Classic Superpermutation
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Initial length is sum(1! to n!)
    # For n=7, length is 5913
    
    # 2. Local Search / Greedy Deletion
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    current_word = word
    target_set = set(range(1, n+1))
    
    # Helper to check validity
    def is_valid(w):
        if len(w) < n:
            return False
        # Check all windows
        for i in range(len(w) - n + 1):
            if set(w[i:i+n]) != target_set:
                return False
        return True
        
    # Optimized check using a set of seen permutations?
    # No, we need to ensure ALL 5040 are present.
    # So we must collect all windows and check if the set size is 5040 and they are all valid permutations.
    # Actually, if the string is derived from a superpermutation by deletions,
    # any window that is a permutation of 1..n is a valid one.
    # We just need to ensure the set of valid windows has size n!
    
    def is_superpermutation(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            win = tuple(w[i:i+n])
            # Check if it's a permutation of 1..n
            # Since n is small (7), set(win) == target_set is fast
            if set(win) == target_set:
                seen.add(win)
                if len(seen) == 5040: # n!
                    return True
        return len(seen) == 5040

    # Initial check (should be true)
    # is_superpermutation(current_word)
    
    improved = True
    while time.time() < deadline and improved:
        improved = False
        L = len(current_word)
        # Randomize order to escape specific local minima
        indices = list(range(L))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            # Try to delete current_word[i]
            # Optimization: Only try to delete if it's not part of a "critical" structure?
            # Just try all.
            
            # Create candidate
            # Using list slicing is O(L)
            candidate = current_word[:i] + current_word[i+1:]
            
            # Check validity
            if is_superpermutation(candidate):
                current_word = candidate
                improved = True
                # Reset indices? No, just continue with the next index in the original list?
                # The indices are for the OLD length.
                # If we delete, the length changes.
                # It's safer to restart the pass if a deletion happens, or adjust indices.
                # Given the cost, restarting the pass is cleaner.
                break
        
    return "".join(map(str, current_word))
