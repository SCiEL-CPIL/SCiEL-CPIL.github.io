"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import defaultdict

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Build standard recursive superpermutation
    def build_standard(m):
        if m == 1:
            return [1]
        if m == 2:
            return [1, 2, 1]
        
        prev = build_standard(m - 1)
        perms = []
        seen = set()
        for i in range(len(prev) - m + 2):
            window = tuple(prev[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        return new_word
    
    w = build_standard(n)
    
    # Efficient Local Search
    # Maintain counts of all permutations in the current word
    perm_counts = defaultdict(int)
    
    def update_counts_add(w, start, end):
        """Add counts for windows starting in [start, end)"""
        for i in range(start, end):
            if i + n > len(w): break
            t = tuple(w[i:i+n])
            if len(set(t)) == n:
                perm_counts[t] += 1

    def update_counts_sub(w, start, end):
        """Subtract counts for windows starting in [start, end)"""
        for i in range(start, end):
            if i + n > len(w): break
            t = tuple(w[i:i+n])
            if len(set(t)) == n:
                perm_counts[t] -= 1

    # Initialize counts
    for i in range(len(w) - n + 1):
        t = tuple(w[i:i+n])
        if len(set(t)) == n:
            perm_counts[t] += 1
    
    total_perms = factorial(n)
    
    # We want to find deletions.
    # Iterate through positions.
    # To avoid index shifting issues, we can iterate from right to left, 
    # but deleting changes indices. 
    # A safer way: iterate, and if deleted, restart or adjust.
    # Given n=7, N~6000, a single pass from right to left might be sufficient 
    # if we are careful, or we can do multiple passes.
    
    # Let's do a single pass from right to left.
    # If we delete w[i], the windows affected are those starting at i-n+1 ... i.
    # The new windows are those starting at i-n+1 ... i-1 (shifted).
    
    # Actually, simpler logic:
    # If we remove w[i]:
    # Old windows involving w[i]: start indices j such that j <= i < j+n => i-n+1 <= j <= i.
    # These windows are "destroyed".
    # New windows formed by joining w[i-1] and w[i+1]:
    # The word becomes w[0:i] + w[i+1:].
    # The new windows are those that cross the gap.
    # Specifically, the window that was w[i:i+n] is gone.
    # The window w[i-1:i-1+n] becomes w[i-1:i-1+n-1] + w[i+1] ... wait.
    # Let's just re-calculate the local neighborhood.
    
    # A robust and simple efficient check:
    # We track the number of "unique" permutations (count == 1).
    # If we delete a char, we only lose a permutation if it was unique AND it is one of the destroyed windows.
    
    # Let's implement the delta update.
    
    # Pass 1: Try to delete from right to left.
    # We need to be careful with indices.
    # If we delete w[i], the length decreases.
    
    # Let's use a list `w`.
    # We will iterate i from len(w)-2 down to 1.
    
    # Optimization: Only attempt deletion if the character is "suspect".
    # But for now, let's just try all.
    
    # To handle index shifts:
    # If we delete w[i], we continue with i-1 (which is now the next char to the left? No, we are going right to left).
    # If we go right to left, and delete w[i], the indices of elements to the left (smaller index) do not change.
    # So we can just continue the loop.
    
    for i in range(len(w) - 2, 0, -1):
        # Check if deletion is safe
        # Destroyed windows: start j in [i-n+1, i]
        # We need to check if any of these windows is "unique" (count 1) and not recreated.
        
        # Actually, it's easier to just:
        # 1. Subtract counts of windows starting at i-n+1 ... i
        # 2. Remove w[i]
        # 3. Add counts of new windows starting at i-n+1 ... i-1 (in the new word)
        # 4. Check if any count became 0.
        
        # Let's implement this carefully.
        
        # Store old windows
        old_starts = range(max(0, i - n + 1), i + 1)
        old_windows = []
        for j in old_starts:
            if j + n <= len(w):
                t = tuple(w[j:j+n])
                if len(set(t)) == n:
                    old_windows.append(t)
        
        # Tentatively subtract
        lost = []
        for t in old_windows:
            perm_counts[t] -= 1
            if perm_counts[t] == 0:
                lost.append(t)
        
        # If no lost unique perms, we might still be safe?
        # No, if a perm count goes to 0, it's lost unless it appears in a new window.
        
        # Remove w[i]
        orig = w[i]
        del w[i]
        
        # New windows are those that cross the deletion point.
        # The deletion is at index i in the old word.
        # In the new word, the gap is between i-1 and i.
        # Windows that cross this gap start at j such that j <= i-1 < j+n (in new word coords? No)
        # Let's think in terms of the new word `w_new`.
        # The characters at indices i-1 and i in `w_new` were i-1 and i+1 in `w_old`.
        # A window in `w_new` starting at `j` includes `w_new[i-1]` and `w_new[i]` if `j <= i-1 < j+n` and `j <= i < j+n`?
        # Actually, any window that contains the "seam" is new.
        # The seam is between old index i-1 and old index i+1.
        # In the new word, these are adjacent at indices i-1 and i.
        # Windows starting at `j` in the new word cover `w_new[j]` to `w_new[j+n-1]`.
        # They cover the seam if `j <= i-1` and `j+n-1 >= i`.
        # So `i - (n-1) <= j <= i-1`.
        
        new_starts = range(max(0, i - n + 1), i)
        gained = []
        for j in new_starts:
            if j + n <= len(w):
                t = tuple(w[j:j+n])
                if len(set(t)) == n:
                    perm_counts[t] += 1
                    if perm_counts[t] == 1:
                        gained.append(t)
        
        # Check if any lost permutation is not in gained
        valid = True
        for t in lost:
            if t not in gained:
                valid = False
                break
        
        if valid:
            # Keep deletion
            continue
        else:
            # Revert
            w.insert(i, orig)
            # Revert counts
            for j in new_starts:
                if j + n <= len(w):
                    t = tuple(w[j:j+n])
                    if len(set(t)) == n:
                        perm_counts[t] -= 1
            for t in old_windows:
                perm_counts[t] += 1
            
    return "".join(map(str, w))
# EVOLVE-BLOCK-END
