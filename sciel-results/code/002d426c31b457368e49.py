import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    # If n is small, return immediately
    if n < 7:
        return "".join(map(str, word))
        
    # For n=7, try to optimize
    # Convert to list of ints for easy manipulation
    w = word
    
    # Pre-compute all permutations
    all_perms = set(itertools.permutations(range(1, n+1)))
    total_perms = len(all_perms)
    
    # Count occurrences of each permutation in the current word
    perm_counts = {}
    for i in range(len(w) - n + 1):
        p = tuple(w[i:i+n])
        if p in all_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
            
    # Check if all are present (should be)
    if len(perm_counts) < total_perms:
        return "".join(map(str, w))
        
    # Helper function to check if a position is removable
    def is_removable(w, i, perm_counts, n, all_perms):
        """Check if character at position i can be removed without losing any permutation."""
        for start in range(max(0, i-n+1), min(i+1, len(w)-n+1)):
            p = tuple(w[start:start+n])
            if p in all_perms:
                if perm_counts.get(p, 0) == 1:
                    return False
        return True
    
    # Helper function to remove a character and update counts
    def remove_char(w, i, perm_counts, n, all_perms):
        """Remove character at position i and update permutation counts."""
        # Decrement counts for old windows containing w[i]
        for start in range(max(0, i-n+1), min(i+1, len(w)-n+1)):
            p = tuple(w[start:start+n])
            if p in all_perms:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    del perm_counts[p]
        
        # Remove the character
        w.pop(i)
        
        # Increment counts for new windows
        for start in range(max(0, i-n+1), i):
            if start + n <= len(w):
                p = tuple(w[start:start+n])
                if p in all_perms:
                    perm_counts[p] = perm_counts.get(p, 0) + 1
    
    # Greedy removal with correct count updates
    # We will scan from left to right.
    # When we remove w[i], we update counts for the affected windows.
    # Affected old windows: starts i-n+1 ... i
    # Affected new windows: starts i-n+1 ... i-1 (in the new string, these windows are shifted/changed)
    
    i = 0
    while i < len(w) - n + 1:
        if time.time() > deadline:
            break
            
        # Check if w[i] is removable
        if is_removable(w, i, perm_counts, n, all_perms):
            remove_char(w, i, perm_counts, n, all_perms)
        else:
            i += 1
            
    # "Remove from End" pass
    while len(w) > n:
        if time.time() > deadline:
            break
        # Check last window
        last_start = len(w) - n
        p = tuple(w[last_start:])
        if p in all_perms and perm_counts.get(p, 0) == 1:
            # Can't remove last char
            break
        else:
            # Safe to remove?
            # Removing last char destroys the last window.
            # If it was a permutation, decrement count.
            if p in all_perms:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    del perm_counts[p]
            w.pop()
    
    # Additional optimization passes
    # Try multiple passes of greedy removal in different orders
    for pass_num in range(10):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Forward pass
        i = 0
        while i < len(w) - n + 1:
            if time.time() > deadline:
                break
            if is_removable(w, i, perm_counts, n, all_perms):
                remove_char(w, i, perm_counts, n, all_perms)
                improved = True
            else:
                i += 1
        
        # Backward pass
        i = len(w) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            if is_removable(w, i, perm_counts, n, all_perms):
                remove_char(w, i, perm_counts, n, all_perms)
                improved = True
            i -= 1
        
        if not improved:
            break
    
    return "".join(map(str, w))
