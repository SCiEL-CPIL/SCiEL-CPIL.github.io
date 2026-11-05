import itertools
import time
import heapq
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # For n < 7, use the standard construction
    if n < 7:
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

    # For n = 7, we try to find a shorter superpermutation
    # The standard construction gives length 5905
    # We'll use a heuristic approach: start with the standard construction,
    # then try to remove redundant characters.

    # Generate the standard superpermutation for n=7
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

    w = word[:]
    w_len = len(w)
    
    # Precompute all permutations as a set for fast lookup
    all_perms = set(itertools.permutations(range(1, n + 1)))
    total_perms = len(all_perms)
    
    # Count occurrences of each permutation in the current word
    perm_counts = {}
    for i in range(w_len - n + 1):
        p = tuple(w[i:i + n])
        if p in all_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
    
    # Check if all permutations are present
    if len(perm_counts) < total_perms:
        return "".join(map(str, w))
    
    # Strategy: Greedy removal of redundant characters
    # We scan from left to right and try to remove characters that are redundant.
    # A character at position i is redundant if removing it doesn't destroy any
    # permutation that only appears in windows containing that character.
    
    # To be efficient, we'll try multiple passes.
    # In each pass, we scan left to right and remove any character that is safe to remove.
    
    def can_remove(i):
        """Check if character at position i can be removed."""
        # Windows affected by removing w[i]:
        # Old windows starting at i-n+1, ..., i
        for start in range(max(0, i - n + 1), min(i + 1, w_len - n + 1)):
            p = tuple(w[start:start + n])
            if p in all_perms:
                if perm_counts.get(p, 0) == 1:
                    return False
        return True
    
    def remove_char(i):
        """Remove character at position i and update perm_counts."""
        nonlocal w_len
        # Decrement counts for windows that contained w[i]
        for start in range(max(0, i - n + 1), min(i + 1, w_len - n + 1)):
            p = tuple(w[start:start + n])
            if p in all_perms:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    del perm_counts[p]
        
        # Remove the character
        w.pop(i)
        w_len -= 1
        
        # After removal, new windows may form.
        # Specifically, windows starting at positions i-n+1, ..., i-1 in the NEW string
        # are different from the old windows at those positions.
        # We need to add counts for these new windows.
        for start in range(max(0, i - n + 1), min(i, w_len - n + 1)):
            p = tuple(w[start:start + n])
            if p in all_perms:
                perm_counts[p] = perm_counts.get(p, 0) + 1
    
    # Do multiple passes of greedy removal
    max_passes = 50
    for _ in range(max_passes):
        if time.time() > deadline - 5:
            break
        
        i = 0
        removed_any = False
        while i < w_len - n + 1:
            if time.time() > deadline - 5:
                break
            
            if can_remove(i):
                remove_char(i)
                removed_any = True
                # Don't increment i, check the same position again
            else:
                i += 1
        
        if not removed_any:
            break
    
    # Additional pass: try removing from the end
    while w_len > n:
        if time.time() > deadline - 2:
            break
        
        # Check if the last character can be removed
        # The last window is w[w_len-n : w_len]
        last_start = w_len - n
        p = tuple(w[last_start:])
        if p in all_perms and perm_counts.get(p, 0) == 1:
            break
        else:
            # Safe to remove
            if p in all_perms:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    del perm_counts[p]
            w.pop()
            w_len -= 1
    
    return "".join(map(str, w))
