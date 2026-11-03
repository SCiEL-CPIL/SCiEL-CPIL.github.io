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
    
    word = build_standard(n)
    w = list(word)
    
    # Fast greedy deletion
    # 1. Count occurrences of all valid permutations
    perm_counts = defaultdict(int)
    for i in range(len(w) - n + 1):
        t = tuple(w[i:i+n])
        if len(set(t)) == n:
            perm_counts[t] += 1
            
    # 2. Iterate through positions and try to delete
    # We iterate from left to right. 
    # Note: Deleting w[i] affects windows starting at i-n+1 to i.
    # We need to be careful with indices as the list shrinks.
    # It is safer to iterate from right to left, or handle indices carefully.
    # Let's iterate from right to left to avoid index shifting issues for the left part.
    # Actually, deleting w[i] shifts everything after i to the left.
    # If we go right-to-left, indices before i are unaffected.
    
    for i in range(len(w) - 1, 0, -1):
        # Check if deletion is safe
        # Windows affected: start indices k where k <= i <= k+n-1
        # i.e., k in [i-n+1, i]
        # We check if any of these windows is the *only* occurrence of its permutation.
        
        safe = True
        for k in range(max(0, i-n+1), min(i, len(w)-n) + 1):
            t = tuple(w[k:k+n])
            if len(set(t)) == n:
                if perm_counts[t] == 1:
                    safe = False
                    break
        
        if safe:
            # Perform deletion
            # 1. Decrement counts for removed windows
            for k in range(max(0, i-n+1), min(i, len(w)-n) + 1):
                t = tuple(w[k:k+n])
                if len(set(t)) == n:
                    perm_counts[t] -= 1
            
            # 2. Delete w[i]
            del w[i]
            
            # 3. Increment counts for new windows formed
            # New windows start at k in [i-n+1, i-1]
            # The window is w[k : i] + w[i : k+n] (since w[i] was deleted, the new index i is the old i+1)
            # Wait, after deletion, the element at index i is the old i+1.
            # The new windows are formed by combining the prefix ending at i-1 and suffix starting at i.
            for k in range(max(0, i-n+1), min(i, len(w)-n) + 1):
                # Check bounds for new window
                if k + n <= len(w):
                    t = tuple(w[k:k+n])
                    if len(set(t)) == n:
                        perm_counts[t] += 1
        # If not safe, do nothing
    
    return "".join(map(str, w))
# EVOLVE-BLOCK-END
