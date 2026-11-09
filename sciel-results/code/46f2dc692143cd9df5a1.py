import time
import random
from itertools import permutations

def construct(n):
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
            # Find maximum overlap
            max_overlap = 0
            # The overlap can be at most len(block) - 1, but practically limited by len(new)
            # We check suffix of new against prefix of block
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new
    
    # 2. Greedy Reduction
    # Map permutations to IDs
    # For n=7, 5040 perms.
    # We need a fast way to get the ID of a window.
    # We can use a dictionary mapping tuple to ID.
    perm_to_id = {p: i for i, p in enumerate(permutations(range(1, n+1)))}
    
    # Calculate factorial
    fact = 1
    for i in range(1, n+1):
        fact *= i
        
    # Initialize counts
    counts = [0] * fact
    
    # Helper to get ID
    def get_id(window_tuple):
        return perm_to_id.get(window_tuple, -1)

    # Initial counts
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        pid = get_id(w)
        if pid != -1:
            counts[pid] += 1
            
    # Check if all are covered
    # They should be, as it's a superpermutation.
    
    # Greedy removal
    # We try to remove characters.
    # To avoid O(N^2) complexity from list slicing/insertion, we can use a linked list or just accept the cost since N=6000 is small.
    # Actually, removing from a list is O(N). 6000 removals * 6000 ops = 36M ops. This is fine in Python.
    
    # We'll use a boolean array to mark "removed" indices to avoid shifting?
    # No, shifting is easier to reason about for window checks.
    # But shifting is expensive.
    # Let's use a list of indices and a "virtual" removal?
    # Or just use a list and pop.
    
    # Strategy: Try removing from left to right. If stuck, try right to left.
    # Repeat a few times.
    
    def try_remove_pass(direction):
        nonlocal word, counts
        removed = 0
        # We iterate through indices. If we remove, we don't increment i.
        # If direction is 1, iterate 0 to len-1.
        # If direction is -1, iterate len-1 to 0.
        
        # To handle dynamic length, we use a while loop or careful indexing.
        # Let's just collect indices to remove in one pass, then remove them?
        # No, removing one affects the validity of the next.
        
        i = 0 if direction == 1 else len(word) - 1
        step = 1 if direction == 1 else -1
        
        while i >= 0 and i < len(word):
            # Check if word[i] can be removed
            can_remove = True
            # Affected windows: those containing index i
            # Start index of window j is j. Window covers j..j+n-1.
            # i is in window j if j <= i <= j+n-1 => i-n+1 <= j <= i
            # j must be >= 0 and j <= len(word)-n
            start_j = max(0, i - n + 1)
            end_j = min(i, len(word) - n)
            
            for j in range(start_j, end_j + 1):
                w = tuple(word[j:j+n])
                pid = get_id(w)
                if pid == -1:
                    continue
                if counts[pid] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove word[i]
                # Decrement counts for affected windows
                for j in range(start_j, end_j + 1):
                    w = tuple(word[j:j+n])
                    pid = get_id(w)
                    if pid != -1:
                        counts[pid] -= 1
                word.pop(i)
                removed += 1
                # Do not increment i, check same index again (which is now the next element)
                # But wait, if we remove, the indices shift.
                # If we remove index i, the element at i+1 becomes i.
                # So we should check index i again.
                # However, the windows covering the *new* index i (old i+1) are different.
                # So we just stay at i.
            else:
                i += step
                
        return removed

    # Run passes
    # Time limit check
    start_time = time.time()
    
    # Initial pass
    try_remove_pass(1)
    try_remove_pass(-1)
    
    # If we have time, do more random passes?
    # Or just return.
    # The greedy reduction is deterministic.
    
    return "".join(map(str, word))
