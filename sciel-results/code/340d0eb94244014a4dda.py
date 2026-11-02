import time
import itertools
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate classic superpermutation
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Prepare for Greedy Deletion
    target_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts of permutations in the current word
    perm_counts = {}
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p in target_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
    
    best_word = list(word)
    best_len = len(word)
    
    # Function to check if deleting index i is valid
    def try_delete(idx, word, perm_counts, n, target_perms):
        # Windows affected are those starting from max(0, idx - n + 1) to min(idx, len(word) - n)
        start_range = max(0, idx - n + 1)
        end_range = min(idx, len(word) - n)
        
        if start_range > end_range:
            return False
        
        # Check if any perm count would drop to 0
        # We need to know which perms are affected and their current counts
        # If any affected perm has count 1, deletion is invalid
        for i in range(start_range, end_range + 1):
            p = tuple(word[i:i + n])
            if p in target_perms:
                if perm_counts.get(p, 0) <= 1:
                    return False
        
        # Apply deletion
        for i in range(start_range, end_range + 1):
            p = tuple(word[i:i + n])
            if p in target_perms:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    del perm_counts[p]
        
        word.pop(idx)
        return True

    # Function to check if swapping i and i+1 is valid
    def try_swap(idx, word, perm_counts, n, target_perms):
        if idx >= len(word) - 1:
            return False
        
        # Windows affected by swap at idx, idx+1
        # Start range: max(0, idx - n + 1) to min(idx+1, len(word) - n)
        start_range = max(0, idx - n + 1)
        end_range = min(idx + 1, len(word) - n)
        
        # Collect affected perms and their counts
        affected = {}
        for i in range(start_range, end_range + 1):
            p = tuple(word[i:i + n])
            if p in target_perms:
                affected[p] = affected.get(p, 0) + 1
        
        # Check if any count would drop to 0
        # Note: The swap changes the content of these windows.
        # We must calculate the NEW perms and check if they are valid permutations?
        # No, we just need to ensure that after swap, all perms still appear.
        # This is harder to check efficiently.
        # Simplified: Only allow swap if it doesn't create a NEW unique perm that wasn't there?
        # Or just check if the set of perms in the affected windows remains the same?
        # If the multiset of perms in the affected windows is the same, validity is preserved.
        
        new_word = word[:]
        new_word[idx], new_word[idx+1] = new_word[idx+1], new_word[idx]
        
        new_affected = {}
        for i in range(start_range, end_range + 1):
            p = tuple(new_word[i:i + n])
            if p in target_perms:
                new_affected[p] = new_affected.get(p, 0) + 1
        
        # If the multisets are identical, the swap is safe (counts don't change globally for these perms, 
        # but wait, other windows outside this range might be affected? No, swap only affects windows containing idx or idx+1.
        # If the multiset of perms in the affected windows is the same, then the global counts for those perms remain the same.
        # However, we must ensure that no NEW perm is introduced that has count 0? 
        # If the multiset is the same, no new perm is introduced.
        
        if affected == new_affected:
             # Apply swap
             word[idx], word[idx+1] = word[idx+1], word[idx]
             return True
        return False

    # Forward greedy deletion pass
    def forward_pass(word, perm_counts, n, target_perms, deadline):
        i = 0
        progress = False
        while i < len(word) and time.time() < deadline:
            if try_delete(i, word, perm_counts, n, target_perms):
                progress = True
            else:
                i += 1
        return progress
    
    # Backward greedy deletion pass
    def backward_pass(word, perm_counts, n, target_perms, deadline):
        i = len(word) - 1
        progress = False
        while i >= 0 and time.time() < deadline:
            if try_delete(i, word, perm_counts, n, target_perms):
                progress = True
            else:
                i -= 1
        return progress

    # Random deletion pass
    def random_pass(word, perm_counts, n, target_perms, deadline):
        indices = list(range(len(word)))
        random.shuffle(indices)
        progress = False
        for i in indices:
            if time.time() > deadline:
                break
            # Index i might be invalid if word changed, but we are deleting from a snapshot?
            # No, we are modifying word in place.
            # If we delete an index, subsequent indices in the list are shifted.
            # This makes random pass tricky.
            # Better: iterate through a copy of indices, but check if index < len(word)
            # and re-verify validity.
            if i < len(word):
                if try_delete(i, word, perm_counts, n, target_perms):
                    progress = True
        return progress

    # Main optimization loop
    while time.time() < deadline - 2:
        # Try forward pass
        if forward_pass(word, perm_counts, n, target_perms, deadline - 2):
            if len(word) < best_len:
                best_word = list(word)
                best_len = len(word)
        
        # Try backward pass
        if backward_pass(word, perm_counts, n, target_perms, deadline - 2):
            if len(word) < best_len:
                best_word = list(word)
                best_len = len(word)
        
        # Try random pass (less frequent)
        if random.random() < 0.1:
             if random_pass(word, perm_counts, n, target_perms, deadline - 2):
                 if len(word) < best_len:
                     best_word = list(word)
                     best_len = len(word)

    return "".join(map(str, best_word))
