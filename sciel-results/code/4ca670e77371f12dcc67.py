import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if n == 1:
        return "1"
    
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
    target_count = len(target_perms)
    
    # Initialize counts of permutations in the current word
    perm_counts = {}
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p in target_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
            
    current_unique = len(perm_counts)
    
    # Function to check if deleting index i is valid
    # Returns True if valid, and updates perm_counts if so
    # If invalid, perm_counts is not modified (or we revert)
    def try_delete(idx):
        nonlocal perm_counts, current_unique
        
        # Windows affected are those starting from max(0, idx - n + 1) to min(idx, len(word) - n)
        start_range = max(0, idx - n + 1)
        end_range = min(idx, len(word) - n)
        
        if start_range > end_range:
            return True # No windows affected? Should not happen if idx < len(word)
            
        # Temporarily remove counts
        affected = []
        valid = True
        
        # We need to be careful: if multiple windows contain the same perm,
        # and we remove one, the count drops. If it hits 0, unique count drops.
        
        for i in range(start_range, end_range + 1):
            p = tuple(word[i:i + n])
            if p in target_perms:
                affected.append(p)
        
        # Check if any perm count would drop to 0
        # We need to know the current count
        # If count[p] == 1 and p is in affected, then unique count will drop.
        # Wait, if p appears in multiple affected windows, we subtract multiple times?
        # No, each window is distinct.
        
        # Let's just simulate the subtraction
        temp_counts = {}
        for p in affected:
            temp_counts[p] = temp_counts.get(p, 0) + 1
            
        for p, cnt in temp_counts.items():
            current_cnt = perm_counts.get(p, 0)
            if current_cnt - cnt == 0:
                return False # This perm would be lost
        
        # If we reach here, deletion is valid
        # Apply deletion
        for p, cnt in temp_counts.items():
            perm_counts[p] -= cnt
            if perm_counts[p] == 0:
                del perm_counts[p]
                current_unique -= 1
                
        word.pop(idx)
        return True

    # 3. Greedy Deletion Loop with Randomization
    # We perform passes. In each pass, we iterate through indices (possibly shuffled)
    # and try to delete.
    
    no_progress_count = 0
    max_no_progress = 50 # If we do 50 full passes with no deletions, stop or restart?
    # Actually, if no progress in a pass, we are in a local minimum.
    # We can try a different random order.
    
    while time.time() < deadline - 5:
        # Shuffle indices for randomization
        indices = list(range(len(word)))
        random.shuffle(indices)
        
        progress = False
        for idx in indices:
            if time.time() > deadline - 5:
                break
            
            # Index might be out of bounds if we deleted previous ones?
            # No, we are iterating a static list of indices from the start of the pass.
            # But if we delete an element, the indices shift.
            # This is a problem. If we delete index 5, then index 6 becomes 5.
            # The `indices` list is stale.
            
            # Better approach: Iterate from 0 to len(word)-1, but if we delete, 
            # we stay at the same index (to check the new character that shifted into place).
            # However, randomization is lost.
            
            # Alternative: Use a set of candidate indices to check.
            # Or, simpler: Do a sequential pass, but randomize the *start* or the *direction*?
            # Or, just do sequential passes. The "randomization" can be achieved by 
            # randomly permuting the word? No, that breaks validity.
            
            # Let's stick to sequential passes but with a "jitter".
            # Actually, the standard greedy deletion is:
            # for i in range(len(word)):
            #    if valid(i): delete(i)
            
            # To escape local minima, we can try to delete in reverse order, or 
            # we can "kick" the word by swapping two adjacent characters if valid?
            # Swapping is hard to check incrementally.
            
            # Let's try: Sequential deletion. If a full pass yields no deletions, 
            # we try a pass in reverse. If that yields no deletions, we try to 
            # "relax" by re-inserting a random character? No.
            
            # A better structural change: 
            # The classic construction has a lot of redundancy.
            # Greedy deletion is very effective.
            # Let's just do multiple sequential passes until stable.
            # To add variety, we can randomize the *initial* word? No, it must be valid.
            
            # Let's implement the sequential pass correctly.
            break # Break out of the random shuffle idea, do sequential
            
        # Sequential Pass
        i = 0
        while i < len(word) and time.time() < deadline - 5:
            if try_delete(i):
                progress = True
                # Do not increment i, check the new character at i
            else:
                i += 1
                
        if not progress:
            # Try reverse pass
            i = len(word) - 1
            while i >= 0 and time.time() < deadline - 5:
                if try_delete(i):
                    progress = True
                    # Do not decrement i? 
                    # If we delete i, the element at i-1 shifts to i-1? No.
                    # If we delete index i, indices < i are unaffected.
                    # So we should decrement i to check the new element at i-1?
                    # Actually, if we delete i, the list shrinks. 
                    # The element previously at i+1 is now at i.
                    # We want to check i again? No, we are going backwards.
                    # If we delete i, the new element at i is the one that was at i+1.
                    # So we should keep i the same?
                    # Let's trace: [A, B, C]. Delete B (i=1). List [A, C].
                    # We were at i=1. Now C is at i=1. We should check i=1 again.
                    pass
                else:
                    i -= 1
                    
            if not progress:
                # Stuck. Try a different strategy?
                # For now, break.
                break

    return "".join(map(str, word))
