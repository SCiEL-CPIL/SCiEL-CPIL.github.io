import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115 # Safety margin

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
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    if n <= 6:
        return "".join(map(str, word))

    # 2. Setup for local search
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if p in perm_to_id:
            counts[perm_to_id[p]] += 1
            
    # Track zero counts for fast validity check
    zero_count = sum(1 for c in counts if c == 0)
    
    if zero_count > 0:
        return "".join(map(str, word))

    # Local Search: Deletion with fast validity check
    def try_delete(i):
        """
        Attempts to delete word[i]. Returns True if successful.
        Modifies word and counts if successful.
        """
        nonlocal zero_count
        L = len(word)
        if L == n:
            return False
        
        # List of (pid, delta) to apply
        updates = []
        
        # 1. Decrement old windows
        # Windows affected: start k in [i-n+1, i]
        for k in range(max(0, i - n + 1), min(i + 1, L - n + 1)):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                # Pre-check: if count is 1, it will become 0
                if counts[pid] == 1:
                    zero_count += 1
                counts[pid] -= 1
                updates.append((pid, -1))
        
        # 2. Increment new windows
        # New windows affected: start k in [i-n+1, i-1] in the NEW word
        # Note: The range of k in the new word is slightly different.
        # Old word length L. New word length L-1.
        # New windows start at k where new_word[k:k+n] is valid.
        # k ranges from 0 to (L-1)-n.
        # The deletion at i affects windows that span the gap.
        # In the new word, these are windows starting at k such that the window includes the position where i was removed.
        # This corresponds to new_word indices k in [i-n+1, i-1] roughly.
        
        for k in range(max(0, i - n + 1), min(i, L - n)): # k is start index in NEW word
            if k < i:
                # Window spans the deletion point
                # new_word[k] = word[k] ... word[i-1], word[i+1] ...
                # We need to construct the tuple
                # Optimization: use slicing
                w1 = word[k:i]
                w2 = word[i+1:k+n+1]
                p = tuple(w1 + w2)
            else: # k >= i
                # new_word[k] = word[k+1]
                p = tuple(word[k+1:k+1+n])
                
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 0:
                    zero_count -= 1
                counts[pid] += 1
                updates.append((pid, 1))
        
        # Check validity
        if zero_count == 0:
            word.pop(i)
            return True
        else:
            # Revert
            for pid, delta in updates:
                counts[pid] -= delta
            # Recalculate zero_count? 
            # We incremented zero_count for deletions that went 1->0.
            # We decremented zero_count for additions that went 0->1.
            # If we revert, we need to revert zero_count too.
            # It's safer to just recalculate or track carefully.
            # Let's just revert counts and recalculate zero_count if it fails?
            # No, we can just reverse the zero_count changes.
            # For each update (pid, delta):
            #   if delta == -1 (was decrement):
            #       if old count was 1, it became 0. We incremented zero_count.
            #       Now we revert: count goes 0->1. We should decrement zero_count.
            #   if delta == 1 (was increment):
            #       if old count was 0, it became 1. We decremented zero_count.
            #       Now we revert: count goes 1->0. We should increment zero_count.
            
            # Simpler: just re-calculate zero_count from counts?
            # O(N) is fast enough (5040).
            zero_count = sum(1 for c in counts if c == 0)
            return False

    # Iterate deletion passes
    improved = True
    while improved:
        improved = False
        # Left to right
        i = 0
        while i < len(word):
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_delete(i):
                improved = True
            else:
                i += 1
        # Right to left
        i = len(word) - 1
        while i >= 0:
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_delete(i):
                improved = True
            else:
                i -= 1

    # 3. Local Search: Swap (Optimized)
    def try_swap(i):
        nonlocal zero_count
        L = len(word)
        if i >= L - 1:
            return False
            
        updates = []
        
        # Decrement old windows
        for k in range(max(0, i - n + 1), min(i + 2, L - n + 1)):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 1:
                    zero_count += 1
                counts[pid] -= 1
                updates.append((pid, -1))
        
        # Perform swap
        word[i], word[i+1] = word[i+1], word[i]
        
        # Increment new windows
        for k in range(max(0, i - n + 1), min(i + 2, L - n + 1)):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 0:
                    zero_count -= 1
                counts[pid] += 1
                updates.append((pid, 1))
        
        if zero_count == 0:
            # Valid swap. Now try to delete to see if it helps.
            saved_word = word[:]
            saved_zero = zero_count
            
            # Try delete i
            if try_delete(i):
                return True
            # Try delete i+1 (index might have shifted if delete i succeeded, but we returned)
            # If delete i failed, word is unchanged (post-swap).
            # try_delete reverts itself if it fails.
            # So state is post-swap.
            if try_delete(i+1):
                return True
            
            # No deletion possible. Revert swap.
            word[:] = saved_word
            zero_count = saved_zero
            # Revert counts? try_delete reverted its own changes.
            # The swap changes are still applied. We need to revert them.
            # The `updates` list contains the swap changes.
            for pid, delta in updates:
                counts[pid] -= delta
            # Recalculate zero_count
            zero_count = sum(1 for c in counts if c == 0)
            return False
        else:
            # Invalid swap. Revert.
            word[i], word[i+1] = word[i+1], word[i]
            for pid, delta in updates:
                counts[pid] -= delta
            zero_count = sum(1 for c in counts if c == 0)
            return False

    # Iterate swap passes
    improved = True
    while improved:
        improved = False
        i = 0
        while i < len(word) - 1:
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_swap(i):
                improved = True
            else:
                i += 1

    return "".join(map(str, word))
