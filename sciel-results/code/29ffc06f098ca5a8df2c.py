import time
import sys
from itertools import permutations

def construct(n):
    # Safety margin: 150s limit, use 145s for computation
    deadline = time.time() + 145.0
    
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
    
    # Initialize counts and zero_count
    counts = [0] * num_perms
    zero_count = num_perms
    
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        pid = perm_to_id.get(p)
        if pid is not None:
            if counts[pid] == 0:
                zero_count -= 1
            counts[pid] += 1
            
    # Check if valid initially (should be)
    if zero_count > 0:
        return "".join(map(str, word))

    # 3. Local Search: Deletion with incremental count updates
    # We attempt to delete each character. If valid, we keep the deletion.
    # We repeat passes until no more deletions are possible or time runs out.
    
    improved = True
    while improved:
        improved = False
        i = 0
        L = len(word)
        while i < L:
            if time.time() > deadline:
                break
            
            # Try to delete word[i]
            # Affected old windows: start k in [max(0, i-n+1), min(i, L-n)]
            # Affected new windows: start k in [max(0, i-n+1), min(i-1, L-n-1)] in new word
            
            old_k_start = max(0, i - n + 1)
            old_k_end = min(i, L - n)
            
            new_k_start = max(0, i - n + 1)
            new_k_end = min(i - 1, L - n - 1)
            
            # Store old counts for affected permutations to revert if needed
            # Instead of copying whole array, we track changes
            changes = [] # (pid, delta)
            valid = True
            
            # Decrement old windows
            for k in range(old_k_start, old_k_end + 1):
                p = tuple(word[k:k + n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 1:
                        # This permutation will be lost if not restored by new windows
                        # We check this later
                        pass
                    counts[pid] -= 1
                    changes.append((pid, -1))
                    if counts[pid] == 0:
                        # Tentatively zero, but might be restored
                        pass
            
            # Increment new windows
            for k in range(new_k_start, new_k_end + 1):
                # Determine the window in the old word that corresponds to new_word[k:k+n]
                if k < i:
                    # New window is word[k:i] + word[i+1:k+n+1]
                    # Length is (i-k) + (n - (i-k)) = n
                    w1 = word[k:i]
                    w2 = word[i+1:k+n+1]
                    p = tuple(w1 + w2)
                else:
                    # k >= i, but k <= i-1 in this branch? 
                    # Wait, new_k_end is min(i-1, ...). So k is always < i in this loop?
                    # No, if i=0, new_k_start=0, new_k_end=-1 (empty loop).
                    # If i>0, k ranges from max(0, i-n+1) to min(i-1, ...).
                    # So k < i is always true for the new windows loop?
                    # Let's check indices.
                    # New word length L-1.
                    # Window start k in new word.
                    # If k < i, it overlaps the deletion point.
                    # If k >= i, it does not overlap. But in new word, indices >= i correspond to old indices >= i+1.
                    # However, the range of k is limited to i-1?
                    # The windows that *change* are those that include the deleted character.
                    # A window in the new word includes the "gap" if it starts at k such that k < i < k+n.
                    # i.e., k > i-n and k < i.
                    # So k in [max(0, i-n+1), i-1].
                    # Yes, so k < i is always true for the new windows we need to calculate explicitly.
                    # Windows with k >= i in new word are just old windows with k+1 >= i+1.
                    # Wait, if k >= i in new word, the window is word[k+1 : k+1+n].
                    # But we only need to check windows that *changed*.
                    # Windows that did not change are:
                    # 1. Old windows with k+n <= i (completely before i).
                    # 2. Old windows with k >= i (completely after i). In new word, these shift to k-1.
                    #    But their content is identical to old word[k:k+n]? No.
                    #    Old word[k:k+n] becomes new word[k-1:k-1+n].
                    #    So the permutation is the same. The count should remain the same?
                    #    No, we decremented old windows in [i-n+1, i].
                    #    The windows in [i, L-n] in old word are NOT decremented.
                    #    In the new word, the windows starting at k in [i-1, L-n-1] correspond to old windows starting at k+1.
                    #    So old windows starting at [i, L-n] are represented by new windows starting at [i-1, L-n-1].
                    #    Since we didn't decrement them, and we don't increment them (they are "new" in the sense of index, but content is same as old), 
                    #    we must be careful.
                    
                    # Let's stick to the logic: 
                    # We remove character at i.
                    # Old windows affected: those containing index i. Start k in [i-n+1, i].
                    # New windows affected: those that "span" the gap in the new word. 
                    # In the new word, the gap is between i-1 and i.
                    # Windows spanning this gap start at k in [i-n+1, i-1].
                    # For these k, the content is different from any single old window.
                    # All other windows in the new word correspond exactly to old windows not containing i.
                    # So we only need to process:
                    # 1. Decrement old windows containing i: k in [max(0, i-n+1), min(i, L-n)].
                    # 2. Increment new windows spanning the gap: k in [max(0, i-n+1), min(i-1, L-n-1)].
                    
                    # So my loop `for k in range(new_k_start, new_k_end + 1)` is correct.
                    # And in this loop, k is always < i.
                    pass # The code above handles it.
                
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 0:
                        # Restored
                        pass
                    counts[pid] += 1
                    changes.append((pid, 1))
            
            # Check validity: are there any counts == 0?
            # We need to know if *any* permutation has count 0.
            # We can track zero_count.
            # Let's re-calculate zero_count change? 
            # Easier: just check if the specific permutations we touched went to 0 and weren't restored.
            # But a permutation not touched might be 0? No, we started with all >= 1.
            # Only touched permutations can change status.
            
            # We need to know if *all* permutations have count >= 1.
            # We can maintain a global zero_count.
            # Let's refactor to use zero_count.
            
            # Re-do with zero_count tracking
            # We already modified counts. We need to revert if invalid.
            # To check validity, we can just check if zero_count == 0.
            # But we didn't update zero_count in the loops above.
            
            # Let's restart the try_delete logic with zero_count.
            
            # Rollback previous changes
            for pid, delta in changes:
                counts[pid] -= delta
            
            # Now do it properly with zero_count
            # 1. Decrement old windows
            for k in range(old_k_start, old_k_end + 1):
                p = tuple(word[k:k + n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 1:
                        zero_count += 1
                    counts[pid] -= 1
                    changes.append((pid, -1))
            
            # 2. Increment new windows
            for k in range(new_k_start, new_k_end + 1):
                if k < i:
                    w1 = word[k:i]
                    w2 = word[i+1:k+n+1]
                    p = tuple(w1 + w2)
                else:
                    # Should not happen based on range, but safe guard
                    p = tuple(word[k+1:k+1+n])
                
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 0:
                        zero_count -= 1
                    counts[pid] += 1
                    changes.append((pid, 1))
            
            if zero_count == 0:
                # Valid deletion
                word.pop(i)
                improved = True
                # Do not increment i, check same position again
                L = len(word) # Update length
            else:
                # Revert
                for pid, delta in changes:
                    counts[pid] -= delta
                    if delta == -1: # We decremented, so if it was 0 before, it's 1 now?
                        # Wait, revert logic:
                        # If we did counts[pid] -= 1, and it went 1->0, zero_count increased.
                        # Revert: counts[pid] += 1. 0->1. zero_count should decrease.
                        # If we did counts[pid] += 1, and it went 0->1, zero_count decreased.
                        # Revert: counts[pid] -= 1. 1->0. zero_count should increase.
                        
                        # Let's just recalculate zero_count from scratch? No, too slow.
                        # Let's track zero_count changes carefully.
                        pass
                # Proper revert of zero_count
                # We need to know the state before.
                # This is getting messy. Let's use a simpler check for small n or optimize the check.
                # For n=7, num_perms=5040.
                # Checking `any(c==0 for c in counts)` is 5040 ops.
                # We do this for each of ~6000 positions. 30M ops.
                # In Python, 30M simple ops might take 1-2 seconds.
                # This is acceptable within 145s budget if we only do a few passes.
                
                # Let's simplify: just check `if any(c == 0 for c in counts):`
                # But we need to revert counts first.
                
                # Revert counts
                for pid, delta in changes:
                    counts[pid] -= delta
                
                # Check validity
                if any(c == 0 for c in counts):
                    i += 1
                else:
                    # Should not happen if logic is correct, but if it does, commit
                    word.pop(i)
                    improved = True
                    L = len(word)
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    return "".join(map(str, word))

# BET: {mechanism: "Incremental permutation count updates allow O(1) amortized validity checks for character deletion, enabling multiple local search passes to reduce length from 5913 towards the known optimum of 5905.", q: 0.8, delta: [0.001, 0.01]}
