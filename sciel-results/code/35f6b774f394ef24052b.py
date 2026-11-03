"""Improved superpermutation constructor with efficient local search."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations
from collections import defaultdict

def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        # Collect all distinct (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Maximum overlap between end of new and start of block
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _build_perm_map(n):
    """Build a mapping from permutation tuple to unique ID."""
    all_perms = list(permutations(range(1, n + 1)))
    # Sort to ensure consistent mapping (optional, but good for determinism)
    all_perms.sort()
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    id_to_perm = {i: p for i, p in enumerate(all_perms)}
    return perm_to_id, id_to_perm

def _optimized_delete(word, n, time_limit):
    """
    Efficiently delete characters using integer ID mapping for window counts.
    """
    start_time = time.time()
    word = list(word)
    L = len(word)
    num_perms = math.factorial(n)
    
    # Build mapping
    perm_to_id, id_to_perm = _build_perm_map(n)
    
    # Initialize counts and perm_ids
    counts = [0] * num_perms
    perm_ids = [0] * (L - n + 1)
    
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        pid = perm_to_id[w]
        perm_ids[i] = pid
        counts[pid] += 1
        
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try deleting from right to left to avoid index shifting issues in the same pass
        # Or just iterate and adjust. Right-to-left is safer for in-place pop.
        for i in range(L - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                return word
            if L <= n:
                break
            
            # Determine affected old window starts
            # Window at s covers i if s <= i <= s+n-1 => i-n+1 <= s <= i
            lo_old = max(0, i - n + 1)
            hi_old = min(i, L - n)
            
            # Determine affected new window starts
            # New word length is L-1.
            # New window at s covers the gap if s <= i-1 and s+n-1 >= i => i-n+1 <= s <= i-1
            # Also s must be valid: 0 <= s <= L-1-n
            lo_new = max(0, i - n + 1)
            hi_new = min(i - 1, L - 1 - n)
            
            # Simulate deletion
            # 1. Decrement old counts
            # We need to be careful: if a window is in both old and new lists (same ID), 
            # the net change might be 0. But it's easier to just apply changes and check min count.
            
            # To avoid modifying counts and then reverting if unsafe, 
            # we can track the minimum count change.
            # Or, we can just copy the counts? No, that's O(N).
            # Better: 
            #   - Decrement old counts.
            #   - Increment new counts.
            #   - If any count < 0, revert.
            #   - If safe, commit.
            
            # However, "new" windows are calculated from the *current* word state (before pop).
            # New window at s (s < i) content: word[s:i] + word[i+1:s+n]
            
            # Let's collect the changes
            old_decrements = []
            for s in range(lo_old, hi_old + 1):
                pid = perm_ids[s]
                old_decrements.append(pid)
                
            new_increments = []
            for s in range(lo_new, hi_new + 1):
                # Construct new window content
                # word[s:i] + word[i+1:s+n]
                # Note: s+n is the exclusive end in the *new* word?
                # New window at s has length n.
                # Elements: new_word[s], ..., new_word[s+n-1]
                # new_word[j] = word[j] for j < i
                # new_word[j] = word[j+1] for j >= i
                # So new_word[s:s+n] consists of:
                # - word[s:i] (length i-s)
                # - word[i+1:s+n] (length (s+n)-(i+1) = s+n-i-1)
                # Total length: i-s + s+n-i-1 = n-1. 
                # Wait.
                # If the window spans the gap, it contains i-s elements from left, and n-(i-s) elements from right?
                # Let k = i - s. (number of elements from left part)
                # Then we need n-k elements from right part.
                # Right part starts at i+1.
                # So right part is word[i+1 : i+1 + (n-k)] = word[i+1 : i+1 + n - (i-s)] = word[i+1 : s+n]
                # So content is word[s:i] + word[i+1:s+n]
                
                new_w = tuple(word[s:i] + word[i+1:s+n])
                # Check if this is a valid permutation (all distinct)
                # Since the original word was a superpermutation, and we are just removing a char,
                # the new window might contain duplicates.
                # If it contains duplicates, it's not a valid permutation, so it doesn't contribute to counts.
                # We only care about valid permutations.
                if len(set(new_w)) == n:
                    pid = perm_to_id.get(new_w, -1)
                    if pid != -1:
                        new_increments.append(pid)
                # If invalid, we don't add it to counts.
                
            # Apply changes to counts
            # We need to check if any count goes negative.
            # Since we are doing this in a loop, we can just apply, check, and revert.
            
            # Optimization: Only check counts that are changed.
            # But reverting requires knowing which ones were changed.
            
            # Let's just apply and check min.
            # To be safe and fast, we can do:
            # 1. Decrement old
            # 2. Increment new
            # 3. Check if any count in the *affected* set is < 0? 
            #    No, if a count was 1 and we decrement, it becomes 0. That's valid.
            #    If a count was 0 (shouldn't happen for valid superperm) and we decrement, it's -1.
            #    We need to ensure no count becomes -1? 
            #    Actually, we need to ensure that after deletion, every permutation still has count >= 1.
            #    So if a count drops to 0, it means that permutation is lost.
            #    So we check: for any pid in old_decrements, if counts[pid] - 1 < 1 (i.e., counts[pid] == 1), 
            #    then we must ensure that pid is in new_increments (so it gets incremented back to 1).
            #    If counts[pid] > 1, it's safe.
            
            # So:
            # unsafe = False
            # for pid in old_decrements:
            #     if counts[pid] == 1:
            #         if pid not in new_increments:
            #             unsafe = True
            #             break
            
            # This is O(n) per deletion attempt.
            
            unsafe = False
            # Create a set of new_increments for O(1) lookup
            # n is small, so list check is fine, but set is safer.
            new_set = set(new_increments)
            
            for pid in old_decrements:
                if counts[pid] == 1:
                    if pid not in new_set:
                        unsafe = True
                        break
            
            if unsafe:
                continue
            
            # Safe to delete
            # Perform deletion
            word.pop(i)
            L -= 1
            improved = True
            
            # Update counts
            for pid in old_decrements:
                counts[pid] -= 1
            for pid in new_increments:
                counts[pid] += 1
                
            # Update perm_ids
            # Old perm_ids length: L_old - n + 1
            # New perm_ids length: L_new - n + 1 = L_old - n
            # We need to update perm_ids for indices in [lo_new, hi_new]
            # And shift indices > i?
            # 
            # Let's just rebuild perm_ids for the affected range and shift the rest?
            # Shifting is O(L). 
            # Alternative: Keep perm_ids as a list and update it.
            # 
            # Indices < lo_old: unchanged.
            # Indices in [lo_old, hi_old]: these are the "old" windows that are gone or changed.
            # Indices > hi_old: these are shifted left by 1?
            # 
            # Let's trace indices.
            # Old word length L. New length L-1.
            # Old window at s corresponds to new window at:
            # - s if s < i-n+1 (no overlap)
            # - s if s >= i (shifted? No, s in new word corresponds to s+1 in old? 
            #   New window at s (s>=i) is old window at s+1.
            #   So old window at s+1 becomes new window at s.
            #   So old indices > i shift to index-1 in new word?
            #   Yes.
            #
            # So:
            # New perm_ids[k]:
            # - if k < lo_old: perm_ids[k]
            # - if lo_old <= k <= hi_new: calculated from new word
            # - if k > hi_new: 
            #   - if k < i: ?
            #   - if k >= i: perm_ids[k+1] (from old)
            #
            # This is getting complex. Given L is small (~6000), 
            # and we are doing this in a loop, 
            # it might be faster to just rebuild perm_ids for the whole array 
            # if the number of deletions per pass is small, 
            # OR to use a simpler data structure.
            #
            # Actually, we can just rebuild perm_ids entirely after each deletion?
            # O(L) per deletion. 6000 deletions * 6000 ops = 36M ops. 
            # In Python, this might be too slow for 120s if we do many passes.
            #
            # Let's try to update perm_ids incrementally.
            # 
            # We can use a list and slice assignment.
            # 
            # new_perm_ids = []
            # for k in range(L - n + 1):
            #     if k < lo_old:
            #         new_perm_ids.append(perm_ids[k])
            #     elif k <= hi_new:
            #         # Calculate from word
            #         w = tuple(word[k:k+n])
            #         new_perm_ids.append(perm_to_id[w])
            #     else:
            #         # k > hi_new
            #         # If k < i, it's in the gap? No, hi_new is i-1 max.
            #         # So k >= i.
            #         # Old index was k+1.
            #         new_perm_ids.append(perm_ids[k+1])
            # 
            # This is O(L).
            # 
            # To speed up, we can only update if L is small?
            # No, let's just do the O(L) update. It's simple and correct.
            
            # Rebuild perm_ids
            new_perm_ids = [0] * (L - n + 1)
            for k in range(L - n + 1):
                if k < lo_old:
                    new_perm_ids[k] = perm_ids[k]
                elif k <= hi_new:
                    w = tuple(word[k:k+n])
                    new_perm_ids[k] = perm_to_id[w]
                else:
                    # k > hi_new implies k >= i (since hi_new <= i-1)
                    # Old index k+1
                    new_perm_ids[k] = perm_ids[k+1]
            perm_ids = new_perm_ids
    
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    
    if remaining > 5:
        # Try optimized deletion
        word = _optimized_delete(word, n, remaining - 5)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
