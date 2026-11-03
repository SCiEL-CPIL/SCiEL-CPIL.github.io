import sys
import time
import itertools
import math
import random
from collections import defaultdict

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

def construct(n):
    if n <= 0:
        return ""
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for safety
    
    # 1. Generate Classic Superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        target_count = 1
        for i in range(1, m):
            target_count *= i
            
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
                    if len(perms) == target_count:
                        break
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[-o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        word = new_word
        
    # 2. Fast Validity Check and Deletion Heuristic
    # We maintain a dictionary of counts for each permutation window.
    # A window is a tuple of length n.
    # We only care about windows that are permutations (all distinct elements from 1..n).
    
    def get_window_permutation(w, i, n):
        """Returns tuple if w[i:i+n] is a permutation of 1..n, else None"""
        # Optimization: check sum first? No, set check is fast enough for small n.
        # But we can pre-check length and uniqueness.
        window = w[i:i+n]
        if len(set(window)) == n:
            return tuple(window)
        return None

    def build_counts(w, n):
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            perm = get_window_permutation(w, i, n)
            if perm:
                counts[perm] += 1
        return counts

    def is_valid_counts(counts, n):
        return len(counts) == factorial(n)

    # Initial counts
    counts = build_counts(word, n)
    if not is_valid_counts(counts, n):
        # Fallback: should not happen
        return "".join(map(str, word))
    
    best_word = word[:]
    best_len = len(word)
    
    # Iterative Deletion
    # We try to delete characters.
    # To make this efficient, we don't rebuild counts from scratch every time.
    # However, updating counts incrementally when deleting a character at index i
    # is complex because windows shift.
    
    # Alternative: 
    # 1. Try deleting index i.
    # 2. Check if the new word is valid.
    # 3. If valid, accept.
    
    # To speed up validity check:
    # Instead of full rebuild, we can note that only windows overlapping i are affected.
    # But indices shift for j > i.
    
    # Let's try a "sliding window" update approach or just optimized full check.
    # For n=7, L~6000.
    # Full check: 6000 * (7 ops for set + hash) ~ 42k ops.
    # 6000 positions * 42k ops = 252M ops.
    # In Python, this is ~10-20s per pass.
    # We have 110s. So we can do ~5-10 passes.
    
    # Optimization: Precompute all permutations as a set for O(1) lookup?
    # There are 5040 permutations.
    # We can map each permutation to an integer ID to speed up hashing?
    # Or just use tuples. Tuples of small ints hash fast.
    
    # Let's try to optimize the inner loop of the validity check.
    # We can use a boolean array or set for "seen" and stop early if we find all.
    
    all_perms_set = set(itertools.permutations(range(1, n+1)))
    target_count = len(all_perms_set)
    
    def check_validity_fast(w, n, all_perms_set):
        seen = set()
        L = len(w)
        for i in range(L - n + 1):
            # Check if w[i:i+n] is in all_perms_set
            # We can try to build the tuple and check membership.
            # But building the tuple takes time.
            # Is there a faster way?
            # Maybe precompute a lookup table for substrings? No, L is large.
            
            # Let's just use the set check.
            window = w[i:i+n]
            # Quick check: if any element is out of range or duplicate?
            # Actually, `if tuple(window) in all_perms_set` is quite fast in CPython.
            if tuple(window) in all_perms_set:
                seen.add(tuple(window)) # Avoid creating tuple twice?
                # Wait, `tuple(window) in all_perms_set` creates a tuple.
                # If it's in the set, we add it.
                # We can do:
                # t = tuple(window)
                # if t in all_perms_set: seen.add(t)
                # This creates the tuple once.
                
                if len(seen) == target_count:
                    return True
        return len(seen) == target_count

    # Even better:
    # We know the word is a superpermutation initially.
    # When we delete a character, we only need to check if we lost any permutation.
    # We can maintain the counts.
    # Let's try to implement incremental count updates.
    
    # When deleting index i:
    # Windows affected:
    # 1. Windows starting at j in [i-n+1, i] that contain i.
    #    These windows are removed or changed.
    # 2. Windows starting at j > i shift to j-1.
    
    # This is tricky to handle with a simple dictionary of counts because the keys (tuples) change for shifted windows?
    # No, the content of the window changes if it includes the deleted element.
    # If a window does NOT include the deleted element, its content stays the same, but its index shifts.
    # Since we only care about the SET of permutations present (and counts if needed for "lost" detection),
    # and we are deleting, we only care if a permutation count drops to 0.
    
    # Let's stick to the "try delete, check validity" approach but optimize the check.
    # We can cache the last validity result? No.
    
    # Let's try a different strategy:
    # Local search with random restarts? No, deterministic is better for reproducibility and speed.
    
    # Let's optimize `check_validity_fast` by using a list of booleans indexed by permutation ID?
    # Map each permutation to an integer 0..5039.
    # Then use a bytearray or list of ints for counts.
    
    perm_to_id = {p: i for i, p in enumerate(all_perms_set)}
    
    def check_validity_with_ids(w, n, perm_to_id, target_count):
        counts = [0] * target_count
        L = len(w)
        for i in range(L - n + 1):
            window = w[i:i+n]
            # Convert to tuple
            t = tuple(window)
            if t in perm_to_id:
                pid = perm_to_id[t]
                counts[pid] += 1
                # Early exit if we see all? 
                # We can't easily early exit if we need to check if ANY count is 0 at the end.
                # But if we just want to know if it's valid, we need all counts > 0.
                # We can track how many unique perms we've seen.
                # If we see a new one, increment unique_count.
                # If unique_count == target_count, we still need to finish to ensure no count drops?
                # No, if we see all 5040 unique perms, then all counts are >= 1.
                # So we CAN early exit!
        # Wait, if we early exit, we don't know if some perm appeared 0 times?
        # No, if we have seen `target_count` unique permutations, and there are only `target_count` possible permutations,
        # then every permutation must have been seen at least once.
        # So yes, early exit is valid.
        
        # Let's re-implement with early exit.
        seen = [False] * target_count
        unique_count = 0
        for i in range(L - n + 1):
            t = tuple(w[i:i+n])
            if t in perm_to_id:
                pid = perm_to_id[t]
                if not seen[pid]:
                    seen[pid] = True
                    unique_count += 1
                    if unique_count == target_count:
                        return True
        return False

    # Now, the deletion loop.
    # We scan left to right. If deletion is valid, we delete and restart?
    # Restarting is O(L^2) in worst case but finds more deletions.
    # Given the budget, let's try a few full passes.
    
    max_passes = 20
    for p in range(max_passes):
        if time.time() > deadline:
            break
            
        changed = False
        i = 0
        while i < len(word):
            if time.time() > deadline:
                break
                
            # Try deleting word[i]
            new_word = word[:i] + word[i+1:]
            
            # Check validity
            if check_validity_with_ids(new_word, n, perm_to_id, target_count):
                word = new_word
                changed = True
                # Restart scan from beginning to catch earlier deletions
                i = 0
            else:
                i += 1
                
        if not changed:
            break
        if len(word) < best_len:
            best_len = len(word)
            best_word = word[:]
            
    # If we have time, try pair deletions?
    # Or just return the best found.
    
    return "".join(map(str, word))
