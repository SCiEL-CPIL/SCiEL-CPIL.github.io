import time
import random
import heapq
from collections import defaultdict
import itertools
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, 35s safety margin
    
    # 1. Recursive Construction (Standard method)
    # This generates a word of length 1 + 2! + 3! + ... + n!
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique (m-1)-permutations from the current word
        # The standard construction ensures we find all (m-1)! permutations
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Block: p + [m] + p
            # But we want to overlap as much as possible
            block = list(p) + [m] + list(p)
            # Find max overlap with end of new
            overlap = 0
            # Max possible overlap is len(block)-1, but practically limited by len(new)
            max_ov = min(len(new), len(block) - 1)
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Initialize counts
    perm_count = defaultdict(int)
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        perm_count[p] += 1
    
    # Helper to check if a position is deletable
    # A position i is deletable if all windows containing i have count > 1
    # Windows containing i are j in [i-n+1, i]
    
    def can_delete(i):
        L = len(word)
        if L <= n:
            return False
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        for j in range(start_j, end_j + 1):
            if perm_count[tuple(word[j:j+n])] == 1:
                return False
        return True

    def delete(i):
        L = len(word)
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        
        # Decrement old windows
        for j in range(start_j, end_j + 1):
            p = tuple(word[j:j+n])
            perm_count[p] -= 1
        
        # Commit deletion
        word.pop(i)
        
        # Increment new windows
        # New length L-1
        # Windows spanning the gap in the new word are at indices j' in [max(0, i-n+1), min(i, L-1-n)]
        start_j_new = max(0, i - n + 1)
        end_j_new = min(i, L - 1 - n)
        
        for j in range(start_j_new, end_j_new + 1):
            # Construct new window
            # Note: indices in word have shifted for k >= i
            new_w = []
            for k in range(n):
                idx = j + k
                if idx < i:
                    new_w.append(word[idx])
                else:
                    new_w.append(word[idx+1])
            p_new = tuple(new_w)
            perm_count[p_new] += 1

    # Phase 1: Greedy Deletion
    # We repeatedly try to delete positions.
    # To be efficient, we can iterate through the word and delete if possible.
    # Since deletion changes indices, we iterate from 0 to L-1, but if we delete,
    # the next index shifts.
    
    # A more robust approach for local search:
    # While progress is made:
    #   Scan all positions. If deletable, delete.
    #   If no deletions, try swaps?
    
    # Let's try a simple iterative deletion pass first.
    progress = True
    while progress and time.time() < deadline:
        progress = False
        L = len(word)
        # We iterate backwards to handle index shifting naturally?
        # No, if we delete i, indices > i shift down by 1.
        # If we go forward, and delete i, the next element is now at i.
        # So we just don't increment i if we deleted.
        
        i = 0
        while i < len(word):
            if can_delete(i):
                delete(i)
                progress = True
                # Do not increment i, check same index again (which is now the next char)
            else:
                i += 1
                # Small time check
                if time.time() > deadline - 10:
                    break

    # Phase 2: Swap-based local search
    # If greedy deletion stalls, we can try swapping symbols to create more redundancy.
    # A swap is valid if it doesn't break any unique window.
    
    def can_swap(i, j):
        if i == j:
            return False
        if i > j:
            i, j = j, i
        L = len(word)
        if L <= n:
            return False
        
        # Affected windows: those containing i or j
        # k such that k <= i <= k+n-1 OR k <= j <= k+n-1
        # k in [i-n+1, i] U [j-n+1, j]
        
        # Since i and j are far apart (usually), the sets are disjoint.
        # If they are close, we need to be careful.
        
        # Let's just check all windows that would change.
        # A window changes if it contains i or j.
        
        # Optimization: If word[i] == word[j], swap is no-op.
        if word[i] == word[j]:
            return False

        # Check windows containing i
        start_i = max(0, i - n + 1)
        end_i = min(i, L - n)
        for k in range(start_i, end_i + 1):
            if perm_count[tuple(word[k:k+n])] == 1:
                # This window would change (since i is in it and word[i] changes)
                # So it's invalid
                return False
                
        # Check windows containing j
        start_j = max(0, j - n + 1)
        end_j = min(j, L - n)
        for k in range(start_j, end_j + 1):
            if perm_count[tuple(word[k:k+n])] == 1:
                return False
                
        return True

    def do_swap(i, j):
        if i > j:
            i, j = j, i
        L = len(word)
        
        # Store old tuples for affected windows
        # Affected windows are those containing i or j
        # Let's collect them to update counts
        
        # We need to update perm_count.
        # 1. Decrement counts of old windows
        # 2. Swap
        # 3. Increment counts of new windows
        
        # To avoid double counting if windows overlap (which happens if j-i < n),
        # it's safer to:
        # 1. Collect all affected window indices
        # 2. Save their old values
        # 3. Swap
        # 4. Update counts based on new values
        
        affected = set()
        start_i = max(0, i - n + 1)
        end_i = min(i, L - n)
        for k in range(start_i, end_i + 1):
            affected.add(k)
            
        start_j = max(0, j - n + 1)
        end_j = min(j, L - n)
        for k in range(start_j, end_j + 1):
            affected.add(k)
            
        old_vals = {k: tuple(word[k:k+n]) for k in affected}
        
        word[i], word[j] = word[j], word[i]
        
        for k in affected:
            old_p = old_vals[k]
            perm_count[old_p] -= 1
            new_p = tuple(word[k:k+n])
            perm_count[new_p] += 1

    # Local search loop
    # We try to find swaps that allow further deletions.
    # Strategy: Random restarts or random swaps.
    
    # Let's try a few random swaps to break out of local minima.
    # We limit the number of swaps to ensure we finish in time.
    
    # A better strategy: 
    # While time remains:
    #   1. Try greedy deletion.
    #   2. If no progress, pick a random pair (i, j) and try swap.
    #      If swap is valid, do it, then try greedy deletion again.
    #      If swap leads to no improvement (length doesn't decrease after deletion), revert?
    #      Or just keep it if it enables future deletions?
    #      Hill climbing: Accept swap only if it leads to a net decrease in length after subsequent deletions.
    #      This is expensive.
    #      Simpler: Accept valid swaps randomly, then delete. If length doesn't decrease, we might be stuck.
    
    # Let's implement: 
    # 1. Delete all possible.
    # 2. If stuck, try random valid swaps. After a swap, try to delete again.
    #    If length decreases, keep. If not, revert swap and try another.
    
    # To revert swap, we just swap back and update counts.
    
    def undo_swap(i, j):
        # Same as do_swap
        if i > j:
            i, j = j, i
        L = len(word)
        affected = set()
        start_i = max(0, i - n + 1)
        end_i = min(i, L - n)
        for k in range(start_i, end_i + 1):
            affected.add(k)
        start_j = max(0, j - n + 1)
        end_j = min(j, L - n)
        for k in range(start_j, end_j + 1):
            affected.add(k)
        old_vals = {k: tuple(word[k:k+n]) for k in affected}
        word[i], word[j] = word[j], word[i]
        for k in affected:
            old_p = old_vals[k]
            perm_count[old_p] -= 1
            new_p = tuple(word[k:k+n])
            perm_count[new_p] += 1

    def try_improve():
        L_before = len(word)
        # Try a random swap
        i = random.randint(0, len(word) - 1)
        j = random.randint(0, len(word) - 1)
        if can_swap(i, j):
            do_swap(i, j)
            # Try to delete
            progress = True
            while progress:
                progress = False
                i_del = 0
                while i_del < len(word):
                    if can_delete(i_del):
                        delete(i_del)
                        progress = True
                    else:
                        i_del += 1
            if len(word) < L_before:
                return True
            else:
                # No improvement, revert
                undo_swap(i, j)
                return False
        return False

    # Run local search
    max_attempts = 10000
    for _ in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        # First, ensure we've deleted all possible
        progress = True
        while progress:
            progress = False
            i = 0
            while i < len(word):
                if can_delete(i):
                    delete(i)
                    progress = True
                else:
                    i += 1
        
        if try_improve():
            continue
        else:
            # If we fail a few times in a row, break?
            # Or just continue trying random swaps.
            pass

    return "".join(map(str, word))
