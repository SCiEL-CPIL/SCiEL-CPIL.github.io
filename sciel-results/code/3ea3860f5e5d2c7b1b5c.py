import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 115.0

    if n <= 1:
        return "1"

    # 1. Generate Classic Superpermutation
    # Returns list of ints 1..n
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
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n <= 4:
        return "".join(map(str, word))

    # 2. Setup for Deletion
    # Convert to 0-indexed for faster hashing? Or just use tuples of 1..n.
    # Tuples of small ints are fast enough in Python.
    # But let's map to index for O(1) array access.
    
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    N = len(all_perms)
    
    # Current word is list of ints
    curr_word = word[:]
    L = len(curr_word)
    
    # Initialize counts
    counts = [0] * N
    for i in range(L - n + 1):
        w = tuple(curr_word[i:i + n])
        counts[perm_to_idx[w]] += 1
        
    # Check if valid initially (should be)
    # All counts > 0?
    
    # Deletion Loop
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Scan left to right
        for i in range(len(curr_word)):
            if time.time() > deadline - 5:
                break
            
            # Try to delete curr_word[i]
            # Affected windows: [i-n+1, i]
            start_w = max(0, i - n + 1)
            end_w = min(i, len(curr_word) - n)
            
            # Store old windows
            old_windows = []
            for w_idx in range(start_w, end_w + 1):
                w = tuple(curr_word[w_idx:w_idx + n])
                old_windows.append(perm_to_idx[w])
                
            # Decrement
            valid = True
            for idx in old_windows:
                counts[idx] -= 1
                if counts[idx] == 0:
                    valid = False
                    # Revert
                    for rev_idx in old_windows:
                        counts[rev_idx] += 1
                    break
                    
            if not valid:
                continue
                
            # If valid, we need to add new windows formed by the gap
            # The gap is at i.
            # New windows start from i-n to i (relative to new length?)
            # Let's just construct the new word slice and check new windows.
            # New word: curr_word[:i] + curr_word[i+1:]
            # New windows affected are those covering the gap.
            # Gap is between i-1 and i (in new coords, it's between i-1 and i? No).
            # In new word, index i is removed.
            # Windows that include the junction at i (new index i-1 and i)
            # Old indices: i-1, i, i+1...
            # New windows:
            # Start from max(0, i-n) to min(i, L-n-1) ?
            
            # Actually, simpler:
            # Just check if any count is 0 after adding new windows.
            # But we need to add new windows first.
            
            # New windows are formed by `curr_word[k:k+n]` where the range includes the deleted index i.
            # In the new list, the deletion shifts indices.
            # It is safer to:
            # 1. Temporarily remove.
            # 2. Identify new windows.
            # 3. Increment counts.
            # 4. Check if any count < 0? No, we decremented old ones.
            # 5. Check if any count == 0?
            
            # Let's do it properly:
            # 1. Decrement old windows (done above).
            # 2. Identify new windows.
            #    The new word is `curr_word[:i] + curr_word[i+1:]`.
            #    The "seam" is at index i in the new word (which is old i+1).
            #    Windows involving the seam start at max(0, i-n+1) to min(i, new_L-n).
            #    Let's just iterate the potential new window starts.
            
            new_starts = range(max(0, i - n + 1), min(i, len(curr_word) - n))
            # Note: len(curr_word) is old L. New L is L-1.
            # Max start for window of size n in new word is (L-1) - n.
            
            new_windows = []
            for w_idx in new_starts:
                # Construct window from curr_word
                # If w_idx + n <= i, it's from left part
                # If w_idx > i, it's from right part
                # If it crosses i, it's mixed.
                
                # Easier: just slice the conceptual new word.
                # But slicing is O(n).
                # We can construct the tuple manually.
                
                # Window starts at w_idx.
                # It ends at w_idx + n - 1.
                # If w_idx + n - 1 < i: all from left.
                # If w_idx > i: all from right (shifted by -1 in old coords).
                # If w_idx <= i <= w_idx + n - 1: crosses.
                
                # Actually, since we only changed one char, only windows crossing i are NEW.
                # Windows entirely left of i are OLD (and already counted).
                # Windows entirely right of i are OLD (but shifted).
                
                # Wait, the "old windows" we decremented were those *containing* index i in the OLD word.
                # The "new windows" are those *containing* the junction in the NEW word.
                
                # Let's just identify the set of windows that cover the gap.
                # In old word, gap is at i.
                # In new word, gap is between i-1 and i.
                # New windows are those starting at s such that s <= i-1 and s+n-1 >= i.
                # i.e., s in [i-n+1, i-1].
                
                for s in range(max(0, i - n + 1), min(i, len(curr_word) - n)):
                    # Construct tuple for window starting at s in NEW word
                    # New word indices: 0..i-1 are same as old. i..end are old[i+1..]
                    t = []
                    for k in range(s, s + n):
                        if k < i:
                            t.append(curr_word[k])
                        else:
                            t.append(curr_word[k + 1])
                    new_windows.append(perm_to_idx[tuple(t)])
                    
            # Increment new windows
            for idx in new_windows:
                counts[idx] += 1
                
            # Check validity: No count should be 0.
            # We don't need to check all, only the ones we touched?
            # No, if a count was 1, we decremented it to 0, then incremented it back?
            # If a window was in BOTH old and new sets, it was -1 then +1.
            # If a window was ONLY in old, it is -1.
            # If a window was ONLY in new, it is +1.
            
            # So we just need to ensure no count is 0.
            # We can check the touched indices.
            touched = set(old_windows) | set(new_windows)
            is_valid_deletion = True
            for idx in touched:
                if counts[idx] == 0:
                    is_valid_deletion = False
                    break
                    
            if is_valid_deletion:
                # Commit
                del curr_word[i]
                improved = True
            else:
                # Revert
                for idx in old_windows:
                    counts[idx] += 1
                for idx in new_windows:
                    counts[idx] -= 1

    return "".join(map(str, curr_word))
