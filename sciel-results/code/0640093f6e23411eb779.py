"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def construct(n):
    deadline = time.time() + 140  # 10s safety margin

    # Classic recursive construction of length 1! + 2! + ... + n!
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

    # For n >= 7, apply local search to shorten the word
    all_perms = set(permutations(range(1, n + 1)))

    # Precompute a mapping from permutation tuple to ID for faster counting
    perm_list = list(all_perms)
    perm_to_id = {p: i for i, p in enumerate(perm_list)}
    num_perms = len(perm_list)

    # Initialize counts of how many times each permutation appears
    counts = [0] * num_perms
    L = len(word)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        if p in perm_to_id:
            counts[perm_to_id[p]] += 1

    # Check if valid initially
    if any(c == 0 for c in counts):
        # Should not happen for standard construction, but return if it does
        return "".join(map(str, word))

    # Local Search: Deletion
    # We will repeatedly try to delete characters.
    # To be efficient, we maintain counts and check if any count drops to 0.
    
    def try_delete(i):
        """
        Attempts to delete word[i]. Returns True if successful.
        Modifies word and counts if successful.
        """
        L = len(word)
        if L == n: # Cannot delete if only one window
            return False
        
        # Save old counts to revert if necessary
        # Copying 5040 ints is fast enough in Python? 5040 * 6000 ops ~ 30M ops.
        # Let's try to avoid full copy if possible, or just accept the cost.
        # A better way: only track the affected permutations.
        
        # Affected windows in old word: start k in [i-n+1, i]
        # We need to decrement counts for these.
        old_affected_pids = []
        for k in range(max(0, i - n + 1), min(i + 1, L - n + 1)):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] -= 1
                old_affected_pids.append(pid)
        
        # Simulate deletion to find new windows
        # New word is word[:i] + word[i+1:]
        # New windows affected are those starting at k in [i-n+1, i] in the NEW word.
        # In the new word, index i is removed.
        # Windows starting at k < i:
        #   If k+n <= i: window is same as old word[k:k+n]. No change.
        #   If k+n > i: window is word[k:i] + word[i+1:k+n+1].
        # Windows starting at k >= i:
        #   In new word, these correspond to old word[k+1:k+1+n].
        
        # Actually, it's easier to just compute the new windows in the affected range.
        # Affected range in new word: start k in [max(0, i-n+1), min(i, L-n)]
        
        new_affected_pids = []
        # Range of start indices in the NEW word that overlap with the deletion point i
        # The deletion point i in old word becomes a gap.
        # Windows in new word that cross the gap:
        # Start index k in new word.
        # If k < i: the window is new_word[k:k+n].
        #   new_word[k:i] = word[k:i]
        #   new_word[i:k+n] = word[i+1:k+n+1]
        #   So the window is word[k:i] + word[i+1:k+n+1]
        #   This is valid if k < i and k+n > i (in old word terms, the window covers i)
        #   Wait, k is index in NEW word.
        #   Let's map carefully.
        #   Old word: ... A B C D E ... where i points to B (0-indexed)
        #   New word: ... A C D E ...
        #   Window in new word starting at k:
        #   If k < i:
        #     new_word[k] = word[k]
        #     ...
        #     new_word[i-1] = word[i-1]
        #     new_word[i] = word[i+1]
        #     ...
        #   So window is word[k:i] + word[i+1:k+n+1]
        #   This window is "new" if it wasn't present in old word at that position.
        #   In old word, the window at k was word[k:k+n].
        #   The window at k in new word is different if it includes the gap.
        #   It includes the gap if k < i and k+n > i (in new word indices, the gap is at i).
        #   Actually, simpler: just compute the tuple and check if it's in perm_to_id.
        
        # Let's iterate k from max(0, i-n+1) to min(i, L-n) in NEW word indices?
        # No, let's just check all windows in the new word that are different from old.
        # The only windows that change are those that span the deletion point.
        # In the new word, the deletion point is between i-1 and i (if i>0).
        # Actually, the indices shift.
        
        # Let's just compute the new windows that involve the region around i.
        # In the new word, the indices are:
        # 0 ... i-1, i+1, i+2, ...
        # The windows that change are those starting at k such that the window covers the "gap".
        # The gap is at position i in the old word.
        # In the new word, the elements at i-1 and i+1 are adjacent.
        # So windows starting at k in [i-n+1, i-1] in the NEW word might be affected?
        # Let's just check k in range(max(0, i-n+1), min(i, L-n)) for NEW word starts?
        # No, let's just check k in range(max(0, i-n+1), min(i, L-n)) is confusing.
        
        # Simpler approach:
        # Identify the set of start indices in the OLD word that are affected: [i-n+1, i]
        # Identify the set of start indices in the NEW word that are affected: [i-n+1, i-1] (shifted)
        # Actually, let's just compute the new windows for k in range(max(0, i-n+1), min(i, L-n))
        # Wait, if i=0, new word starts at 1.
        
        # Let's just compute the new windows for all k in the range where the window content might change.
        # The content changes for windows that include the position i in the old word.
        # In the new word, these windows are formed by concatenating parts from before and after i.
        
        # Let's iterate k from max(0, i-n+1) to min(i, L-n) in the NEW word?
        # No, let's just check the windows that start at k in the NEW word where k is in [max(0, i-n+1), min(i-1, L-n)]
        # And also the window that starts at i-1?
        
        # Let's just be safe and check k in range(max(0, i-n+1), min(i, L-n))
        # And also k = i-1?
        
        # Actually, the simplest correct way:
        # The new word is word[:i] + word[i+1:]
        # The windows that are different from the old word are those that span the boundary between word[i-1] and word[i+1].
        # In the new word, this boundary is at index i-1 (if i>0).
        # So windows starting at k in [i-n, i-1] in the NEW word might be new.
        # Let's just check k in range(max(0, i-n), min(i, L-n))
        
        for k in range(max(0, i-n), min(i, L-n)):
            # Construct the window in the new word
            # If k < i:
            #   window = word[k:i] + word[i+1:k+n+1]
            # If k >= i:
            #   window = word[k+1:k+1+n]
            # But we only care about windows that are "new" or "changed".
            # The windows with k >= i are just the old windows shifted. They were already counted.
            # The windows with k < i that do NOT span the gap are also old windows.
            # The only windows that are NEW are those that span the gap.
            # A window in the new word spans the gap if it includes both word[i-1] and word[i+1].
            # This happens if k < i and k+n > i (in old word terms, the window covers i).
            # In new word terms, the gap is at i-1.
            # So k < i-1 and k+n > i-1?
            
            # Let's just compute the tuple and check if it's in perm_to_id.
            # If it is, increment its count.
            # We need to be careful not to double count.
            
            if k < i:
                w1 = word[k:i]
                w2 = word[i+1:k+n+1]
                p = tuple(w1 + w2)
            else:
                p = tuple(word[k+1:k+1+n])
            
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] += 1
                new_affected_pids.append(pid)
        
        # Check if all counts are >= 1
        valid = True
        # We only need to check the affected pids?
        # No, because we decremented some and incremented some.
        # If any count became 0, it must be one of the decremented ones that wasn't incremented back.
        # So we only need to check the old_affected_pids?
        # Yes, because other counts are unchanged or incremented.
        for pid in old_affected_pids:
            if counts[pid] == 0:
                valid = False
                break
        
        if valid:
            # Perform deletion
            word.pop(i)
            return True
        else:
            # Revert counts
            for pid in old_affected_pids:
                counts[pid] += 1
            for pid in new_affected_pids:
                counts[pid] -= 1
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
                # Do not increment i, check same position again (new character)
            else:
                i += 1
        # Right to left
        i = len(word) - 1
        while i >= 0:
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_delete(i):
                improved = True
                # Do not decrement i, check same position again
            else:
                i -= 1

    return "".join(map(str, word))
# EVOLVE-BLOCK-END
