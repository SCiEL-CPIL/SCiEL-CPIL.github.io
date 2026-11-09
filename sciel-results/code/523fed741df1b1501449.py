"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import permutations
import sys

def construct(n):
    # Standard recursive construction
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
    
    if n <= 7:
        # Fast local optimization using counts
        all_perms = list(permutations(range(1, n + 1)))
        perm_to_idx = {p: i for i, p in enumerate(all_perms)}
        num_perms = len(all_perms)
        
        # Convert word to list of ints for faster processing
        w = word[:]
        
        def update_counts(w, counts):
            counts = [0] * num_perms
            for i in range(len(w) - n + 1):
                # Check if window is a permutation
                window = w[i:i+n]
                if len(set(window)) == n:
                    idx = perm_to_idx.get(tuple(window), -1)
                    if idx != -1:
                        counts[idx] += 1
            return counts

        counts = update_counts(w, None)
        
        # Check if initial word is valid (should be)
        # if any(count == 0 for count in counts):
        #     pass # Fallback?
        
        improved = True
        # Limit iterations to avoid timeout
        max_iterations = 200 
        iter_count = 0
        
        while improved and iter_count < max_iterations:
            improved = False
            iter_count += 1
            # Try removing from right to left to avoid index shifting issues in a single pass
            # Or just scan left to right and restart on success
            for pos in range(len(w) - 1, -1, -1):
                # Check if removal is safe
                # Windows affected: those starting from pos-n+1 to pos
                # We need to ensure that after removal, no count drops to 0
                
                # Identify windows to decrement
                to_decrement = []
                for start in range(max(0, pos - n + 1), min(pos + 1, len(w) - n + 1)):
                    window = w[start:start+n]
                    if len(set(window)) == n:
                        idx = perm_to_idx.get(tuple(window), -1)
                        if idx != -1:
                            to_decrement.append(idx)
                
                # Check if any would become 0
                valid_removal = True
                for idx in to_decrement:
                    if counts[idx] == 1:
                        valid_removal = False
                        break
                
                if valid_removal:
                    # Perform removal
                    w.pop(pos)
                    # Update counts
                    for idx in to_decrement:
                        counts[idx] -= 1
                    
                    # Update counts for new windows formed by the shift
                    # Windows starting from pos-n+1 to pos-1 (shifted left)
                    # Actually, simpler to just recompute counts for the affected range?
                    # Or update incrementally.
                    # The shift affects windows starting at pos-n+1 to pos-1 (new ones)
                    # And the windows we just decremented are gone/changed.
                    
                    # Let's just recompute counts for the whole word if it's small enough?
                    # For n=7, len(w) ~ 6000. Recomputing is 6000 * 7 ops = 42000.
                    # This is fast enough in Python if done ~1000 times? 42M ops. Might be tight.
                    # Let's try incremental update.
                    
                    # The windows that *change* are those that cross the removal point.
                    # Old windows: start in [pos-n+1, pos]
                    # New windows: start in [pos-n+1, pos-1] (shifted left)
                    
                    # We already decremented the old ones.
                    # Now we need to increment the new ones.
                    # Note: The window at start=pos in the new word corresponds to old start=pos+1? No.
                    # w is now shorter.
                    # New window at index i (in new w) corresponds to old w[i] if i < pos, old w[i+1] if i >= pos.
                    
                    # Let's just recompute counts to be safe and simple, as incremental is bug-prone.
                    # Optimization: Only recompute if len(w) is small? No, always.
                    counts = update_counts(w, counts)
                    
                    improved = True
                    break
        
    s = "".join(map(str, w))
    return s
# EVOLVE-BLOCK-END
