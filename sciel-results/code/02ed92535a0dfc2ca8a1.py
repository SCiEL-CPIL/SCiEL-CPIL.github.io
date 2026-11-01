"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import Counter

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Classic recursive construction of superpermutation
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
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For n <= 4, the classic construction is optimal, return immediately
    if n <= 4:
        return "".join(map(str, word))
    
    # For n >= 5, attempt greedy deletion to shorten the word
    # Precompute all permutations of 1..n
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Count how many times each permutation appears as a window
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if window in all_perms:
            perm_counts[window] += 1
    
    # Build a map: for each position, which permutations are "solely" supported by it
    # A position i is in windows starting from max(0, i-n+1) to min(i, len(word)-n)
    
    def try_delete(word, perm_counts, all_perms, n, i):
        """Check if position i can be deleted. Returns (can_delete, new_perm_counts_if_changed)"""
        s_min = max(0, i - n + 1)
        s_max = min(i, len(word) - n)
        
        # Check if any window that includes position i has count == 1
        for s in range(s_min, s_max + 1):
            window = tuple(word[s:s + n])
            if window in all_perms and perm_counts[window] == 1:
                return False, None
        return True, None
    
    # More efficient: batch deletion using a worklist approach
    # Track for each permutation, which positions support it
    # But that's expensive. Instead, use the count-based check but with incremental updates.
    
    # Strategy: repeatedly scan for deletable positions, delete them, update counts incrementally
    # After deleting position i, windows that shifted need count updates.
    
    changed = True
    iteration = 0
    while changed and (time.time() - start_time) < deadline - 5:
        iteration += 1
        changed = False
        i = 0
        while i < len(word):
            if time.time() - start_time > deadline - 5:
                break
            
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            
            # Check if removing position i would uncover any permutation
            safe = True
            for s in range(s_min, s_max + 1):
                window = tuple(word[s:s + n])
                if window in all_perms and perm_counts[window] == 1:
                    safe = False
                    break
            
            if safe:
                # Remove position i
                word.pop(i)
                changed = True
                # Update perm_counts: recompute counts for windows near position i
                # Windows affected: those that start in [max(0, i-n+1), min(i, len(word)-n)]
                # After deletion, windows starting >= i shift left by 1
                # Recompute counts for a local region
                recompute_start = max(0, i - n)
                recompute_end = min(len(word) - n + 1, i + n + 1)
                
                # Decrement counts for old windows in the affected region
                # This is complex; for efficiency with n=7, recompute a wider region
                # Actually, let's just recompute all counts if the word is small enough
                # For n=7, len(word) starts at ~13699, so full recompute is 13699 * 7 = ~96k ops
                # That's fast enough. Let's do full recompute.
                perm_counts = Counter()
                for s in range(len(word) - n + 1):
                    window = tuple(word[s:s + n])
                    if window in all_perms:
                        perm_counts[window] += 1
                
                # Don't increment i since we deleted; the next position is now at index i
                continue
            else:
                i += 1
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
