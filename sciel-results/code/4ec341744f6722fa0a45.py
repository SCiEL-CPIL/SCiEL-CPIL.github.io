"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import Counter

# EVOLVE-BLOCK-START
def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
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
    
    # Precompute all permutations of 1..n
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Build initial permutation count dictionary
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if window in all_perms:
            perm_counts[window] += 1
    
    # Count how many permutations are currently covered
    covered = sum(1 for p in all_perms if perm_counts.get(p, 0) > 0)
    
    # Greedy deletion with efficient incremental updates
    # A position i is safe to delete if no permutation that appears ONLY at windows
    # containing position i would become uncovered
    
    changed = True
    i = 0
    while changed and (time.time() - start_time) < deadline - 3:
        changed = False
        while i < len(word) and (time.time() - start_time) < deadline - 3:
            # Determine which windows are affected by removing position i
            # Window starting at s includes position i iff s <= i <= s+n-1
            # So s in [max(0, i-n+1), min(i, len(word)-n)]
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            
            # Check if any permutation would become uncovered
            safe = True
            for s in range(s_min, s_max + 1):
                window = tuple(word[s:s + n])
                if window in all_perms and perm_counts[window] == 1:
                    safe = False
                    break
            
            if safe:
                # Remove position i
                # Update perm_counts for affected windows
                # Windows that start in [s_min, s_max] are removed (they contained position i)
                # After removal, new windows form: those starting in [s_min, s_max] 
                # but shifted, plus potentially new windows from the join point
                
                # Remove old windows that contained position i
                for s in range(s_min, s_max + 1):
                    window = tuple(word[s:s + n])
                    if window in all_perms:
                        perm_counts[window] -= 1
                        if perm_counts[window] == 0:
                            covered -= 1
                
                # Remove the character
                word.pop(i)
                
                # After removal, new windows form where the gap is closed
                # New windows start at positions s where the window crosses the join point
                # The join point is at position i (where we removed)
                # New windows: those starting at s in [max(0, i-n+1), min(i, len(word)-n)]
                # but we need to check which are new
                # Actually, after removing word[i], the sequence is word[0..i-1] + word[i+1..]
                # New windows that cross position i-1/i boundary:
                # s from max(0, i-n+1) to min(i-1, len(word)-n)
                # Wait, let me think more carefully.
                
                # Before removal: word = [..., word[i-1], word[i], word[i+1], ...]
                # After removal: word = [..., word[i-1], word[i+1], ...]
                # Windows that are NEW (didn't exist before) must cross the join point
                # A window starting at s includes positions s, s+1, ..., s+n-1
                # In the new word, position i-1 is followed by old position i+1
                # So new windows are those where s <= i-1 < s+n, i.e., s in [i-n+1, i-1]
                # But also s >= 0 and s <= len(word)-n (new length)
                
                new_s_min = max(0, i - n + 1)
                new_s_max = min(i - 1, len(word) - n)
                
                for s in range(new_s_min, new_s_max + 1):
                    window = tuple(word[s:s + n])
                    if window in all_perms:
                        old_count = perm_counts.get(window, 0)
                        perm_counts[window] = old_count + 1
                        if old_count == 0:
                            covered += 1
                
                changed = True
                # Don't increment i; re-examine the same position
                # (the character that was at i+1 is now at i)
            else:
                i += 1
        
        # If we went through the whole word without any change, we're done
        if not changed:
            break
    
    # Final verification
    final_perms = set()
    for i in range(len(word) - n + 1):
        final_perms.add(tuple(word[i:i + n]))
    
    if not all_perms.issubset(final_perms):
        # Shouldn't happen, but return original if something went wrong
        # Rebuild from scratch
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
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
