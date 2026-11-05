import itertools
import time
import heapq
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    if n == 1:
        return "1"
    if n == 2:
        return "121"

    # Step 1: Build standard superpermutation (length = sum(i! for i=1..n))
    # For n=7: 1+2+6+24+120+720+5040 = 5913
    def build_standard(n):
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
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    word = build_standard(n)
    L = len(word)

    # Build frequency map of all n-windows
    window_freq = defaultdict(int)
    for i in range(L - n + 1):
        p = tuple(word[i:i + n])
        window_freq[p] += 1

    # Verify all permutations present
    required = set(itertools.permutations(range(1, n + 1)))
    present = set(window_freq.keys())
    if present != required:
        return "".join(map(str, word))

    # Helper to delete a position and update window_freq
    def do_delete(pos):
        """Delete position pos. Returns True if successful."""
        nonlocal word, window_freq, L
        if pos <= 0 or pos >= L - 1:
            return False

        lo = max(0, pos - n + 1)
        hi = min(pos, L - n)

        if lo > hi:
            return False

        # Check: all affected windows must have freq > 1
        for w in range(lo, hi + 1):
            p = tuple(word[w:w + n])
            if window_freq[p] <= 1:
                return False

        # Save old affected windows
        old_perms = [tuple(word[w:w + n]) for w in range(lo, hi + 1)]

        # Perform deletion
        word = word[:pos] + word[pos + 1:]
        L = len(word)

        # New affected region
        new_lo = max(0, pos - n)
        new_hi = min(pos, L - n)

        # Decrement old
        for p in old_perms:
            window_freq[p] -= 1
            if window_freq[p] == 0:
                del window_freq[p]

        # Increment new
        for w in range(new_lo, new_hi + 1):
            p = tuple(word[w:w + n])
            window_freq[p] += 1

        return True

    # Randomized Deletion Search
    # This approach tries to escape local minima by randomly selecting positions to delete
    # rather than strictly following a greedy priority.
    
    import random
    random.seed(42)
    
    max_failures = 500
    total_deletions = 0
    
    # Phase 1: Randomized Deletion
    while time.time() < deadline - 20:
        L = len(word)
        if L <= n:
            break
        
        failures = 0
        # Try to delete random positions
        while failures < max_failures:
            if time.time() > deadline - 20:
                break
            
            # Pick a random position between 1 and L-2
            pos = random.randint(1, L - 2)
            
            if do_delete(pos):
                total_deletions += 1
                failures = 0 # Reset failure count on success
            else:
                failures += 1
        
        # If we had a long streak of failures, try end trimming
        if failures >= max_failures:
             # End trimming
             changed = True
             while changed:
                 changed = False
                 L = len(word)
                 if L > n:
                     first_perm = tuple(word[:n])
                     if window_freq[first_perm] > 1:
                         window_freq[first_perm] -= 1
                         if window_freq[first_perm] == 0:
                             del window_freq[first_perm]
                         word = word[1:]
                         L = len(word)
                         new_perm = tuple(word[:n])
                         window_freq[new_perm] += 1
                         changed = True

                 if L > n:
                     last_perm = tuple(word[-n:])
                     if window_freq[last_perm] > 1:
                         window_freq[last_perm] -= 1
                         if window_freq[last_perm] == 0:
                             del window_freq[last_perm]
                         word = word[:-1]
                         L = len(word)
                         new_perm = tuple(word[-n:])
                         window_freq[new_perm] += 1
                         changed = True
             
             # If no changes made by trimming, break the main loop
             # We can't easily know if trimming made changes without a flag, 
             # so we just check if L changed or just break if no deletions in this round.
             # Actually, if do_delete failed max_failures times, we are likely stuck.
             # Let's break if no deletions were made in the last "batch"
             pass

    # Final verification
    final_perms = set()
    for i in range(len(word) - n + 1):
        final_perms.add(tuple(word[i:i + n]))

    if len(final_perms) != len(required):
        # Fallback: return the standard construction
        word = build_standard(n)

    return "".join(map(str, word))
