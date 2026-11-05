import itertools
import time
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Step 1: Build standard superpermutation (length = sum(i! for i=1..n))
    # For n=7: 1+2+6+24+120+720+5040 = 5913
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
    
    if time.time() > deadline - 5:
        return "".join(map(str, word))
    
    # Step 2: Build frequency map of all n-windows
    L = len(word)
    window_freq = defaultdict(int)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        window_freq[p] += 1
    
    # Verify all 5040 permutations present
    required = set(itertools.permutations(range(1, n+1)))
    present = set(window_freq.keys())
    if present != required:
        return "".join(map(str, word))
    
    # Step 3: Aggressive deletion using frequency-count tracking
    # Key insight: deleting position pos affects windows [pos-n+1, pos]
    # A deletion is safe iff for every affected window, its freq > 1 (so another copy remains)
    # We maintain window_freq incrementally to avoid O(L) full scans per deletion
    
    def try_delete(pos):
        """Try to delete position pos. Returns True if successful."""
        nonlocal word, window_freq, L
        if pos <= 0 or pos >= L - 1:
            return False
        
        # Windows affected: starting at max(0, pos-n+1) to min(pos, L-n)
        lo = max(0, pos - n + 1)
        hi = min(pos, L - n)
        
        if lo > hi:
            return False
        
        # Check: all affected windows must have freq > 1
        for w in range(lo, hi + 1):
            p = tuple(word[w:w+n])
            if window_freq[p] <= 1:
                return False
        
        # Perform deletion
        word = word[:pos] + word[pos+1:]
        L = len(word)
        
        # Update window_freq: remove old affected windows, add new ones
        # Old windows that are gone: those starting at lo..hi in old word
        # But some may still exist shifted. Let's just recompute the affected region.
        # Actually, deleting pos removes windows starting at lo..hi (old indices).
        # New windows appear starting at lo..hi-1 (if they changed) and possibly hi (shifted).
        # Simpler: recompute freq for windows in [lo-1, hi] in new word, adjust counts.
        
        # The windows that definitely change are those overlapping position pos.
        # In the new word, windows starting at max(0, pos-n) to min(pos, L-n) may differ.
        # Let's just decrement for old windows and increment for new ones in the affected zone.
        
        # Old affected windows (in old word): start at lo..hi
        # These are removed. But some windows in the new word at those positions may be the same.
        # Safest: decrement all old affected, increment all new affected, net effect is correct.
        
        # Old affected windows
        for w in range(lo, hi + 1):
            p = tuple(word[w+n-1-pos:w+n-1-pos+n])  # This is wrong after deletion
            # Actually we already deleted, so let me recompute properly
            pass
        
        # Let me redo this more carefully.
        # Before deletion, windows at old positions lo..hi were affected.
        # After deletion, the word is shorter by 1.
        # Windows at new positions lo..hi may be different.
        # The correct approach: 
        #   - For each old window position w in [lo, hi]: decrement freq of old_word[w:w+n]
        #   - For each new window position w in [lo, hi-1] (and possibly hi): increment freq of new_word[w:w+n]
        # But we've already modified word. Let's just recompute the small affected region.
        
        # Actually, the cleanest approach for small n: just recompute freq for windows
        # in the range [lo-1, hi] in the new word, and adjust.
        # But we need to know what was there before. Let's use a different strategy:
        # Keep a set of "dirty" positions and periodically rebuild.
        
        return True
    
    # The above try_delete is complex. Let's use a simpler but effective approach:
    # Maintain window_freq. For each deletion attempt, check the local condition,
    # then update window_freq by recomputing only the affected windows.
    
    # Reset: rebuild window_freq from current word
    L = len(word)
    window_freq = defaultdict(int)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        window_freq[p] += 1
    
    # Iterative deletion: scan left to right, try to delete each position
    # Use a flag to track if we need to rescan
    deleted_total = 0
    max_rounds = 50  # limit rounds to avoid infinite loop
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 10:
            break
        
        L = len(word)
        if L <= n:
            break
        
        # Rebuild critical info
        # A position pos is deletable if all windows [pos-n+1..pos] have freq > 1
        # Scan left to right
        pos = 1
        made_change = False
        while pos < L - 1:
            if time.time() > deadline - 10:
                break
            
            lo = max(0, pos - n + 1)
            hi = min(pos, L - n)
            
            if lo > hi:
                pos += 1
                continue
            
            # Check if all affected windows have freq > 1
            safe = True
            for w in range(lo, hi + 1):
                p = tuple(word[w:w+n])
                if window_freq[p] <= 1:
                    safe = False
                    break
            
            if safe:
                # Delete position pos
                # Update window_freq: 
                # Old windows at lo..hi are removed
                # New windows at lo..hi-1 (shifted) and possibly hi are added
                
                # Save old affected window permutations
                old_perms = []
                for w in range(lo, hi + 1):
                    old_perms.append(tuple(word[w:w+n]))
                
                # Perform deletion
                word = word[:pos] + word[pos+1:]
                L = len(word)
                
                # New affected region: windows that overlap the deletion point
                # In new word, windows starting at max(0, pos-n) to min(pos, L-n)
                new_lo = max(0, pos - n)
                new_hi = min(pos, L - n)
                
                # Decrement old perms
                for p in old_perms:
                    window_freq[p] -= 1
                    if window_freq[p] == 0:
                        del window_freq[p]
                
                # Increment new perms
                for w in range(new_lo, new_hi + 1):
                    p = tuple(word[w:w+n])
                    window_freq[p] += 1
                
                deleted_total += 1
                made_change = True
                # Don't increment pos; check same position again (word shifted)
            else:
                pos += 1
        
        if not made_change:
            break
    
    # Step 4: Try end trimming
    if time.time() < deadline - 5:
        changed = True
        while changed:
            changed = False
            L = len(word)
            if L > n:
                first_perm = tuple(word[:n])
                if window_freq[first_perm] > 1:
                    # Remove first char
                    # Update freq
                    # Old window at 0 is removed
                    window_freq[first_perm] -= 1
                    if window_freq[first_perm] == 0:
                        del window_freq[first_perm]
                    # New window at 0 (shifted)
                    word = word[1:]
                    L = len(word)
                    new_perm = tuple(word[:n])
                    window_freq[new_perm] += 1
                    changed = True
            
            if L > n:
                last_perm = tuple(word[-n:])
                if window_freq[last_perm] > 1:
                    # Remove last char
                    window_freq[last_perm] -= 1
                    if window_freq[last_perm] == 0:
                        del window_freq[last_perm]
                    word = word[:-1]
                    L = len(word)
                    new_perm = tuple(word[-n:])
                    window_freq[new_perm] += 1
                    changed = True
    
    # Step 5: Second round of internal deletions (after trimming)
    if time.time() < deadline - 5:
        L = len(word)
        # Rebuild window_freq to be safe
        window_freq = defaultdict(int)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            window_freq[p] += 1
        
        for rnd in range(20):
            if time.time() > deadline - 5:
                break
            
            L = len(word)
            if L <= n:
                break
            
            pos = 1
            made_change = False
            while pos < L - 1:
                if time.time() > deadline - 5:
                    break
                
                lo = max(0, pos - n + 1)
                hi = min(pos, L - n)
                
                if lo > hi:
                    pos += 1
                    continue
                
                safe = True
                for w in range(lo, hi + 1):
                    p = tuple(word[w:w+n])
                    if window_freq[p] <= 1:
                        safe = False
                        break
                
                if safe:
                    old_perms = []
                    for w in range(lo, hi + 1):
                        old_perms.append(tuple(word[w:w+n]))
                    
                    word = word[:pos] + word[pos+1:]
                    L = len(word)
                    
                    new_lo = max(0, pos - n)
                    new_hi = min(pos, L - n)
                    
                    for p in old_perms:
                        window_freq[p] -= 1
                        if window_freq[p] == 0:
                            del window_freq[p]
                    
                    for w in range(new_lo, new_hi + 1):
                        p = tuple(word[w:w+n])
                        window_freq[p] += 1
                    
                    made_change = True
                else:
                    pos += 1
            
            if not made_change:
                break
    
    # Final verification
    final_perms = set()
    for i in range(len(word) - n + 1):
        final_perms.add(tuple(word[i:i+n]))
    
    if len(final_perms) != len(required):
        # Fallback: return the standard construction
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
    
    return "".join(map(str, word))
