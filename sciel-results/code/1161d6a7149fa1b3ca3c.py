import itertools
import time
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

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
            # Find max overlap with end of new
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

    # Step 3: Efficient deletion using local verification
    # Key insight: deleting position pos only affects windows [pos-n+1, pos]
    # A deletion is safe iff for every affected window with freq==1, 
    # the permutation still appears at a position not affected by the deletion.
    # More precisely: a window at position w (in original) is "lost" if:
    #   - it contains position pos (i.e., w <= pos <= w+n-1)
    #   - its permutation has freq == 1 in the original word
    # But we also need to check that no NEW window created by the deletion
    # has a permutation that only appeared in deleted windows.
    # Actually, the simplest correct check:
    #   After deletion, for each window position w in the new word,
    #   the permutation at w must appear at least once in the new word.
    #   The only windows that could be problematic are those affected by the deletion.
    #   A window in the new word at position w corresponds to:
    #     - original window at w if w < pos
    #     - a "merged" window (original words w and w+1 minus pos) if w >= pos
    #   For w < pos: unchanged, so safe if original freq >= 1 (always true)
    #   For w >= pos: the new window is word[w:pos] + word[pos+1:w+n+1] (in original indices)
    #   We need to check if this new permutation appears elsewhere in the new word.
    
    # Better approach: track which permutations are "unique" and where they appear.
    # A deletion at pos is safe iff:
    #   For every unique permutation (freq==1) whose window contains pos,
    #   that permutation still appears in the new word (i.e., at some position 
    #   that doesn't involve the deleted element in a way that changes it).
    #   AND for the new windows created at positions >= pos, their permutations
    #   are not new (they appeared before) or appear multiple times.
    
    # Simplest efficient approach:
    # 1. Find all positions where the permutation has freq==1 (unique windows)
    # 2. For a deletion at pos, check if any unique window contains pos
    #    If so, we need to verify that permutation still exists after deletion
    # 3. Also check the new windows formed at positions >= pos
    
    # Let me use a different strategy: maintain a set of "essential" positions
    # where the window is unique. A position pos can be deleted if no essential
    # window contains it. But this is too conservative.
    
    # Actually, the most efficient correct approach:
    # For each position pos, the deletion is safe iff:
    #   (a) No window with freq==1 that contains pos is the ONLY occurrence of its perm
    #       (this is automatic since freq==1 means it's the only one)
    #   Wait, if a window has freq==1 and contains pos, then after deletion that
    #   perm is gone UNLESS the same perm appears in a new window formed by the merge.
    #   
    # Let me think differently. The new word after deleting pos has windows:
    #   - Positions 0..pos-n: same as original (if pos-n >= 0)
    #   - Position pos-n+1: word[pos-n+1:pos] + word[pos+1:pos+2] (n chars, skipping pos)
    #   - ...
    #   - Position new_len-n: last window
    #   
    # The key observation: a permutation p is lost iff p appears in the original word
    # only in windows that all contain pos, AND p does not appear in any of the
    # new "merged" windows.
    #
    # For efficiency, I'll use this approach:
    # 1. Precompute for each position pos, which unique windows contain it
    # 2. For a candidate deletion, check if any unique window containing pos has its
    #    perm appearing in a merged window or in an unaffected window
    
    # Even simpler: just do the local check efficiently.
    # For deletion at pos, the affected windows in the NEW word are positions:
    #   lo_new = max(0, pos - n + 1) to hi_new = min(pos, new_len - n)
    # For each such window w in the new word:
    #   If w < pos: it's the same as original window w, which had freq >= 1, so safe
    #   If w >= pos: it's a new window. Check if its perm appears elsewhere.
    #   But "elsewhere" means at some position in the new word.
    #   We can check: does this perm appear in the original word at a position 
    #   that is not in [w, w+n-1] shifted appropriately?
    #   
    # This is getting complicated. Let me use a pragmatic approach:
    # For each candidate pos, do a quick check:
    #   - Count how many unique windows contain pos
    #   - If 0, the deletion is very likely safe (but still verify)
    #   - If > 0, do a full local verification
    
    # Build list of unique window positions
    unique_windows = []  # list of (position, permutation)
    for i in range(L - n + 1):
        if window_freq[tuple(word[i:i+n])] == 1:
            unique_windows.append(i)
    
    # For each position, count how many unique windows contain it
    unique_count_at = [0] * L
    for uw in unique_windows:
        for j in range(uw, uw + n):
            unique_count_at[j] += 1
    
    # Candidate positions: those with fewest unique windows
    candidates = []
    for pos in range(1, L - 1):
        if unique_count_at[pos] == 0:
            candidates.append((0, pos))
        elif unique_count_at[pos] <= 2:
            candidates.append((unique_count_at[pos], pos))
    
    candidates.sort()
    
    # For verification, use a helper that checks if deletion at pos is safe
    def is_deletion_safe(word, pos, n, window_freq):
        """Check if deleting position pos from word is safe (all perms still present)."""
        L_orig = len(word)
        new_len = L_orig - 1
        if new_len < n:
            return False
        
        # Windows in the new word that are "new" (affected by deletion):
        # They start at positions max(0, pos-n+1) to min(pos, new_len-n)
        lo = max(0, pos - n + 1)
        hi = min(pos, new_len - n)
        
        if lo > hi:
            return True
        
        # Build the new word
        new_word = word[:pos] + word[pos+1:]
        
        # For each new window position w in [lo, hi]:
        # Get the permutation and check if it appears elsewhere in new_word
        # To make this fast, we check against the original window_freq
        # adjusted for the deletion
        
        for w in range(lo, hi + 1):
            p = tuple(new_word[w:w+n])
            # Check if p appears in new_word at some position other than w
            # Quick check: if p's original freq was > 1, and the other occurrence
            # is not in the affected range, it's safe
            orig_freq = window_freq.get(p, 0)
            
            if orig_freq == 0:
                # New permutation not in original - must appear at least twice in new_word
                # This is rare, check directly
                count = 0
                for cp in range(new_len - n + 1):
                    if cp == w:
                        continue
                    if tuple(new_word[cp:cp+n]) == p:
                        count += 1
                        if count >= 1:
                            break
                if count < 1:
                    return False
            elif orig_freq == 1:
                # Only one occurrence in original. Check if it's at position w in new_word
                # or if the original occurrence is affected
                # Find where p was in original
                # The original window containing p: since freq was 1, there's exactly one
                # position where p appeared. If that position is not w in the new word
                # and is not destroyed by the deletion, we're fine.
                # Actually, if p appears in new_word at position w, and orig_freq was 1,
                # then p might have been at a position that got merged into w, or at w
                # if w < pos.
                # Simplest: check if p appears in new_word at any position != w
                found = False
                for cp in range(new_len - n + 1):
                    if cp == w:
                        continue
                    if tuple(new_word[cp:cp+n]) == p:
                        found = True
                        break
                if not found:
                    return False
            else:
                # orig_freq >= 2: check if at least one occurrence survives
                # The occurrences in original are at various positions.
                # After deletion, positions >= pos shift left by 1.
                # An original occurrence at position q survives if:
                #   - q < pos: it's at position q in new_word
                #   - q > pos: it's at position q-1 in new_word  
                #   - q == pos: destroyed (but q is a window start, and we deleted element at pos)
                #   - q <= pos <= q+n-1: the window is affected
                # This is complex. Just check directly but limit the search.
                # Since orig_freq >= 2, likely safe. Do a bounded check.
                found = False
                for cp in range(new_len - n + 1):
                    if cp == w:
                        continue
                    if tuple(new_word[cp:cp+n]) == p:
                        found = True
                        break
                if not found:
                    return False
        
        return True
    
    deleted_count = 0
    max_deletions = 10000
    
    # First pass: delete positions with 0 unique windows
    for _, pos in candidates:
        if time.time() > deadline - 15:
            break
        if deleted_count >= max_deletions:
            break
        
        if pos >= len(word):
            continue
        
        # Quick check: if unique_count_at[pos] == 0, likely safe
        # But still verify
        if is_deletion_safe(word, pos, n, window_freq):
            word = word[:pos] + word[pos+1:]
            deleted_count += 1
            
            # Rebuild data structures periodically
            if deleted_count % 100 == 0:
                L = len(word)
                window_freq = defaultdict(int)
                for i in range(L - n + 1):
                    p = tuple(word[i:i+n])
                    window_freq[p] += 1
                
                unique_windows = []
                for i in range(L - n + 1):
                    if window_freq[tuple(word[i:i+n])] == 1:
                        unique_windows.append(i)
                
                unique_count_at = [0] * L
                for uw in unique_windows:
                    for j in range(uw, uw + n):
                        unique_count_at[j] += 1
                
                candidates = []
                for p2 in range(1, L - 1):
                    if unique_count_at[p2] == 0:
                        candidates.append((0, p2))
                    elif unique_count_at[p2] <= 2:
                        candidates.append((unique_count_at[p2], p2))
                candidates.sort()
    
    # Step 4: End trimming
    if time.time() < deadline - 10:
        changed = True
        while changed:
            changed = False
            L = len(word)
            if L > n:
                first_perm = tuple(word[:n])
                count = 0
                for i in range(L - n + 1):
                    if tuple(word[i:i+n]) == first_perm:
                        count += 1
                if count > 1:
                    word = word[1:]
                    changed = True
            
            if L > n:
                L = len(word)
                last_perm = tuple(word[-n:])
                count = 0
                for i in range(L - n + 1):
                    if tuple(word[i:i+n]) == last_perm:
                        count += 1
                if count > 1:
                    word = word[:-1]
                    changed = True
    
    # Step 5: Second round with more aggressive search
    if time.time() < deadline - 10:
        L = len(word)
        window_freq = defaultdict(int)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            window_freq[p] += 1
        
        unique_count_at = [0] * L
        for i in range(L - n + 1):
            if window_freq[tuple(word[i:i+n])] == 1:
                for j in range(i, i + n):
                    unique_count_at[j] += 1
        
        # Try all positions, not just low-unique ones
        candidates2 = []
        for pos in range(1, L - 1):
            candidates2.append((unique_count_at[pos], pos))
        candidates2.sort()
        
        for _, pos in candidates2:
            if time.time() > deadline - 5:
                break
            if pos >= len(word):
                continue
            
            if is_deletion_safe(word, pos, n, window_freq):
                word = word[:pos] + word[pos+1:]
                deleted_count += 1
                
                if deleted_count % 200 == 0:
                    L = len(word)
                    window_freq = defaultdict(int)
                    for i in range(L - n + 1):
                        p = tuple(word[i:i+n])
                        window_freq[p] += 1
                    
                    unique_count_at = [0] * L
                    for i in range(L - n + 1):
                        if window_freq[tuple(word[i:i+n])] == 1:
                            for j in range(i, i + n):
                                unique_count_at[j] += 1
                    
                    candidates2 = []
                    for pos2 in range(1, L - 1):
                        candidates2.append((unique_count_at[pos2], pos2))
                    candidates2.sort()
    
    # Step 6: More end trimming
    if time.time() < deadline - 5:
        changed = True
        while changed:
            changed = False
            L = len(word)
            if L > n:
                first_perm = tuple(word[:n])
                count = 0
                for i in range(L - n + 1):
                    if tuple(word[i:i+n]) == first_perm:
                        count += 1
                if count > 1:
                    word = word[1:]
                    changed = True
            
            if L > n:
                L = len(word)
                last_perm = tuple(word[-n:])
                count = 0
                for i in range(L - n + 1):
                    if tuple(word[i:i+n]) == last_perm:
                        count += 1
                if count > 1:
                    word = word[:-1]
                    changed = True
    
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
