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

    # Fast deletion check:
    # Deleting position pos creates new windows at positions [max(0,pos-n+1), min(pos, new_len-n)]
    # in the new word. For each such new window, we need its permutation to appear
    # at least once elsewhere in the new word.
    #
    # Key insight: A new window at position w in the new word (where w >= pos-n+1 and w <= pos)
    # corresponds to original word positions [w, pos) + [pos+1, w+n+1) (skipping pos).
    # The permutation of this new window is:
    #   new_perm = tuple(word[w:pos] + word[pos+1:w+n+1])
    #
    # This new_perm is "safe" if it appears in the new word at some position other than w.
    # It appears at position w' in the new word if:
    #   - w' < pos: original window at w' (unchanged)
    #   - w' >= pos: it's another new window
    #
    # For efficiency, we check: does new_perm appear in the ORIGINAL word at a position
    # that survives the deletion (i.e., not destroyed)?
    # An original window at position q survives if q < pos-n+1 (before affected region)
    # or q > pos (after, shifted left by 1).
    # Windows at positions [pos-n+1, pos] are all affected.
    #
    # So new_perm is safe if:
    #   window_freq[new_perm] >= 2 AND at least one occurrence is outside [pos-n+1, pos]
    #   OR window_freq[new_perm] == 1 AND that single occurrence is outside [pos-n+1, pos]
    #   (in which case the new window at w is a second occurrence)
    #   OR the new window at w itself is the only occurrence but it's fine since it exists.
    #
    # Actually simpler: after deletion, new_perm must appear at least once in new_word.
    # The new window at position w IS one occurrence. So we need at least one MORE occurrence
    # OR we need to verify that w is the only one and it's valid (which it is by construction).
    #
    # Wait - the requirement is that every permutation appears at least once. So if new_perm
    # appears only at position w in the new word, that's fine! We just need it to appear AT LEAST ONCE.
    #
    # So the check is: does new_perm appear in the new word at all?
    # It appears at position w by construction. So it ALWAYS appears at least once!
    #
    # But wait - we also need to make sure we haven't DESTROYED the last occurrence of some
    # OTHER permutation. The windows that are destroyed are the original windows at positions
    # [pos-n+1, pos] (those that contain position pos). If any of those had freq==1, we've
    # lost that permutation UNLESS it reappears as a new window.
    #
    # So the correct check for deletion at pos:
    # For each original window position q in [max(0,pos-n+1), min(pos, L-n)]:
    #   perm_q = tuple(word[q:q+n])
    #   If window_freq[perm_q] == 1:
    #     Check if perm_q appears in the new word at some position.
    #     It could appear as:
    #       - An unaffected window (position < pos-n+1 or > pos in original, shifted)
    #       - A new merged window
    #     Since window_freq[perm_q] == 1, it only appeared at position q in original.
    #     After deletion, it can only appear if one of the new merged windows equals perm_q.
    #     New merged windows are at positions w in [max(0,pos-n+1), min(pos, new_len-n)] in new word.
    #     The new window at position w is: word[w:pos] + word[pos+1:w+n+1] (original indices)
    #     We need to check if any of these equals perm_q.
    #
    # This is still O(n) per unique window. But the number of unique windows containing pos
    # is at most n (since each window has n positions). So total is O(n^2) per candidate.
    #
    # Let me implement this efficiently:

    def is_deletion_safe_fast(word, pos, n, window_freq, L):
        """Fast check if deleting position pos is safe."""
        new_len = L - 1
        if new_len < n:
            return False

        # Affected original window positions: those containing pos
        # Window at position q contains pos if q <= pos <= q+n-1
        # So q is in [pos-n+1, pos], intersected with [0, L-n]
        q_lo = max(0, pos - n + 1)
        q_hi = min(pos, L - n)

        # For each affected window, check if it's unique and if so, verify it reappears
        for q in range(q_lo, q_hi + 1):
            perm_q = tuple(word[q:q+n])
            if window_freq[perm_q] != 1:
                continue  # Not unique, safe

            # perm_q is unique and was at position q. Check if it appears in new word.
            # New windows are at positions w in [max(0, pos-n+1), min(pos, new_len-n)]
            # New window at position w (in new word) = word[w:pos] + word[pos+1:w+n+1]
            # (using original word indices, where w ranges over new word positions)
            #
            # Actually, let me think about this more carefully.
            # In the new word, position w corresponds to:
            #   If w < pos: new_word[w] = word[w], so new_word[w:w+n] = word[w:w+n]
            #   If w >= pos: new_word[w] = word[w+1], so new_word[w:w+n] = word[w+1:w+1+n]
            #     but this skips position pos, so it's word[w+1:pos+1] + word[pos+1:w+1+n]
            #     Wait, that's not right either.
            #
            # Let me be precise. new_word = word[:pos] + word[pos+1:]
            # new_word[w] = word[w] if w < pos, else word[w+1]
            # new_word[w:w+n] = 
            #   if w+n <= pos: word[w:w+n] (all before pos, unchanged)
            #   if w >= pos: word[w+1:w+1+n] (all after pos, shifted)
            #   if w < pos < w+n: word[w:pos] + word[pos+1:w+n+1] (spans the gap)
            #
            # For the spanning case: w < pos < w+n, i.e., pos-n+1 <= w <= pos-1
            # Wait, w < pos and w+n > pos, so w > pos-n. Thus w in [pos-n+1, pos-1].
            # Also w >= 0 and w+n <= new_len, so w <= new_len-n = L-n-1.
            #
            # So the "new" windows (that differ from original) are at positions w in:
            #   [max(0, pos-n+1), min(pos-1, L-n-1)]  (spanning the gap)
            # Windows at w < pos-n+1 are unchanged (before the gap)
            # Windows at w >= pos are shifted (after the gap), new_word[w:w+n] = word[w+1:w+1+n]
            #
            # So for perm_q to appear in new_word:
            #   Case 1: It appears as an unchanged window at position w < pos-n+1
            #     This means word[w:w+n] = perm_q for some w < pos-n+1
            #     But window_freq[perm_q] == 1 and it was at position q >= pos-n+1,
            #     so no such w exists.
            #
            #   Case 2: It appears as a shifted window at position w >= pos
            #     new_word[w:w+n] = word[w+1:w+1+n] = perm_q
            #     This means word[w+1:w+1+n] = perm_q for some w >= pos
            #     i.e., word[q':q'+n] = perm_q for some q' = w+1 >= pos+1
            #     But window_freq[perm_q] == 1 and it was at position q, so q' = q.
            #     So we need q >= pos+1. If q >= pos+1, then yes it appears.
            #     If q < pos+1 (i.e., q <= pos), it doesn't appear this way.
            #
            #   Case 3: It appears as a spanning window at position w in [pos-n+1, pos-1]
            #     new_word[w:w+n] = word[w:pos] + word[pos+1:w+n+1] = perm_q
            #     We need to check if any such w gives perm_q.
            #
            # So: perm_q survives iff (q >= pos+1) OR (there exists w in [max(0,pos-n+1), min(pos-1, L-n-1)]
            #     such that word[w:pos] + word[pos+1:w+n+1] == perm_q)
            #
            # Since q is in [pos-n+1, pos], q >= pos+1 is impossible (q <= pos).
            # So we only need Case 3.

            # Check spanning windows
            w_lo = max(0, pos - n + 1)
            w_hi = min(pos - 1, L - n - 1)
            found = False
            for w in range(w_lo, w_hi + 1):
                # New window at position w: word[w:pos] + word[pos+1:w+n+1]
                # Check if this equals perm_q
                # perm_q = word[q:q+n]
                # We can compare piece by piece for speed
                # word[w:pos] should equal perm_q[0:pos-w]
                # word[pos+1:w+n+1] should equal perm_q[pos-w:n]
                left_len = pos - w
                if left_len < 0 or left_len > n:
                    continue
                # Compare left part
                match = True
                for i in range(left_len):
                    if word[w + i] != perm_q[i]:
                        match = False
                        break
                if not match:
                    continue
                # Compare right part
                for i in range(n - left_len):
                    if word[pos + 1 + i] != perm_q[left_len + i]:
                        match = False
                        break
                if match:
                    found = True
                    break

            if not found:
                return False

        return True

    # Step 3: Iterative deletion with fast verification
    # Maintain data structures for efficiency
    def rebuild_data(word, n):
        L = len(word)
        window_freq = defaultdict(int)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            window_freq[p] += 1
        return L, window_freq

    L, window_freq = rebuild_data(word, n)
    deleted_count = 0

    # Repeatedly find and delete safe positions
    while time.time() < deadline - 20:
        L = len(word)
        if L <= n:
            break

        # Find all positions with minimal "risk" (fewest unique windows containing them)
        unique_count_at = [0] * L
        for i in range(L - n + 1):
            if window_freq[tuple(word[i:i+n])] == 1:
                for j in range(i, i + n):
                    unique_count_at[j] += 1

        # Try positions from lowest unique count, skipping ends
        # Use a simple approach: try all positions in order of unique_count
        min_count = min(unique_count_at[1:-1]) if L > 2 else 0
        candidates = [pos for pos in range(1, L - 1) if unique_count_at[pos] == min_count]

        # If all have high count, try the minimum anyway
        if not candidates:
            candidates = [pos for pos in range(1, L - 1)]

        made_progress = False
        for pos in candidates:
            if time.time() > deadline - 20:
                break
            if pos >= len(word):
                continue

            if is_deletion_safe_fast(word, pos, n, window_freq, L):
                word = word[:pos] + word[pos+1:]
                deleted_count += 1
                made_progress = True
                # Rebuild data structures
                L, window_freq = rebuild_data(word, n)
                break

        if not made_progress:
            # No more deletions possible with current strategy
            break

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

    # Step 5: Second round of deletions after end trimming
    if time.time() < deadline - 15:
        L, window_freq = rebuild_data(word, n)
        while time.time() < deadline - 15:
            L = len(word)
            if L <= n:
                break

            unique_count_at = [0] * L
            for i in range(L - n + 1):
                if window_freq[tuple(word[i:i+n])] == 1:
                    for j in range(i, i + n):
                        unique_count_at[j] += 1

            min_count = min(unique_count_at[1:-1]) if L > 2 else 0
            candidates = [pos for pos in range(1, L - 1) if unique_count_at[pos] == min_count]
            if not candidates:
                candidates = list(range(1, L - 1))

            made_progress = False
            for pos in candidates:
                if time.time() > deadline - 15:
                    break
                if pos >= len(word):
                    continue

                if is_deletion_safe_fast(word, pos, n, window_freq, L):
                    word = word[:pos] + word[pos+1:]
                    made_progress = True
                    L, window_freq = rebuild_data(word, n)
                    break

            if not made_progress:
                break

    # Step 6: Final end trimming
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
