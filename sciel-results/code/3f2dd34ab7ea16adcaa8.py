import itertools
import time
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

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
    window_positions = defaultdict(list)  # perm -> list of start positions
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        window_freq[p] += 1
        window_positions[p].append(i)

    # Verify all 5040 permutations present
    required = set(itertools.permutations(range(1, n+1)))
    present = set(window_freq.keys())
    if present != required:
        # Shouldn't happen with standard construction, but safety
        return "".join(map(str, word))

    # Step 3: Efficient deletion using local verification
    # Key insight: deleting position pos only affects windows [pos-n+1, pos]
    # A deletion is safe iff for every affected window, its permutation still has freq >= 1 after removal

    # Precompute critical array: critical[i] = True if window at position i appears exactly once
    L = len(word)
    critical = [False] * (L - n + 1)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        critical[i] = (window_freq[p] == 1)

    # Build priority list: positions where all affected windows are non-critical
    # Score by number of redundant windows (higher = more likely to be removable)
    candidates = []
    for pos in range(1, L - 1):
        lo = max(0, pos - n + 1)
        hi = min(pos, L - n)
        if lo > hi:
            continue
        redundant_count = 0
        all_redundant = True
        for w in range(lo, hi + 1):
            if critical[w]:
                all_redundant = False
                break
            redundant_count += 1
        if all_redundant and redundant_count > 0:
            candidates.append((-redundant_count, pos))

    candidates.sort()

    deleted_count = 0
    for _, pos in candidates:
        if time.time() > deadline - 10:
            break

        if pos >= len(word):
            continue

        # Local check: only verify windows affected by deletion
        test_word = word[:pos] + word[pos+1:]
        new_len = len(test_word)
        if new_len < n:
            continue

        # Windows affected: those starting at max(0, pos-n+1) to min(pos, new_len-n)
        lo = max(0, pos - n + 1)
        hi = min(pos, new_len - n)

        safe = True
        for w in range(lo, hi + 1):
            # Check if this window's permutation still exists elsewhere in test_word
            # We need to verify that the permutation at window w (in test_word) 
            # is still present in the word after deletion
            # Since we deleted pos, windows shifted:
            # - Windows before pos are unchanged
            # - Windows after pos shift left by 1
            # The key question: is the permutation at each affected window still covered?
            
            # For window starting at w in test_word:
            # If w < pos: same as original word
            # If w >= pos: corresponds to original window at w+1
            # But we need to check if this specific permutation appears elsewhere
            
            # Simpler approach: just check if the permutation is in the frequency map
            # and appears at least once in the new word
            p = tuple(test_word[w:w+n])
            # Check if p appears in test_word at some position other than w
            # This is expensive, so use a different strategy:
            # Track which permutations are "lost" by this deletion
            
            # Actually, the safest local check:
            # For each affected window, verify its permutation still has count >= 1
            # We can do this by checking if the permutation appears in the original
            # word at a position that's not deleted and not in the affected range
            
            # Let's use a simpler but correct approach:
            # After deletion, for each affected window position w, get the permutation
            # and check if it appears elsewhere in the new word
            # This is O(n^2) per deletion which is acceptable for n=7
            
            # Check if p appears in test_word at any position other than w
            found_elsewhere = False
            for check_pos in range(new_len - n + 1):
                if check_pos == w:
                    continue
                if tuple(test_word[check_pos:check_pos+n]) == p:
                    found_elsewhere = True
                    break
            if not found_elsewhere:
                safe = False
                break

        if safe:
            word = test_word
            deleted_count += 1

            # Periodically rebuild critical array and candidates
            if deleted_count % 50 == 0 and time.time() < deadline - 15:
                L = len(word)
                window_freq = defaultdict(int)
                for i in range(L - n + 1):
                    p = tuple(word[i:i+n])
                    window_freq[p] += 1

                critical = [False] * (L - n + 1)
                for i in range(L - n + 1):
                    p = tuple(word[i:i+n])
                    critical[i] = (window_freq[p] == 1)

                # Rebuild candidates
                candidates = []
                for pos in range(1, L - 1):
                    lo = max(0, pos - n + 1)
                    hi = min(pos, L - n)
                    if lo > hi:
                        continue
                    all_redundant = True
                    redundant_count = 0
                    for w in range(lo, hi + 1):
                        if critical[w]:
                            all_redundant = False
                            break
                        redundant_count += 1
                    if all_redundant and redundant_count > 0:
                        candidates.append((-redundant_count, pos))
                candidates.sort()

        if deleted_count >= 500:
            break

    # Step 4: Try end trimming
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

    # Step 5: Second round of internal deletions (after trimming)
    if time.time() < deadline - 5:
        L = len(word)
        window_freq = defaultdict(int)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            window_freq[p] += 1

        critical = [False] * (L - n + 1)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            critical[i] = (window_freq[p] == 1)

        # Try deleting positions from left to right, skipping critical ones
        pos = 1
        while pos < len(word) - 1:
            if time.time() > deadline - 3:
                break

            lo = max(0, pos - n + 1)
            hi = min(pos, len(word) - n)

            safe = True
            for w in range(lo, hi + 1):
                if critical[w]:
                    safe = False
                    break

            if safe:
                test_word = word[:pos] + word[pos+1:]
                new_len = len(test_word)
                if new_len >= n:
                    # Local verification
                    lo2 = max(0, pos - n + 1)
                    hi2 = min(pos, new_len - n)
                    local_safe = True
                    for w in range(lo2, hi2 + 1):
                        p = tuple(test_word[w:w+n])
                        found = False
                        for cp in range(new_len - n + 1):
                            if cp == w:
                                continue
                            if tuple(test_word[cp:cp+n]) == p:
                                found = True
                                break
                        if not found:
                            local_safe = False
                            break
                    if local_safe:
                        word = test_word
                        continue

            pos += 1

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
