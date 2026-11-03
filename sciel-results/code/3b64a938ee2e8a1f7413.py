import math
import time
import sys

def construct(n):
    # Classic recursive construction: length sum(i! for i=1..n)
    word = [1]
    for m in range(2, n + 1):
        # Collect distinct (m-1)-permutations from current word
        seen = set()
        perms = []
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        # Build new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap between tail of new and head of block
            max_o = min(len(new), len(block))
            overlap = 0
            for o in range(max_o, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n > 7:
        return "".join(map(str, word))

    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    target_count = int(math.factorial(n))

    # Verify initial validity
    N = len(word)
    perm_counts = {}
    for i in range(N - n + 1):
        w = word[i:i + n]
        if len(set(w)) == n:
            p = tuple(w)
            perm_counts[p] = perm_counts.get(p, 0) + 1
    if len(perm_counts) != target_count:
        # Fallback: return as-is if somehow invalid (shouldn't happen)
        return "".join(map(str, word))

    # --- Greedy deletion pass (repeat until no more deletions) ---
    improved = True
    while improved and time.time() < deadline:
        improved = False
        N = len(word)
        num_windows = N - n + 1

        for i in range(N - 1, -1, -1):
            if time.time() > deadline:
                break

            # Windows in old word containing position i:
            # window j covers positions j..j+n-1, so i is in it iff j <= i <= j+n-1
            # i.e., j in [i-n+1, i] clamped to [0, num_windows-1]
            win_lo = max(0, i - n + 1)
            win_hi = min(i, num_windows - 1)

            # Quick reject: if any window containing i is a unique perm, skip
            can_delete = True
            for j in range(win_lo, win_hi + 1):
                w = word[j:j + n]
                if len(set(w)) == n:
                    p = tuple(w)
                    if perm_counts[p] == 1:
                        can_delete = False
                        break
            if not can_delete:
                continue

            # Perform deletion: save old affected windows, remove, add new
            old_affected = []
            for j in range(win_lo, win_hi + 1):
                w = word[j:j + n]
                if len(set(w)) == n:
                    p = tuple(w)
                    old_affected.append(p)
                    perm_counts[p] -= 1
                    if perm_counts[p] == 0:
                        del perm_counts[p]

            new_word = word[:i] + word[i + 1:]
            new_N = N - 1
            new_num_windows = new_N - n + 1

            # New affected windows: those crossing the gap at position i in new_word
            # In new_word, the gap is between indices i-1 and i
            # Window j covers j..j+n-1; crosses gap iff j <= i-1 and j+n-1 >= i
            # i.e., j in [i-n+1, i-1] clamped to [0, new_num_windows-1]
            new_win_lo = max(0, i - n + 1)
            new_win_hi = min(i - 1, new_num_windows - 1)

            new_affected = []
            for j in range(new_win_lo, new_win_hi + 1):
                w = new_word[j:j + n]
                if len(set(w)) == n:
                    p = tuple(w)
                    new_affected.append(p)
                    perm_counts[p] = perm_counts.get(p, 0) + 1

            # Check validity: all target perms must be present
            if len(perm_counts) == target_count:
                word = new_word
                improved = True
                break  # Restart scan from end
            else:
                # Revert
                for p in new_affected:
                    perm_counts[p] -= 1
                    if perm_counts[p] == 0:
                        del perm_counts[p]
                for p in old_affected:
                    perm_counts[p] = perm_counts.get(p, 0) + 1

    # --- Second pass: try block moves (move a block of length 2-3 to another position) ---
    if time.time() < deadline:
        N = len(word)
        block_sizes = [2, 3]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            N = len(word)
            num_windows = N - n + 1

            for bsz in block_sizes:
                if time.time() > deadline:
                    break
                for src in range(0, N - bsz + 1):
                    if time.time() > deadline:
                        break
                    # Skip if moving block would destroy a unique perm
                    # Windows affected by removal of block [src, src+bsz-1]:
                    # j in [src-bsz-n+1... src+n-1] roughly, but let's be precise
                    # A window j is affected if it overlaps [src, src+bsz-1]
                    # j <= src+bsz-1 and j+n-1 >= src => j in [src-n+1, src+bsz-1]
                    rem_lo = max(0, src - n + 1)
                    rem_hi = min(src + bsz - 1, num_windows - 1)

                    has_unique = False
                    for j in range(rem_lo, rem_hi + 1):
                        w = word[j:j + n]
                        if len(set(w)) == n:
                            p = tuple(w)
                            if perm_counts[p] == 1:
                                has_unique = True
                                break
                    if has_unique:
                        continue

                    # Try inserting after each position dst (dst is index after which to insert)
                    # To avoid too many trials, limit dst to a window around src
                    dst_lo = max(0, src - 20)
                    dst_hi = min(N - 1, src + 20)
                    found = False
                    for dst in range(dst_hi, dst_lo - 1, -1):
                        if time.time() > deadline:
                            break
                        if dst == src - 1 or dst == src + bsz - 1:
                            continue  # No-op
                        # Build new word: remove block, insert after dst
                        # After removal, indices shift
                        new_word = word[:src] + word[src + bsz:]
                        # Insert position: after dst in original, but adjusted for removal
                        # If dst >= src + bsz, insert at dst - bsz + 1 (after removal, shifted left by bsz)
                        # If dst < src, insert at dst + 1
                        if dst >= src + bsz:
                            ins_pos = dst - bsz + 1
                        else:
                            ins_pos = dst + 1
                        new_word = new_word[:ins_pos] + word[src:src + bsz] + new_word[ins_pos:]

                        if len(new_word) != N:
                            continue

                        # Check validity by recomputing affected windows
                        # This is complex; for now, do a full check only if quick pass
                        # Full check: O(N*n)
                        new_perm_counts = {}
                        for j in range(N - n + 1):
                            w = new_word[j:j + n]
                            if len(set(w)) == n:
                                p = tuple(w)
                                new_perm_counts[p] = new_perm_counts.get(p, 0) + 1
                        if len(new_perm_counts) == target_count:
                            # Verify no perm count went to 0 (should be covered by len check)
                            word = new_word
                            perm_counts = new_perm_counts
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break

    # --- Third pass: try swapping adjacent symbols (length-preserving) ---
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            N = len(word)
            num_windows = N - n + 1

            for i in range(N - 1):
                if time.time() > deadline:
                    break
                if word[i] == word[i + 1]:
                    continue

                # Windows affected by swapping positions i and i+1:
                # j in [i-n+1, i+1] clamped
                sw_lo = max(0, i - n + 1)
                sw_hi = min(i + 1, num_windows - 1)

                # Check if any unique perm is affected
                has_unique = False
                for j in range(sw_lo, sw_hi + 1):
                    w = word[j:j + n]
                    if len(set(w)) == n:
                        p = tuple(w)
                        if perm_counts[p] == 1:
                            has_unique = True
                            break
                if has_unique:
                    continue

                # Perform swap
                word[i], word[i + 1] = word[i + 1], word[i]

                # Update perm_counts
                old_affected = []
                # Recompute: we already swapped, so current word has the swap
                # We need to know what the old windows were. Let's revert, compute, then redo.
                # Actually, let's just recompute affected windows in the new state and check.
                # But we need to know old counts. Let's do: save old affected, undo swap, update, redo swap, update.
                # Simpler: revert, save old, swap, add new, check, revert if bad.
                word[i], word[i + 1] = word[i + 1], word[i]  # Undo swap (back to original)

                old_affected = []
                for j in range(sw_lo, sw_hi + 1):
                    w = word[j:j + n]
                    if len(set(w)) == n:
                        p = tuple(w)
                        old_affected.append(p)
                        perm_counts[p] -= 1
                        if perm_counts[p] == 0:
                            del perm_counts[p]

                word[i], word[i + 1] = word[i + 1], word[i]  # Do swap

                new_affected = []
                for j in range(sw_lo, sw_hi + 1):
                    w = word[j:j + n]
                    if len(set(w)) == n:
                        p = tuple(w)
                        new_affected.append(p)
                        perm_counts[p] = perm_counts.get(p, 0) + 1

                if len(perm_counts) == target_count:
                    improved = True
                else:
                    # Revert
                    for p in new_affected:
                        perm_counts[p] -= 1
                        if perm_counts[p] == 0:
                            del perm_counts[p]
                    for p in old_affected:
                        perm_counts[p] = perm_counts.get(p, 0) + 1
                    word[i], word[i + 1] = word[i + 1], word[i]

    # Final deletion pass after swaps/moves
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            N = len(word)
            num_windows = N - n + 1

            for i in range(N - 1, -1, -1):
                if time.time() > deadline:
                    break

                win_lo = max(0, i - n + 1)
                win_hi = min(i, num_windows - 1)

                can_delete = True
                for j in range(win_lo, win_hi + 1):
                    w = word[j:j + n]
                    if len(set(w)) == n:
                        p = tuple(w)
                        if perm_counts[p] == 1:
                            can_delete = False
                            break
                if not can_delete:
                    continue

                old_affected = []
                for j in range(win_lo, win_hi + 1):
                    w = word[j:j + n]
                    if len(set(w)) == n:
                        p = tuple(w)
                        old_affected.append(p)
                        perm_counts[p] -= 1
                        if perm_counts[p] == 0:
                            del perm_counts[p]

                new_word = word[:i] + word[i + 1:]
                new_N = N - 1
                new_num_windows = new_N - n + 1

                new_win_lo = max(0, i - n + 1)
                new_win_hi = min(i - 1, new_num_windows - 1)

                new_affected = []
                for j in range(new_win_lo, new_win_hi + 1):
                    w = new_word[j:j + n]
                    if len(set(w)) == n:
                        p = tuple(w)
                        new_affected.append(p)
                        perm_counts[p] = perm_counts.get(p, 0) + 1

                if len(perm_counts) == target_count:
                    word = new_word
                    improved = True
                    break
                else:
                    for p in new_affected:
                        perm_counts[p] -= 1
                        if perm_counts[p] == 0:
                            del perm_counts[p]
                    for p in old_affected:
                        perm_counts[p] = perm_counts.get(p, 0) + 1

    return "".join(map(str, word))
