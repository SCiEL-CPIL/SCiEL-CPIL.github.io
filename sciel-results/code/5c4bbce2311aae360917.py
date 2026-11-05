import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    # ---- 1. Build standard recursive superpermutation (length = sum_{k=1..n} k!) ----
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
            max_ov = min(len(new), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new.extend(block[overlap:])
        word = new

    S = word[:]
    best_S = S[:]
    best_len = len(S)

    # ---- 2. Initialize window counts ----
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i + n])
        counts[p] = counts.get(p, 0) + 1

    # ---- 3. Local search: removal + swap-to-unlock ----
    def try_removal_pass():
        """Attempt to remove any character. Returns (num_removed, S, counts)."""
        removed = 0
        i = len(S) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            # Old windows affected: those that include position i
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)

            valid = True
            old_perms = []
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms.append(p)

            if not valid:
                i -= 1
                continue

            # New straddling windows after removal
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 1 - n)

            new_perms = []
            for k in range(start_new, end_new + 1):
                window_chars = []
                for idx in range(k, k + n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                new_perms.append(tuple(window_chars))

            # Commit removal
            for p in old_perms:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            for p in new_perms:
                counts[p] = counts.get(p, 0) + 1

            S.pop(i)
            removed += 1
            # Do not decrement i: the next position i-1 is now at the same index
            # (since we removed i, positions < i are unchanged)
        return removed, S, counts

    def try_swap_unlock(i):
        """
        Try swapping S[i] with a nearby S[j]. For each swap, check if it enables
        at least one removal. If so, perform the swap and return the new S/counts
        with the swap applied (but no removal yet). Returns None if no helpful swap.
        """
        if time.time() > deadline:
            return None
        # Try swaps with positions within a window of i
        window_size = min(n + 2, len(S) // 4)
        for j in range(max(0, i - window_size), min(len(S), i + window_size + 1)):
            if j == i:
                continue
            if time.time() > deadline:
                break

            # Simulate swap S[i] <-> S[j]
            S[i], S[j] = S[j], S[i]

            # Check if any position now becomes removable
            # We only need to check positions near i and j
            candidates = set()
            for pos in range(max(0, i - n + 1), min(len(S), i + n)):
                candidates.add(pos)
            for pos in range(max(0, j - n + 1), min(len(S), j + n)):
                candidates.add(pos)

            found_removal = False
            for pos in candidates:
                start_old = max(0, pos - n + 1)
                end_old = min(pos, len(S) - n)
                removable = True
                for k in range(start_old, end_old + 1):
                    p = tuple(S[k:k + n])
                    if counts.get(p, 0) == 1:
                        removable = False
                        break
                if removable:
                    found_removal = True
                    break

            if found_removal:
                return i, j  # swap is beneficial, keep it
            else:
                # Undo swap
                S[i], S[j] = S[j], S[i]
        return None

    # ---- Main search loop ----
    stuck_rounds = 0
    max_stuck_rounds = 5

    while time.time() < deadline:
        # Phase 1: removal pass
        removed, S, counts = try_removal_pass()
        if removed > 0:
            if len(S) < best_len:
                best_len = len(S)
                best_S = S[:]
            stuck_rounds = 0
        else:
            stuck_rounds += 1

        if stuck_rounds >= max_stuck_rounds:
            # Try swap-to-unlock: find a swap that enables a removal
            if time.time() > deadline:
                break
            found_swap = False
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    break
                result = try_swap_unlock(i)
                if result is not None:
                    si, sj = result
                    S[si], S[sj] = S[sj], S[si]
                    found_swap = True
                    break
            if found_swap:
                stuck_rounds = 0
            else:
                break  # truly stuck, no swap helps

    return "".join(map(str, best_S))
