"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n):
    """
    Constructs a superpermutation for n symbols.
    Uses the classic recursive construction, then applies removal-based local search
    to minimize length while preserving all permutation windows.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill

    # Base case
    if n == 1:
        return "1"

    # Step 1: Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
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

    S = list(word)
    L = len(S)

    # Step 2: Build count map of all n-length windows
    counts = {}
    for i in range(L - n + 1):
        p = tuple(S[i:i + n])
        counts[p] = counts.get(p, 0) + 1

    best_S = S[:]
    best_len = L

    # Step 3: Removal-based local search
    # Try to remove each character (from right to left) if it doesn't
    # cause any permutation window to drop to count 0.
    while time.time() < deadline:
        removed_any = False
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break

            # Windows in old string that include position i:
            # window starting at j covers S[j..j+n-1], includes i iff j <= i <= j+n-1
            # i.e., j in [i-n+1, i], clipped to [0, len(S)-n]
            j_start = max(0, i - n + 1)
            j_end = min(i, len(S) - n)

            valid = True
            old_perms = []
            for j in range(j_start, j_end + 1):
                p = tuple(S[j:j + n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms.append(p)

            if not valid:
                continue

            # Perform removal: decrement old windows
            for p in old_perms:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]

            # New windows formed by straddling the gap at position i:
            # In new string (len = len(S)-1), window at index k straddles the gap
            # if k < i and k+n-1 >= i, i.e., k in [i-n+1, i-1]
            # clipped to [0, len(S)-1-n]
            new_len = len(S) - 1
            k_start = max(0, i - n + 1)
            k_end = min(i - 1, new_len - n)

            for k in range(k_start, k_end + 1):
                # Build window from S with gap at i
                w = []
                for idx in range(k, k + n):
                    s_idx = idx if idx < i else idx + 1
                    w.append(S[s_idx])
                p_new = tuple(w)
                counts[p_new] = counts.get(p_new, 0) + 1

            # Commit removal
            S.pop(i)
            removed_any = True

            if len(S) < best_len:
                best_len = len(S)
                best_S = S[:]

        # If no removal in this entire pass, we're stuck
        if not removed_any:
            break

    return "".join(map(str, best_S))
# EVOLVE-BLOCK-END
