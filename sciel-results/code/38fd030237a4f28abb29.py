import time
import numpy as np
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 130.0  # Safety margin

    # 1. Generate initial superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 from current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        new = []
        for p in perms:
            # Construct block: p + [m] + p
            block = list(p) + [m] + list(p)
            # Find max overlap with current new
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n <= 4:
        return "".join(map(str, word))

    # 2. Local Search Deletion with NumPy-accelerated counting
    current = np.array(word, dtype=np.int32)
    L = len(current)
    target_perms = 1
    for i in range(2, n + 1):
        target_perms *= i

    def compute_counts(arr, L_val, n_val):
        """Compute permutation counts using NumPy for speed."""
        if L_val < n_val:
            return {}
        # Create sliding windows: shape (L-n+1, n)
        windows = np.lib.stride_tricks.sliding_window_view(arr, n_val)
        # Convert each window to a hashable integer code
        # Code: sum of (val-1) * (n)^(n-1-i) for position i
        # For n=7, max code = 6 * (7^6 + 7^5 + ... + 1) which fits in int64
        base = np.array([n_val ** (n_val - 1 - i) for i in range(n_val)], dtype=np.int64)
        codes = windows.astype(np.int64) @ base
        unique, counts = np.unique(codes, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    perm_count = compute_counts(current, L, n)

    # If we don't have all perms, return as is (should not happen with valid construction)
    if len(perm_count) < target_perms:
        return "".join(map(str, current.tolist()))

    # Precompute base for code computation
    base = np.array([n ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    # Deletion loop with alternating directions
    while time.time() < deadline:
        deleted = False
        # Alternate direction each pass
        if deleted:
            # Continue in same direction after a deletion (restart scan)
            pass
        else:
            # Try both directions
            for direction in ['forward', 'backward']:
                if direction == 'forward':
                    indices = range(L - 1, -1, -1)  # End to start
                else:
                    indices = range(L)  # Start to end

                for i in indices:
                    if time.time() > deadline:
                        break

                    # Identify old windows containing position i
                    start_w = max(0, i - n + 1)
                    end_w = min(i, L - n)

                    # Get old window codes
                    old_codes = []
                    can_delete = True

                    for w in range(start_w, end_w + 1):
                        window = current[w:w + n].astype(np.int64)
                        code = int(window @ base)
                        old_codes.append((w, code))
                        if perm_count[code] == 1:
                            # This permutation only appears here, need to check if it survives
                            pass

                    # Identify new windows formed after deletion
                    # New windows start at j in [max(0, i-n+1), i-1] in the new string
                    # where the window crosses position i
                    new_codes_set = set()
                    for j in range(max(0, i - n + 1), i):
                        # Window in new string starts at j
                        # prefix: current[j:i], suffix: current[i+1 : i+1 + (n - (i-j))]
                        prefix_len = i - j
                        suffix_len = n - prefix_len
                        if prefix_len + suffix_len != n:
                            continue
                        # Check bounds: suffix must fit in current[i+1:]
                        if i + 1 + suffix_len > L:
                            continue
                        # Build the window
                        if prefix_len > 0:
                            prefix = current[j:i]
                        else:
                            prefix = np.array([], dtype=np.int32)
                        if suffix_len > 0:
                            suffix = current[i + 1:i + 1 + suffix_len]
                        else:
                            suffix = np.array([], dtype=np.int32)
                        window = np.concatenate([prefix, suffix]).astype(np.int64)
                        code = int(window @ base)
                        new_codes_set.add(code)

                    # Check: for each old window with count 1, it must be in new_codes_set
                    for w, code in old_codes:
                        if perm_count[code] == 1:
                            if code not in new_codes_set:
                                can_delete = False
                                break

                    if can_delete:
                        # Perform deletion
                        # Update perm_count
                        for w, code in old_codes:
                            perm_count[code] -= 1
                            if perm_count[code] == 0:
                                del perm_count[code]

                        for code in new_codes_set:
                            perm_count[code] = perm_count.get(code, 0) + 1

                        current = np.concatenate([current[:i], current[i + 1:]])
                        L -= 1
                        deleted = True
                        break  # Restart scan

                if deleted:
                    break

        if not deleted:
            break  # No more deletions possible

    return "".join(map(str, current.tolist()))
