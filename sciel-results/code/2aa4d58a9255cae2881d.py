import time
import sys
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate standard recursive superpermutation
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
            max_check = min(len(new), len(block))
            for o in range(max_check, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    L = len(word)

    # Integer encoding for windows
    base = n

    def encode_from_list(arr, start):
        val = 0
        for i in range(n):
            val = val * base + arr[start + i]
        return val

    # Initial window counts
    window_counts = defaultdict(int)
    for i in range(L - n + 1):
        w = encode_from_list(word, i)
        window_counts[w] += 1

    total_perms = 1
    for i in range(1, n + 1):
        total_perms *= i

    missing = total_perms - len(window_counts)

    # Deletion optimization
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(L):
            if time.time() > deadline:
                break

            # Compute windows removed (containing position i)
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            removed_keys = []
            temp_missing = missing
            for j in range(start_min, start_max + 1):
                w = encode_from_list(word, j)
                removed_keys.append(w)
                window_counts[w] -= 1
                if window_counts[w] == 0:
                    del window_counts[w]
                    temp_missing += 1

            # Compute new windows after deletion
            L_prime = L - 1
            added_keys = []
            if L_prime >= n:
                new_start_min = max(0, i - n + 1)
                new_start_max = min(i - 1, L_prime - n)
                for k in range(new_start_min, new_start_max + 1):
                    if k + n <= i:
                        w = encode_from_list(word, k)
                    elif k >= i:
                        w = encode_from_list(word, k + 1)
                    else:
                        left_len = i - k
                        right_len = n - left_len
                        val = 0
                        for idx in range(k, i):
                            val = val * base + word[idx]
                        for idx in range(i + 1, i + 1 + right_len):
                            val = val * base + word[idx]
                        w = val
                    added_keys.append(w)
                    if w not in window_counts:
                        window_counts[w] = 0
                        temp_missing -= 1
                    window_counts[w] += 1

            if temp_missing == 0:
                del word[i]
                L -= 1
                improved = True
                break
            else:
                # Rollback
                for w in removed_keys:
                    window_counts[w] += 1
                for w in added_keys:
                    window_counts[w] -= 1
                    if window_counts[w] == 0:
                        del window_counts[w]

    # Swap optimization: try swapping adjacent elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(L - 1):
            if time.time() > deadline:
                break
            if word[i] == word[i + 1]:
                continue

            # Affected window starts: [max(0, i-n+1), min(i+1, L-n)]
            start_min = max(0, i - n + 1)
            start_max = min(i + 1, L - n)

            # Save old window keys
            old_keys = []
            for j in range(start_min, start_max + 1):
                old_keys.append(encode_from_list(word, j))

            # Perform swap
            word[i], word[i + 1] = word[i + 1], word[i]

            # Compute new window keys
            new_keys = []
            for j in range(start_min, start_max + 1):
                new_keys.append(encode_from_list(word, j))

            # Check if swap is valid: no permutation is lost
            # A permutation is lost if its only occurrence was in old_keys and it's not in new_keys
            valid = True
            for w in old_keys:
                if window_counts[w] == 1:
                    # Only occurrence was here; check if it appears in new_keys
                    if w not in new_keys:
                        valid = False
                        break

            if valid:
                # Apply swap: update counts
                for w in old_keys:
                    window_counts[w] -= 1
                    if window_counts[w] == 0:
                        del window_counts[w]
                for w in new_keys:
                    if w not in window_counts:
                        window_counts[w] = 0
                        missing -= 1
                    window_counts[w] += 1
                improved = True
                break
            else:
                # Rollback swap
                word[i], word[i + 1] = word[i + 1], word[i]

    return "".join(map(str, word))
