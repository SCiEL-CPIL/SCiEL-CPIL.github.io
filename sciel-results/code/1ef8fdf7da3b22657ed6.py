import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n

    # Precompute the neighbor structure: for each codeword c,
    # the set of words it covers (itself + words at Hamming distance radius).
    # For radius=1 in ternary code length 6: each codeword covers 1 + 6*2 = 13 words.

    # Build B as a list of arrays: B[c] is a boolean array of length total
    B = []
    for c in range(total):
        covered = np.zeros(total, dtype=bool)
        covered[c] = True
        for i in range(n):
            pow_q_i = q ** i
            digit = (c // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    u = c + (new_digit - digit) * pow_q_i
                    covered[u] = True
        B.append(covered)

    # Convert to numpy array for faster matrix-vector ops
    B_arr = np.array(B, dtype=np.int8)

    rng = random.Random(42)
    best_code = None
    best_size = total + 1

    def greedy_cover():
        """Run greedy set cover and return the code (list of indices)."""
        uncovered = np.ones(total, dtype=np.int8)
        code = []

        while np.sum(uncovered) > 0:
            if time.monotonic() > deadline - 3.0:
                break
            # Compute gain for each candidate
            gains = B_arr @ uncovered
            best_c = int(np.argmax(gains))
            if gains[best_c] == 0:
                break
            code.append(best_c)
            uncovered = (uncovered & ~B_arr[best_c].astype(bool)).astype(np.int8)

        return code

    def verify_and_fix(code):
        """Verify code covers everything; if not, add words greedily."""
        counts = np.zeros(total, dtype=np.int8)
        for c in code:
            counts += B_arr[c]
        if np.any(counts == 0):
            uncovered_mask = (counts == 0).astype(np.int8)
            while np.sum(uncovered_mask) > 0 and time.monotonic() < deadline - 2.0:
                gains = B_arr @ uncovered_mask
                best_c = int(np.argmax(gains))
                if gains[best_c] == 0:
                    break
                code.append(best_c)
                counts += B_arr[best_c]
                uncovered_mask = (counts == 0).astype(np.int8)
        return code

    def local_search(code):
        """Try to reduce code size via removals, replacements, and swaps."""
        code = list(code)
        if not code:
            return code, 0

        counts = np.zeros(total, dtype=np.int16)
        for c in code:
            counts += B_arr[c]

        # Verify valid
        if np.any(counts == 0):
            code = verify_and_fix(code)
            counts = np.zeros(total, dtype=np.int16)
            for c in code:
                counts += B_arr[c]

        improved = True
        stall_count = 0

        while improved and time.monotonic() < deadline - 5.0:
            improved = False

            # --- Try removals (in reverse order to keep indices stable) ---
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                mask = B_arr[c] == 1
                if np.any(counts[mask] == 1):
                    continue
                # Safe to remove
                counts -= B_arr[c]
                code.pop(i)
                improved = True
                stall_count = 0
                break

            if improved:
                continue

            # --- Try 1-for-1 replacements ---
            replaced = False
            for i in range(len(code)):
                if replaced:
                    break
                old_c = code[i]
                old_mask = B_arr[old_c] == 1

                # Find words that would lose coverage if old_c is removed
                # These are words where count == 1 and covered by old_c
                vulnerable = old_mask & (counts == 1)
                if not np.any(vulnerable):
                    # Just remove old_c
                    counts -= B_arr[old_c]
                    code.pop(i)
                    improved = True
                    stall_count = 0
                    continue

                # Try random replacements
                for _ in range(20):
                    new_c = rng.randint(0, total - 1)
                    if new_c == old_c or new_c in code:
                        continue

                    new_mask = B_arr[new_c] == 1
                    # Words losing coverage: in old_mask but not new_mask
                    losing = old_mask & ~new_mask
                    if np.any(counts[losing] == 1):
                        continue

                    # Safe replacement
                    counts -= B_arr[old_c]
                    counts += B_arr[new_c]
                    code[i] = new_c
                    improved = True
                    replaced = True
                    stall_count = 0
                    break

            if improved:
                continue

            # --- Try 2-for-1: replace two codewords with one ---
            two_for_one = False
            if len(code) >= 2:
                # Try random pairs
                for _ in range(50):
                    i, j = rng.sample(range(len(code)), 2)
                    c1, c2 = code[i], code[j]
                    # Combined coverage of c1 and c2
                    combined = B_arr[c1] | B_arr[c2]
                    # Words that would lose coverage: in combined but count would drop below 1
                    # After removing c1 and c2, counts for words in B[c1] or B[c2] decrease
                    # A word u is unsafe if counts[u] - (B_arr[c1][u] + B_arr[c2][u]) < 1
                    # i.e., counts[u] < B_arr[c1][u] + B_arr[c2][u]
                    # Since B_arr values are 0/1, this means counts[u] == 1 and (B_arr[c1][u] or B_arr[c2][u])
                    # OR counts[u] == 2 and B_arr[c1][u] and B_arr[c2][u]

                    # Try to find a single codeword new_c that covers all vulnerable words
                    vulnerable_words = combined & (counts <= (B_arr[c1].astype(np.int16) + B_arr[c2].astype(np.int16)))
                    # Actually simpler: words where after removing both, count would be 0
                    # count_after[u] = counts[u] - B[c1][u] - B[c2][u]
                    # unsafe if count_after[u] < 1, i.e., counts[u] - B[c1][u] - B[c2][u] < 1

                    # Find candidate new_c that covers all unsafe words
                    # unsafe_words = where counts[u] < B[c1][u] + B[c2][u]
                    sum_both = (B_arr[c1].astype(np.int16) + B_arr[c2].astype(np.int16))
                    unsafe = (counts < sum_both).astype(bool)

                    if not np.any(unsafe):
                        # Just remove both
                        counts -= B_arr[c1]
                        counts -= B_arr[c2]
                        # Remove j first (higher index), then i
                        code.pop(j)
                        code.pop(i)
                        improved = True
                        two_for_one = True
                        stall_count = 0
                        break

                    # Need a replacement that covers all unsafe words
                    # Try random candidates
                    for __ in range(10):
                        new_c = rng.randint(0, total - 1)
                        if new_c == c1 or new_c == c2 or new_c in code:
                            continue
                        new_mask = B_arr[new_c] == 1
                        # new_c must cover all unsafe words
                        if np.any(unsafe & ~new_mask):
                            continue
                        # Also, new_c must not cause issues elsewhere (it only adds coverage)
                        # Safe!
                        counts -= B_arr[c1]
                        counts -= B_arr[c2]
                        counts += B_arr[new_c]
                        code.pop(j)
                        code.pop(i)
                        code.append(new_c)
                        improved = True
                        two_for_one = True
                        stall_count = 0
                        break

                    if two_for_one:
                        break

            if improved:
                continue

            # --- Try 1-for-2: replace one codeword with two ---
            # This increases size, so only do it if we're stuck and want to escape local opt
            # Skip for now to keep size minimal

            # --- Perturbation: remove a random subset and re-greedy ---
            if stall_count > 0:
                if len(code) > 20 and time.monotonic() < deadline - 10.0:
                    # Remove 10-20% random codewords
                    n_remove = max(3, len(code) // 8)
                    to_remove = set(rng.sample(range(len(code)), n_remove))
                    # Build new counts
                    new_code = [c for idx, c in enumerate(code) if idx not in to_remove]
                    counts = np.zeros(total, dtype=np.int16)
                    for c in new_code:
                        counts += B_arr[c]

                    # Greedily fill
                    uncovered = (counts == 0).astype(np.int8)
                    while np.sum(uncovered) > 0 and time.monotonic() < deadline - 5.0:
                        gains = B_arr @ uncovered
                        best_c = int(np.argmax(gains))
                        if gains[best_c] == 0:
                            break
                        new_code.append(best_c)
                        counts += B_arr[best_c]
                        uncovered = (counts == 0).astype(np.int8)

                    if len(new_code) < len(code):
                        code = new_code
                        improved = True
                        stall_count = 0
                        continue
                    else:
                        # Revert
                        counts = np.zeros(total, dtype=np.int16)
                        for c in code:
                            counts += B_arr[c]

            stall_count += 1

        return code, len(code)

    # Multiple restarts
    num_restarts = 0
    while time.monotonic() < deadline - 10.0:
        num_restarts += 1

        # Greedy initialization
        code = greedy_cover()
        if not code:
            break

        # Local search
        code, size = local_search(code)

        if size < best_size:
            best_size = size
            best_code = code[:]

        # Early termination
        if best_size <= 73:
            # Try a few more restarts
            if num_restarts >= 30:
                break
        elif num_restarts >= 50:
            break

    # Fallback
    if best_code is None:
        best_code = greedy_cover()
        best_code = verify_and_fix(best_code)

    # Convert indices to words
    result = []
    for idx in best_code:
        word = [0] * n
        tmp = idx
        for i in range(n):
            word[i] = tmp % q
            tmp //= q
        result.append(word)

    return result
