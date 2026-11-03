import time
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    total_space = q ** n
    full_mask = (1 << total_space) - 1

    # Precompute powers of q
    q_powers = [1]
    for i in range(1, n):
        q_powers.append(q_powers[-1] * q)

    # Precompute neighbor masks (Hamming ball of given radius)
    neighbor_masks = [0] * total_space
    for i in range(total_space):
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % q)
            temp //= q

        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = digits[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    delta = (val - orig_val) * q_powers[pos]
                    new_index = i + delta
                    if 0 <= new_index < total_space:
                        mask |= (1 << new_index)

        neighbor_masks[i] = mask

    # Helper: index -> list of digits
    def idx_to_digits(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        return digits

    # Helper: list of digits -> index
    def digits_to_idx(digits):
        idx = 0
        for pos, val in enumerate(digits):
            idx += val * q_powers[pos]
        return idx

    # ---- Phase 1: Structured initialization using tetracode product ----
    # Tetracode: length 4 over {0,1,2}, 9 codewords (from [4,2,2] ternary code)
    # The tetracode is the set of all words (a, b, a+b, 2a+b) mod 3 for a,b in {0,1,2}
    tetracode = []
    for a in range(q):
        for b in range(q):
            w = [a, b, (a + b) % q, (2 * a + b) % q]
            tetracode.append(w)
    # tetracode has q^2 = 9 words for q=3

    # Product code: combine two tetracode words -> length 8 word
    # This gives 9 * 9 = 81 codewords
    # The product code is a covering code with radius 2, but we need radius 1.
    # Actually, the product of two tetracodes (each radius 1) gives radius 2 for the product.
    # We need a different approach for radius 1.
    # Let's use the greedy approach but start from a good structured set.

    # For radius 1, the tetracode product isn't directly applicable.
    # Instead, let's use a structured greedy: start with codewords from a grid pattern.

    # Grid-based initialization: place codewords at regular intervals
    # For n=8, q=3, we can use a lattice-like structure.
    # Simple approach: use all words where positions 0,1,2,3 are free and 4,5,6,7 are determined.
    # But that's 3^4 = 81 words, which might not cover everything.

    # Let's just use a smarter greedy: start with a few well-chosen codewords.
    # Criticality-driven greedy: pick codeword that covers the most *uncovered* points,
    # but weight by how "hard" those points are to cover (i.e., how few codewords cover them).

    # Initialize with empty code
    code_indices = []
    uncovered = full_mask

    # Build initial coverage counts (how many codewords cover each point)
    # Start with 0 coverage for all
    coverage = np.zeros(total_space, dtype=np.int32)

    # Criticality-driven greedy
    # For each step, find the word that maximizes the sum of 1/coverage[p] for uncovered p in its neighborhood
    # This is expensive, so we approximate: just use standard greedy (max uncovered covered)
    # but process in a structured order.

    # Let's do standard greedy first to get a valid code, then optimize.
    while uncovered != 0:
        best_word = -1
        max_cover = 0

        for w in range(total_space):
            cover_count = (uncovered & neighbor_masks[w]).bit_count()

            if cover_count > max_cover:
                max_cover = cover_count
                best_word = w

        if best_word == -1:
            break

        code_indices.append(best_word)
        uncovered &= ~neighbor_masks[best_word]

        # Update coverage
        mask = neighbor_masks[best_word]
        m = mask
        while m:
            lb = m & (-m)
            pos = lb.bit_length() - 1
            coverage[pos] += 1
            m ^= lb

        if time.time() > deadline - 20:
            break

    # Verify coverage
    covered = 0
    for idx in code_indices:
        covered |= neighbor_masks[idx]

    if covered != full_mask:
        uncovered = full_mask & ~covered
        while uncovered != 0 and time.time() < deadline - 10:
            best_word = -1
            max_cover = 0
            for w in range(total_space):
                cover_count = (uncovered & neighbor_masks[w]).bit_count()
                if cover_count > max_cover:
                    max_cover = cover_count
                    best_word = w
            if best_word == -1:
                break
            code_indices.append(best_word)
            uncovered &= ~neighbor_masks[best_word]

    # ---- Phase 2: Local search with 2-opt pair replacement ----
    # Try to replace pairs of codewords with a single codeword that covers their union
    # Also try single removals

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_size = len(code_indices)

        # First, try single removals
        for i in range(current_size):
            if time.time() > deadline - 5:
                break
            idx = code_indices[i]
            mask = neighbor_masks[idx]
            m = mask
            redundant = True
            while m:
                lb = m & (-m)
                pos = lb.bit_length() - 1
                if coverage[pos] <= 1:
                    redundant = False
                    break
                m ^= lb

            if redundant:
                # Remove codeword i
                m = mask
                while m:
                    lb = m & (-m)
                    pos = lb.bit_length() - 1
                    coverage[pos] -= 1
                    m ^= lb
                code_indices.pop(i)
                improved = True
                break  # Restart the loop after a removal

        if improved:
            continue

        # Try pair replacement: for each pair (i, j), see if there's a single word
        # that covers all points covered by i and j (that aren't covered by others)
        # This is expensive, so we limit the search.
        current_size = len(code_indices)
        for i in range(current_size):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, current_size):
                if time.time() > deadline - 5:
                    break

                # Compute the set of points that are ONLY covered by i or j (not by others)
                # These are the points that would become uncovered if we remove both i and j
                mask_i = neighbor_masks[code_indices[i]]
                mask_j = neighbor_masks[code_indices[j]]

                # Points covered by i or j
                union_mask = mask_i | mask_j

                # Points covered by other codewords
                other_cover = 0
                for k in range(current_size):
                    if k != i and k != j:
                        other_cover |= neighbor_masks[code_indices[k]]

                # Points that are only covered by i or j
                critical = union_mask & ~other_cover

                # If critical is empty, we can remove both i and j
                if critical == 0:
                    # Remove both
                    m = mask_i
                    while m:
                        lb = m & (-m)
                        pos = lb.bit_length() - 1
                        coverage[pos] -= 1
                        m ^= lb
                    m = mask_j
                    while m:
                        lb = m & (-m)
                        pos = lb.bit_length() - 1
                        coverage[pos] -= 1
                        m ^= lb
                    # Remove j first (higher index) to avoid index shifting
                    code_indices.pop(j)
                    code_indices.pop(i)
                    improved = True
                    break

                # Try to find a single word that covers all critical points
                # This is a set cover problem on the critical points
                # For small critical sets, we can brute force
                critical_list = []
                m = critical
                while m:
                    lb = m & (-m)
                    pos = lb.bit_length() - 1
                    critical_list.append(pos)
                    m ^= lb

                if len(critical_list) <= 8:  # Limit search to small critical sets
                    # Try all words to find one that covers all critical points
                    best_replacement = -1
                    for w in range(total_space):
                        mask_w = neighbor_masks[w]
                        # Check if w covers all critical points
                        if (mask_w & critical) == critical:
                            # Also check if w doesn't introduce new uncovered points
                            # (it shouldn't, since we're replacing i and j)
                            best_replacement = w
                            break

                    if best_replacement != -1:
                        # Replace i and j with best_replacement
                        m = mask_i
                        while m:
                            lb = m & (-m)
                            pos = lb.bit_length() - 1
                            coverage[pos] -= 1
                            m ^= lb
                        m = mask_j
                        while m:
                            lb = m & (-m)
                            pos = lb.bit_length() - 1
                            coverage[pos] -= 1
                            m ^= lb
                        code_indices.pop(j)
                        code_indices.pop(i)
                        code_indices.append(best_replacement)
                        m = neighbor_masks[best_replacement]
                        while m:
                            lb = m & (-m)
                            pos = lb.bit_length() - 1
                            coverage[pos] += 1
                            m ^= lb
                        improved = True
                        break

            if improved:
                break

    # Final verification
    covered = 0
    for idx in code_indices:
        covered |= neighbor_masks[idx]

    if covered != full_mask:
        uncovered = full_mask & ~covered
        while uncovered != 0 and time.time() < deadline - 2:
            best_word = -1
            max_cover = 0
            for w in range(total_space):
                cover_count = (uncovered & neighbor_masks[w]).bit_count()
                if cover_count > max_cover:
                    max_cover = cover_count
                    best_word = w
            if best_word == -1:
                break
            code_indices.append(best_word)
            uncovered &= ~neighbor_masks[best_word]

    # Convert indices to lists of integers
    result = []
    for idx in code_indices:
        result.append(idx_to_digits(idx))

    return result
