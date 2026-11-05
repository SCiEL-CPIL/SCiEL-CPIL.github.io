import numpy as np
import time

def construct():
    N = 17
    MIN_D = 6
    deadline = time.time() + 55  # safety margin before 60s limit

    # Step 1: Generate the 256-word linear [17,8,6] code
    # f1(x) = x^8 + x^4 + x^3 + x^2 + 1
    f1 = [1, 0, 0, 1, 1, 1, 0, 0, 1]

    basis = []
    for i in range(9):
        vec = [0] * 17
        for j in range(len(f1)):
            if i + j < 17:
                vec[i+j] = f1[j]
        basis.append(vec)

    words_17 = []
    for mask in range(512):
        word = [0] * 17
        for i in range(9):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= basis[i][j]
        words_17.append(word)

    # Extend to 18 bits
    words_18 = []
    for w in words_17:
        parity = sum(w) % 2
        words_18.append(w + [parity])

    # Shorten: keep words where bit 0 is 0
    codewords = []
    for w18 in words_18:
        if w18[0] == 0:
            new_word = w18[1:]
            val = 0
            for i, bit in enumerate(new_word):
                if bit:
                    val |= (1 << i)
            codewords.append(val)

    # Convert to bit matrix
    def to_bit_matrix(codes):
        return np.array([[(c >> i) & 1 for i in range(N)] for c in codes], dtype=np.uint8)

    code_bits = to_bit_matrix(codewords)

    # Precompute all 17-bit words as bit matrix
    all_words = np.arange(2**N, dtype=np.int64)
    bits_all = np.array([[(w >> i) & 1 for i in range(N)] for w in all_words], dtype=np.uint8)
    all_weights = np.sum(bits_all, axis=1)

    # Precompute weights for code
    code_weights = np.sum(code_bits, axis=1)

    # Step 2: Greedily add words at distance >= 6 from all existing
    # Use vectorized distance computation in batches
    batch_size = 5000

    # First pass: find all words that can be added
    def find_addable(code_bits, code_weights, bits_all, all_weights, MIN_D, deadline, batch_size=5000):
        addable = []
        n_codes = code_bits.shape[0]
        for start in range(0, 2**N, batch_size):
            if time.time() >= deadline:
                break
            end = min(start + batch_size, 2**N)
            batch = bits_all[start:end]
            batch_w = all_weights[start:end]

            # Compute distances: (batch_size, n_codes)
            dots = batch @ code_bits.T
            dists = batch_w[:, None] + code_weights[None, :] - 2 * dots
            min_dists = np.min(dists, axis=1)
            valid = np.where(min_dists >= MIN_D)[0]
            for idx in valid:
                addable.append(start + idx)
        return addable

    # Iterative greedy addition
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        code_weights = np.sum(code_bits, axis=1)
        addable = find_addable(code_bits, code_weights, bits_all, all_weights, MIN_D, deadline, batch_size)
        if addable:
            # Add all found words (they were all valid simultaneously, but after adding some,
            # others may become invalid - so add one at a time and recheck)
            # Actually, let's add them one by one and verify
            for cand in addable:
                if time.time() >= deadline:
                    break
                # Verify it's still valid
                cand_bits = bits_all[cand]
                dots = code_bits @ cand_bits
                dists = code_weights + all_weights[cand] - 2 * dots
                if np.min(dists) >= MIN_D:
                    codewords.append(cand)
                    code_bits = np.vstack([code_bits, cand_bits[None, :]])
                    code_weights = np.sum(code_bits, axis=1)
                    added_any = True

    # Step 3: Local search - try swapping codewords to allow more additions
    # Strategy: for each codeword, try replacing it with a word at distance >= 6 from all others
    # This can open up space for additional words

    if time.time() < deadline - 5:
        # Try to find a word that's close to being addable (distance 5 from exactly one codeword)
        # and see if we can swap that codeword
        code_weights = np.sum(code_bits, axis=1)

        # Find candidates at distance exactly 5 from some codeword
        # For each codeword, find words at distance 5
        # Then check if that word is at distance >= 6 from all other codewords

        swap_improved = True
        while swap_improved and time.time() < deadline - 2:
            swap_improved = False
            n_codes = len(codewords)
            code_weights = np.sum(code_bits, axis=1)

            # For each codeword, generate all words at distance 5 from it
            # A word at distance 5 from codeword i has 5 bits flipped
            # There are C(17,5) = 6188 such words
            from itertools import combinations

            # Precompute all 5-bit flip masks
            flip_masks_5 = []
            for combo in combinations(range(N), 5):
                mask = 0
                for bit in combo:
                    mask |= (1 << bit)
                flip_masks_5.append(mask)

            # For efficiency, process in chunks
            for i in range(n_codes):
                if time.time() >= deadline - 2:
                    break
                # Get words at distance 5 from codeword i
                cw_i = codewords[i]
                candidates_5 = [cw_i ^ fm for fm in flip_masks_5]

                # For each candidate, check if it's at distance >= 6 from all codewords EXCEPT i
                # If so, we can swap codeword i with this candidate
                # But we also need the candidate to be at distance >= 6 from all other codewords

                # Vectorized check
                cand_arr = np.array(candidates_5, dtype=np.int64)
                cand_bits_arr = np.array([[(c >> b) & 1 for b in range(N)] for c in cand_arr], dtype=np.uint8)
                cand_w = np.sum(cand_bits_arr, axis=1)

                # Distances to all codewords except i
                other_indices = [j for j in range(n_codes) if j != i]
                if not other_indices:
                    continue
                other_bits = code_bits[other_indices]
                other_w = code_weights[other_indices]

                dots = cand_bits_arr @ other_bits.T
                dists = cand_w[:, None] + other_w[None, :] - 2 * dots
                min_dists = np.min(dists, axis=1)

                valid = np.where(min_dists >= MIN_D)[0]
                if len(valid) > 0:
                    # Perform the swap
                    new_cand = candidates_5[valid[0]]
                    # Update
                    codewords[i] = new_cand
                    code_bits[i] = bits_all[new_cand]
                    code_weights = np.sum(code_bits, axis=1)
                    swap_improved = True
                    # After a swap, try to add more words
                    addable = find_addable(code_bits, code_weights, bits_all, all_weights, MIN_D, deadline, batch_size)
                    for cand in addable:
                        if time.time() >= deadline - 2:
                            break
                        cand_bits = bits_all[cand]
                        dots = code_bits @ cand_bits
                        dists = code_weights + all_weights[cand] - 2 * dots
                        if np.min(dists) >= MIN_D:
                            codewords.append(cand)
                            code_bits = np.vstack([code_bits, cand_bits[None, :]])
                            code_weights = np.sum(code_bits, axis=1)
                            swap_improved = True

    return codewords
