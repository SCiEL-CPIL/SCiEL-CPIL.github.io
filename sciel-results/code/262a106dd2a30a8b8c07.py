import numpy as np
import time
from itertools import combinations

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
    all_weights = np.sum(bits_all, axis=1).astype(np.int32)

    # Precompute weights for code
    code_weights = np.sum(code_bits, axis=1).astype(np.int32)

    # Precompute flip masks for distances 5 and 6 (for swap search)
    flip_masks_5 = []
    for combo in combinations(range(N), 5):
        mask = 0
        for bit in combo:
            mask |= (1 << bit)
        flip_masks_5.append(mask)
    
    flip_masks_6 = []
    for combo in combinations(range(N), 6):
        mask = 0
        for bit in combo:
            mask |= (1 << bit)
        flip_masks_6.append(mask)

    # Step 2: Greedily add words at distance >= 6 from all existing
    # Use vectorized distance computation in batches
    batch_size = 4096

    # Find all words that can be added
    def find_addable(code_bits, code_weights, deadline):
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
    while added_any and time.time() < deadline - 10:
        added_any = False
        code_weights = np.sum(code_bits, axis=1).astype(np.int32)
        addable = find_addable(code_bits, code_weights, deadline - 10)
        if addable:
            for cand in addable:
                if time.time() >= deadline - 10:
                    break
                cand_bits = bits_all[cand]
                dots = code_bits @ cand_bits
                dists = code_weights + all_weights[cand] - 2 * dots
                if np.min(dists) >= MIN_D:
                    codewords.append(cand)
                    code_bits = np.vstack([code_bits, cand_bits[None, :]])
                    code_weights = np.sum(code_bits, axis=1).astype(np.int32)
                    added_any = True

    # Step 3: Local search - try swapping codewords to allow more additions
    # For each codeword, try replacing it with a word at distance >= 6 from all others
    # This can open up space for additional words

    if time.time() < deadline - 5:
        swap_improved = True
        # Track which codewords we've already tried swapping to avoid redundant work
        swap_attempted = set()
        
        while swap_improved and time.time() < deadline - 3:
            swap_improved = False
            n_codes = len(codewords)
            code_weights = np.sum(code_bits, axis=1).astype(np.int32)

            # Try both distance-5 and distance-6 neighbors for swaps
            # Distance-5: word is at distance 5 from codeword i (so we "move" it)
            # Distance-6: word is at distance 6 from codeword i (alternative position)
            
            for i in range(n_codes):
                if time.time() >= deadline - 3:
                    break
                if i in swap_attempted:
                    continue
                    
                cw_i = codewords[i]
                
                # Try distance-5 flips first (more likely to find valid swaps)
                for fm in flip_masks_5:
                    if time.time() >= deadline - 3:
                        break
                    cand = cw_i ^ fm
                    # Check if cand is at distance >= 6 from all codewords except i
                    cand_bits = bits_all[cand]
                    dots = code_bits @ cand_bits
                    dists = code_weights + all_weights[cand] - 2 * dots
                    # Set distance to self to infinity (we're replacing codeword i)
                    dists[i] = 999
                    if np.min(dists) >= MIN_D:
                        # Perform the swap
                        codewords[i] = cand
                        code_bits[i] = cand_bits
                        code_weights = np.sum(code_bits, axis=1).astype(np.int32)
                        swap_improved = True
                        swap_attempted.add(i)
                        
                        # After a swap, try to add more words
                        addable = find_addable(code_bits, code_weights, deadline - 3)
                        for add_cand in addable:
                            if time.time() >= deadline - 3:
                                break
                            add_cand_bits = bits_all[add_cand]
                            dots2 = code_bits @ add_cand_bits
                            dists2 = code_weights + all_weights[add_cand] - 2 * dots2
                            if np.min(dists2) >= MIN_D:
                                codewords.append(add_cand)
                                code_bits = np.vstack([code_bits, add_cand_bits[None, :]])
                                code_weights = np.sum(code_bits, axis=1).astype(np.int32)
                                swap_improved = True
                        break  # Only do one swap per codeword per pass
                
                # If no distance-5 swap worked, try distance-6 flips
                if not swap_improved or i not in swap_attempted:
                    for fm in flip_masks_6:
                        if time.time() >= deadline - 3:
                            break
                        cand = cw_i ^ fm
                        cand_bits = bits_all[cand]
                        dots = code_bits @ cand_bits
                        dists = code_weights + all_weights[cand] - 2 * dots
                        dists[i] = 999
                        if np.min(dists) >= MIN_D:
                            codewords[i] = cand
                            code_bits[i] = cand_bits
                            code_weights = np.sum(code_bits, axis=1).astype(np.int32)
                            swap_improved = True
                            swap_attempted.add(i)
                            
                            # After a swap, try to add more words
                            addable = find_addable(code_bits, code_weights, deadline - 3)
                            for add_cand in addable:
                                if time.time() >= deadline - 3:
                                    break
                                add_cand_bits = bits_all[add_cand]
                                dots2 = code_bits @ add_cand_bits
                                dists2 = code_weights + all_weights[add_cand] - 2 * dots2
                                if np.min(dists2) >= MIN_D:
                                    codewords.append(add_cand)
                                    code_bits = np.vstack([code_bits, add_cand_bits[None, :]])
                                    code_weights = np.sum(code_bits, axis=1).astype(np.int32)
                                    swap_improved = True
                            break
    
    # Final pass: try to add any remaining words
    if time.time() < deadline - 1:
        code_weights = np.sum(code_bits, axis=1).astype(np.int32)
        addable = find_addable(code_bits, code_weights, deadline - 1)
        for cand in addable:
            if time.time() >= deadline - 1:
                break
            cand_bits = bits_all[cand]
            dots = code_bits @ cand_bits
            dists = code_weights + all_weights[cand] - 2 * dots
            if np.min(dists) >= MIN_D:
                codewords.append(cand)
                code_bits = np.vstack([code_bits, cand_bits[None, :]])
                code_weights = np.sum(code_bits, axis=1).astype(np.int32)

    return codewords
