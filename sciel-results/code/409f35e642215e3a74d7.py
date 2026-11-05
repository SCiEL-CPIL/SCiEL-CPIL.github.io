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

    # Step 2: Greedily add words at distance >= 6 from all existing
    # Use vectorized distance computation in batches
    batch_size = 5000

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
            # Add words one by one, rechecking validity after each addition
            # Sort by some heuristic: prefer words that are "central" or have good spread
            # For now, just add them in order, but verify each against the growing code
            for cand in addable:
                if time.time() >= deadline:
                    break
                cand_bits = bits_all[cand][None, :]
                # Check distance to all current codewords
                dots = cand_bits @ code_bits.T
                cand_w = all_weights[cand]
                dists = cand_w + code_weights - 2 * dots
                if np.min(dists) >= MIN_D:
                    codewords.append(cand)
                    code_bits = np.vstack([code_bits, cand_bits])
                    code_weights = np.sum(code_bits, axis=1)
                    added_any = True

    return codewords
