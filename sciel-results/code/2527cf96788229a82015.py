import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Step 1: Construct the 256-word linear code [17,8,6]
    poly = 0x1D

    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break

    if alpha is None:
        return [0]

    Q = [1, 2, 4, 8, 9, 13, 15, 16]

    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j + 1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g

    G = np.zeros((9, 17), dtype=np.int64)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                pos = (j + k) % 17
                G[k, pos] = 1

    words_18 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        parity = int(np.sum(word) % 2)
        word_ext = np.append(word, parity)
        words_18.append(word_ext)

    words_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_short.append(val)

    if not words_short:
        return [0]

    # Precompute popcount table
    popcount_table = np.array([i.bit_count() for i in range(2**17)], dtype=np.int8)

    # Base code
    base_code = np.array(words_short, dtype=np.int64)
    best_code = base_code.copy()
    best_size = len(base_code)

    all_words = np.arange(2**17, dtype=np.int64)

    # Step 2: Greedy extension using forbidden set approach
    # Maintain a forbidden boolean array: forbidden[w] = True if w is within distance < 6 of any codeword
    forbidden = np.zeros(2**17, dtype=bool)
    in_code = np.zeros(2**17, dtype=bool)

    # Initialize forbidden set from base code
    for w in base_code:
        diffs = all_words ^ w
        dists = popcount_table[diffs]
        forbidden[dists < 6] = True
    in_code[base_code] = True

    code_list = list(base_code)
    current_size = len(code_list)

    # Greedy: try to add words in random order
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break

        rng = np.random.default_rng(seed=restart * 12345 + 67890)
        candidates = rng.permutation(all_words)

        added_any = False
        for w in candidates:
            if time.time() > deadline - 10.0:
                break
            w = int(w)
            if in_code[w] or forbidden[w]:
                continue
            # w can be added
            code_list.append(w)
            in_code[w] = True
            current_size += 1
            added_any = True

            # Update forbidden set
            diffs = all_words ^ w
            dists = popcount_table[diffs]
            forbidden[dists < 6] = True

        if current_size > best_size:
            best_size = current_size
            best_code = np.array(code_list, dtype=np.int64)

        if not added_any:
            # No more words can be added in this order; try another order
            # But we need to reset forbidden/in_code if we want to try a different starting point
            # For now, just continue with the current code
            break

    # Step 3: Local search - try to swap words to enable more additions
    if time.time() < deadline - 5.0:
        code_list = list(best_code)
        in_code = np.zeros(2**17, dtype=bool)
        in_code[code_list] = True
        forbidden = np.zeros(2**17, dtype=bool)
        for w in code_list:
            diffs = all_words ^ w
            dists = popcount_table[diffs]
            forbidden[dists < 6] = True

        for swap_round in range(15):
            if time.time() > deadline - 5.0:
                break

            # Try to add any remaining words
            added_any = False
            rng = np.random.default_rng(seed=swap_round * 999 + 1)
            candidates = rng.permutation(all_words)
            for w in candidates:
                if time.time() > deadline - 5.0:
                    break
                w = int(w)
                if in_code[w] or forbidden[w]:
                    continue
                code_list.append(w)
                in_code[w] = True
                added_any = True
                diffs = all_words ^ w
                dists = popcount_table[diffs]
                forbidden[dists < 6] = True

            if len(code_list) > best_size:
                best_size = len(code_list)
                best_code = np.array(code_list, dtype=np.int64)

            if not added_any:
                # Try swaps: remove one word, see if we can add 2+
                rng = np.random.default_rng(seed=swap_round * 55555 + 2)
                for swap_attempt in range(20):
                    if time.time() > deadline - 5.0:
                        break

                    idx = int(rng.integers(0, len(code_list)))
                    removed = code_list[idx]
                    code_list_new = code_list[:idx] + code_list[idx+1:]
                    in_code[removed] = False

                    # Recompute forbidden set without removed word
                    forbidden_new = np.zeros(2**17, dtype=bool)
                    for w in code_list_new:
                        diffs = all_words ^ w
                        dists = popcount_table[diffs]
                        forbidden_new[dists < 6] = True

                    # Try to add words
                    added_count = 0
                    candidates2 = rng.permutation(all_words)
                    for w in candidates2:
                        if time.time() > deadline - 5.0:
                            break
                        w = int(w)
                        if in_code[w] or forbidden_new[w]:
                            continue
                        code_list_new.append(w)
                        in_code[w] = True
                        added_count += 1
                        diffs = all_words ^ w
                        dists = popcount_table[diffs]
                        forbidden_new[dists < 6] = True
                        if added_count >= 2:
                            break

                    if added_count >= 2:
                        code_list = code_list_new
                        forbidden = forbidden_new
                        if len(code_list) > best_size:
                            best_size = len(code_list)
                            best_code = np.array(code_list, dtype=np.int64)
                    else:
                        # Restore
                        in_code[removed] = True

    return list(best_code)
