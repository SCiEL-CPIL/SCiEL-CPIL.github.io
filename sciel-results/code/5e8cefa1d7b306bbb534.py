"""Ternary covering code K_3(7,1) via greedy set-cover."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0

    total = q ** n
    all_words = np.zeros((total, n), dtype=np.int8)
    for i in range(total):
        v = i
        for j in range(n):
            all_words[i, j] = v % q
            v //= q

    # Precompute cover_by_codeword as a list of arrays
    cover_by_codeword = []
    for c_idx in range(total):
        c = all_words[c_idx]
        covered = [c_idx]
        for pos in range(n):
            for alt in range(q):
                if alt == c[pos]:
                    continue
                c2 = c.copy()
                c2[pos] = alt
                w_idx = 0
                for j in range(n):
                    w_idx = w_idx * q + c2[j]
                covered.append(w_idx)
        cover_by_codeword.append(np.array(covered, dtype=np.int32))

    # Build reverse: for each word, which codewords cover it
    # word_coverers[w] = list of c_idx
    word_coverers = [[] for _ in range(total)]
    for c_idx in range(total):
        for w_idx in cover_by_codeword[c_idx]:
            word_coverers[w_idx].append(c_idx)
    word_coverers = [np.array(x, dtype=np.int32) for x in word_coverers]

    cover_count = np.zeros(total, dtype=np.int32)
    gain = np.zeros(total, dtype=np.int32)
    for c_idx in range(total):
        gain[c_idx] = len(cover_by_codeword[c_idx])

    chosen = []
    while time.time() < deadline:
        best_c = int(np.argmax(gain))
        if gain[best_c] == 0:
            break
        chosen.append(best_c)
        # For each word covered by best_c that was previously uncovered,
        # decrement gain for all codewords that cover that word
        for w_idx in cover_by_codeword[best_c]:
            if cover_count[w_idx] == 0:
                cover_count[w_idx] = 1
                for c2 in word_coverers[w_idx]:
                    gain[c2] -= 1

    result = [all_words[c].tolist() for c in chosen]
    return result
# EVOLVE-BLOCK-START
