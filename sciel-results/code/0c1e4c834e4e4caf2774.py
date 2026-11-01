import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    if q != 3 or n != 8 or radius != 1:
        return _general_greedy(q, n, radius, deadline - start_time)

    total_words = 3 ** n
    indices = np.arange(total_words, dtype=np.int32)
    
    # Vectorized Ball Precomputation
    powers = [3 ** j for j in range(n)]
    digits_all = np.zeros((total_words, n), dtype=np.int32)
    for j in range(n):
        digits_all[:, j] = (indices // powers[j]) % 3
        
    balls = np.empty((total_words, 17), dtype=np.int32)
    balls[:, 0] = indices
    
    col_idx = 1
    for j in range(n):
        d = digits_all[:, j]
        p = powers[j]
        for v in range(3):
            mask = (d != v)
            if mask.any():
                offset = (v - d) * p
                balls[mask, col_idx] = indices[mask] + offset[mask]
                col_idx += 1

    # Greedy Cover
    remaining = np.ones(total_words, dtype=bool)
    codewords = []
    
    while remaining.any():
        if time.time() > deadline - 10:
            break
        counts = remaining[balls].sum(axis=1)
        best_candidate = np.argmax(counts)
        if counts[best_candidate] == 0:
            break
        codewords.append(best_candidate)
        remaining[balls[best_candidate]] = False

    if remaining.any():
        while remaining.any():
            if time.time() > deadline - 5:
                break
            idx = np.where(remaining)[0][0]
            codewords.append(idx)
            remaining[balls[idx]] = False

    code_list = list(codewords)
    
    # Redundancy Removal
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        cover_count[balls[c]] += 1
        
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        new_code_list = []
        for c in code_list:
            if cover_count[balls[c]].min() >= 2:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_code_list.append(c)
        code_list = new_code_list

    def int_to_base3(x, n):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = x % 3
            x //= 3
        return digits

    return [int_to_base3(c, n) for c in code_list]

def _general_greedy(q, n, radius, time_limit):
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    def int_to_word(x):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = x % q
            x //= q
        return w

    def word_to_int(w):
        res = 0
        for d in w:
            res = res * q + d
        return res

    start = time.time()
    while covered.sum() < total:
        if time.time() - start > time_limit - 5:
            break
        best_c = -1
        best_count = -1
        for c in range(total):
            if covered[c]: continue
            w = int_to_word(c)
            count = 0
            if not covered[c]: count += 1
            for j in range(n):
                for v in range(q):
                    if v == w[j]: continue
                    w2 = w[:]
                    w2[j] = v
                    idx = word_to_int(w2)
                    if not covered[idx]: count += 1
            if count > best_count:
                best_count = count
                best_c = c
        if best_c == -1: break
        codewords.append(best_c)
        w = int_to_word(best_c)
        covered[best_c] = True
        for j in range(n):
            for v in range(q):
                if v == w[j]: continue
                w2 = w[:]
                w2[j] = v
                covered[word_to_int(w2)] = True
    return [int_to_word(c) for c in codewords]
