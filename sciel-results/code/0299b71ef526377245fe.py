import time
import random
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Step 1: Build superpermutation for n-1 using standard construction
    def build_standard(m):
        word = [1]
        for k in range(2, m + 1):
            perms = []
            seen = set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    word_np1 = build_standard(n - 1)
    
    # Step 2: Extract the sequence of (n-1)! permutations from word_np1
    perms = []
    seen = set()
    for i in range(len(word_np1) - (n - 1) + 1):
        window = tuple(word_np1[i:i + (n - 1)])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    num_perms = len(perms)  # Should be (n-1)!
    
    # Step 3: Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    max_k = n - 2
    overlap = np.zeros((num_perms, num_perms), dtype=np.int32)
    
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap[i][j] = 0
                    break
                if pi[-k:] == pj[:k]:
                    overlap[i][j] = k
                    break
    
    # Step 4: Local search to optimize ordering
    # Initial order: 0, 1, 2, ..., num_perms-1
    def calc_length(order):
        # First block: 2*(n-1) + 1
        total = 2 * (n - 1) + 1
        for i in range(len(order) - 1):
            u = order[i]
            v = order[i + 1]
            total += (2 * (n - 1) + 1) - overlap[u, v]
        return total
    
    best_order = list(range(num_perms))
    best_len = calc_length(best_order)
    
    # Adjacent swap local search
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(num_perms - 1):
            if time.time() > deadline - 10.0:
                break
            # Try swap i, i+1
            order = best_order[:]
            order[i], order[i + 1] = order[i + 1], order[i]
            new_len = calc_length(order)
            if new_len < best_len:
                best_len = new_len
                best_order = order
                improved = True
            # Try swap i-1, i (if i > 0)
            if i > 0:
                order2 = best_order[:]
                order2[i - 1], order2[i] = order2[i], order2[i - 1]
                new_len2 = calc_length(order2)
                if new_len2 < best_len:
                    best_len = new_len2
                    best_order = order2
                    improved = True
    
    # 2-opt local search: try reversing a segment
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(num_perms - 1):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 2, num_perms):
                if time.time() > deadline - 5.0:
                    break
                order = best_order[:]
                order[i:j+1] = order[i:j+1][::-1]
                new_len = calc_length(order)
                if new_len < best_len:
                    best_len = new_len
                    best_order = order
                    improved = True
                    break
    
    # Step 5: Construct final word using optimized ordering
    final_word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_ov = min(len(final_word), len(block))
        ov = 0
        for o in range(max_ov, -1, -1):
            if final_word[len(final_word) - o:] == block[:o]:
                ov = o
                break
        final_word += block[ov:]
    
    return "".join(map(str, final_word))
