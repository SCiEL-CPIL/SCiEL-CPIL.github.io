import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Pre-compute popcount table for 17-bit integers (2^17 = 131072 entries)
    popcount = [0] * (1 << 17)
    for i in range(1, 1 << 17):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def popcount_fast(x):
        return popcount[x & 0x1FFFF]

    # Pre-compute all weight-6 words as numpy array
    weight6_words = []
    for combo in combinations(range(17), 6):
        w = 0
        for bit in combo:
            w |= (1 << bit)
        weight6_words.append(w)
    weight6_arr = np.array(weight6_words, dtype=np.int64)

    def is_valid_code(words):
        for i in range(len(words)):
            wi = words[i]
            for j in range(i + 1, len(words)):
                if popcount_fast(wi ^ words[j]) < 6:
                    return False
        return True

    # Step 1: Build the [17,8,6] linear code (256 words)
    gf_poly = 0x1d

    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = ((a << 1) ^ gf_poly) & 0xFF
            else:
                a = (a << 1) & 0xFF
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
    for candidate in range(1, 256):
        if gf_pow(candidate, 17) == 1 and candidate != 1:
            alpha = candidate
            break

    if alpha is None:
        alpha = 3

    Q = [1, 2, 4, 8, 9, 13, 15, 16]

    g = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            new_g[j] ^= gf_mul(g[j], root)
            new_g[j + 1] ^= g[j]
        g = new_g

    words_17_9_5 = []
    for m in range(512):
        c = [0] * 17
        for j in range(9):
            if (m >> j) & 1:
                for k in range(9):
                    if (g[k] != 0):
                        pos = (j + k) % 17
                        c[pos] ^= 1
        val = 0
        for bit in c:
            val = (val << 1) | bit
        words_17_9_5.append(val)

    words_18_9_6 = []
    for w in words_17_9_5:
        parity = popcount_fast(w) % 2
        ext_val = w | (parity << 17)
        words_18_9_6.append(ext_val)

    words_17_8_6 = []
    for w in words_18_9_6:
        if (w >> 17) & 1 == 0:
            w_short = w & 0x1FFFF
            words_17_8_6.append(w_short)

    words_17_8_6 = list(set(words_17_8_6))
    base_code = words_17_8_6

    if not is_valid_code(base_code):
        valid_code = [base_code[0]]
        for w in base_code[1:]:
            ok = True
            for v in valid_code:
                if popcount_fast(w ^ v) < 6:
                    ok = False
                    break
            if ok:
                valid_code.append(w)
        base_code = valid_code

    current_code = list(base_code)
    best_code = list(base_code)
    best_size = len(current_code)
    if best_size < 32:
        return [0]

    # Vectorized extension search
    popcount_np = np.array(popcount, dtype=np.int8)
    
    code_np = np.array(current_code, dtype=np.int64)
    
    # Active candidates: start with all weight-6 words
    active_indices = np.arange(len(weight6_arr))
    np.random.shuffle(active_indices)
    
    # Greedy vectorized addition loop with pre-allocated array
    max_size = 400
    code_buf = np.zeros(max_size, dtype=np.int64)
    code_buf[:len(current_code)] = code_np
    code_len = len(current_code)
    
    while time.time() < deadline - 10.0 and len(active_indices) > 0:
        idx = active_indices[0]
        w_new = weight6_arr[idx]
        
        # Check validity against current code
        dists = popcount_np[w_new ^ code_buf[:code_len]]
        
        if np.min(dists) >= 6:
            # Valid
            code_buf[code_len] = w_new
            code_len += 1
            current_code.append(int(w_new))
            
            if code_len > best_size:
                best_size = code_len
                best_code = list(current_code)
            
            active_indices = active_indices[1:]
            
            # Filter remaining active candidates
            if len(active_indices) > 0:
                active_cands = weight6_arr[active_indices]
                dists_to_new = popcount_np[w_new ^ active_cands]
                mask = dists_to_new >= 6
                active_indices = active_indices[mask]
        else:
            active_indices = active_indices[1:]

    # If we still have time, try to add weight-7 or higher words via random sampling
    remaining_time = deadline - 5.0 - time.time()
    if remaining_time > 2.0:
        # Generate weight-7 and weight-8 candidates vectorized
        weight7_words = []
        for combo in combinations(range(17), 7):
            w = 0
            for bit in combo:
                w |= (1 << bit)
            weight7_words.append(w)
        weight7_arr = np.array(weight7_words, dtype=np.int64)
        
        weight8_words = []
        for combo in combinations(range(17), 8):
            w = 0
            for bit in combo:
                w |= (1 << bit)
            weight8_words.append(w)
        weight8_arr = np.array(weight8_words, dtype=np.int64)
        
        # Combine and shuffle
        extra_candidates = np.concatenate([weight7_arr, weight8_arr])
        np.random.shuffle(extra_candidates)
        
        # Filter against current code
        code_set = set(current_code)
        filtered = [w for w in extra_candidates if int(w) not in code_set]
        if filtered:
            filtered = np.array(filtered, dtype=np.int64)
            np.random.shuffle(filtered)
            
            for w_new in filtered:
                if time.time() > deadline - 2.0:
                    break
                dists = popcount_np[w_new ^ code_buf[:code_len]]
                if np.min(dists) >= 6:
                    code_buf[code_len] = w_new
                    code_len += 1
                    current_code.append(int(w_new))
                    if code_len > best_size:
                        best_size = code_len
                        best_code = list(current_code)

    # Local search: try to swap out poorly-placed words for better ones
    remaining_time = deadline - 3.0 - time.time()
    if remaining_time > 3.0 and best_size < 300:
        # Try several random restarts of the greedy from the base code
        for restart in range(5):
            if time.time() > deadline - 3.0:
                break
            trial_code = list(base_code)
            trial_buf = np.array(trial_code, dtype=np.int64)
            trial_len = len(trial_code)
            
            active = np.arange(len(weight6_arr))
            np.random.shuffle(active)
            
            while time.time() < deadline - 3.0 and len(active) > 0 and trial_len < max_size:
                idx = active[0]
                w_new = weight6_arr[idx]
                dists = popcount_np[w_new ^ trial_buf[:trial_len]]
                if np.min(dists) >= 6:
                    trial_buf[trial_len] = w_new
                    trial_len += 1
                    trial_code.append(int(w_new))
                    active = active[1:]
                    if len(active) > 0:
                        ac = weight6_arr[active]
                        mask = popcount_np[w_new ^ ac] >= 6
                        active = active[mask]
                else:
                    active = active[1:]
            
            if trial_len > best_size:
                best_size = trial_len
                best_code = list(trial_code)
    
    # Final verification
    if len(best_code) >= 32:
        if is_valid_code(best_code):
            return best_code
        else:
            final = []
            for w in best_code:
                ok = True
                for v in final:
                    if popcount_fast(w ^ v) < 6:
                        ok = False
                        break
                if ok:
                    final.append(w)
            if len(final) >= 32:
                return final
            return base_code
    else:
        return base_code if len(base_code) >= 32 else [0]
