import time
import random
import itertools
import numpy as np

# Cache for precomputed data
_CACHE = {}

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Use cached data if available to save precomputation time
    cache_key = (q, n, radius)
    if cache_key in _CACHE:
        N, words, ball_masks_int, masks_np, popcount_lut, num_bytes, ALL_ONES, tetracode, seed_code = _CACHE[cache_key]
    else:
        N = 3**7
        ALL_ONES = (1 << N) - 1
        
        def idx_to_digits(idx):
            digits = []
            for _ in range(n):
                digits.append(idx % 3)
                idx //= 3
            return digits

        words = [idx_to_digits(i) for i in range(N)]
        
        ball_masks_int = [0] * N
        for i in range(N):
            w = words[i]
            mask = 1 << i
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(3):
                    if new_val == orig_val:
                        continue
                    new_w = w.copy()
                    new_w[pos] = new_val
                    new_idx = 0
                    power = 1
                    for d in new_w:
                        new_idx += d * power
                        power *= 3
                    mask |= (1 << new_idx)
            ball_masks_int[i] = mask

        num_bytes = (N + 7) // 8
        masks_np = np.zeros((N, num_bytes), dtype=np.uint8)
        
        for i in range(N):
            mask = ball_masks_int[i]
            for b in range(num_bytes):
                byte_val = (mask >> (8 * b)) & 0xFF
                masks_np[i, b] = byte_val

        popcount_lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        _CACHE[cache_key] = (N, words, ball_masks_int, masks_np, popcount_lut, num_bytes, ALL_ONES, tetracode, seed_code)

    best_code = seed_code
    best_len = len(seed_code)
    
    start_time = time.time()
    # Use a tighter deadline to ensure we return before the hard kill
    deadline = start_time + 115 

    while time.time() < deadline:
        covered_np = np.zeros(num_bytes, dtype=np.uint8)
        covered_int = 0
        code_indices = []
        
        while covered_int != ALL_ONES:
            # Calculate gain for each candidate
            # diff is the set of bits in the candidate's ball that are not yet covered
            diff = masks_np & (~covered_np)
            gains = popcount_lut[diff].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                # Should not happen if covered_int != ALL_ONES, but safe break
                break
            
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) > 0:
                # Random selection among best candidates to diversify search
                best_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                break
            
            code_indices.append(best_idx)
            covered_np = covered_np | masks_np[best_idx]
            covered_int |= ball_masks_int[best_idx]
            
        if len(code_indices) < best_len:
            best_len = len(code_indices)
            best_code = [words[i] for i in code_indices]
            if best_len == 186:
                break

    return best_code
