import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    def find_generator_poly():
        for mask in range(128):
            P = (1 << 8) | mask
            R = P ^ (1 << 8)
            
            def mul_x(A):
                A_shifted = A << 1
                if A_shifted & (1 << 8):
                    A_shifted ^= R
                return A_shifted

            x = 2
            if x == 1:
                continue
            x_curr = x
            for _ in range(16):
                x_curr = mul_x(x_curr)
            if x_curr == 1:
                return P
        return None

    P = find_generator_poly()
    initial_code = set()
    
    if P is not None:
        g_vec = [0] * 17
        for i in range(8):
            g_vec[i] = (P >> i) & 1
        g_vec[8] = 1
        
        G = []
        for i in range(9):
            row = [0] * 17
            for j in range(9):
                if g_vec[j]:
                    idx = (j + i) % 17
                    row[idx] ^= 1
            G.append(row)
            
        G_np = np.array(G, dtype=np.uint8)
        
        for mask in range(512):
            word = np.zeros(17, dtype=np.uint8)
            for k in range(9):
                if (mask >> k) & 1:
                    word ^= G_np[k]
            
            parity = np.sum(word) % 2
            if parity == 0:
                val = 0
                for bit in word:
                    val = (val << 1) | bit
                initial_code.add(val)
    else:
        # Fallback: Random initialization if QR fails
        # Just pick a few random words?
        # This is a bad fallback, but prevents crash.
        # Actually, we can just start with empty and let greedy do the work, 
        # but greedy from empty is very slow to reach 256.
        # We will assume QR works.
        pass

    best_code = list(initial_code)
    
    all_candidates = [i for i in range(1 << 17) if i not in initial_code]
    
    def try_greedy(candidates, current_code_set, current_code_list, time_limit):
        for c in candidates:
            if time.time() > time_limit:
                break
            if c in current_code_set:
                continue
            
            valid = True
            for w in current_code_list:
                if (c ^ w).bit_count() < 6:
                    valid = False
                    break
            
            if valid:
                current_code_set.add(c)
                current_code_list.append(c)

    # Run greedy with natural order
    current_set = set(initial_code)
    current_list = list(initial_code)
    try_greedy(all_candidates, current_set, current_list, deadline)
    
    if len(current_set) > len(best_code):
        best_code = current_list
        
    # Run greedy with random orders
    while time.time() < deadline - 5.0:
        random.shuffle(all_candidates)
        current_set = set(initial_code)
        current_list = list(initial_code)
        try_greedy(all_candidates, current_set, current_list, deadline)
        
        if len(current_set) > len(best_code):
            best_code = current_list

    return best_code
