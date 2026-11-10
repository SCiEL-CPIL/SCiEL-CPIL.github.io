import time

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin for 90s kill
    
    target_num = 5
    target_den = 31
    
    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    def check_time():
        return time.time() > deadline

    # Search for k-term solutions
    # We will search k=3, 4, 5, 6
    # To minimize max denominator, we should search smaller max denominators first?
    # Or just do a depth-first search with pruning.
    
    # 3-term search
    # 1/a + 1/b + 1/c = 5/31
    for a in range(7, 19):
        if check_time(): break
        if a >= best_max: break
        rem_num = 5 * a - 31
        rem_den = 31 * a
        if rem_num <= 0: continue
        
        min_b = max(a + 1, rem_den // rem_num + 1)
        max_b = min((2 * rem_den) // rem_num, best_max - 1)
        for b in range(min_b, max_b + 1):
            if check_time(): break
            rem2_num = rem_num * b - rem_den
            rem2_den = rem_den * b
            if rem2_num <= 0: continue
            
            if rem2_den % rem2_num == 0:
                c = rem2_den // rem2_num
                if c > b and c < best_max:
                    update_best([a, b, c])

    # 4-term search
    for a in range(7, 25):
        if check_time(): break
        if a >= best_max: break
        rem_num = 5 * a - 31
        rem_den = 31 * a
        if rem_num <= 0: continue
        
        min_b = max(a + 1, rem_den // rem_num + 1)
        max_b = min((3 * rem_den) // rem_num, best_max - 1)
        for b in range(min_b, max_b + 1):
            if check_time(): break
            rem2_num = rem_num * b - rem_den
            rem2_den = rem_den * b
            if rem2_num <= 0: continue
            
            min_c = max(b + 1, rem2_den // rem2_num + 1)
            max_c = min((2 * rem2_den) // rem2_num, best_max - 1)
            for c in range(min_c, max_c + 1):
                if check_time(): break
                rem3_num = rem2_num * c - rem2_den
                rem3_den = rem2_den * c
                if rem3_num <= 0: continue
                
                if rem3_den % rem3_num == 0:
                    d = rem3_den // rem3_num
                    if d > c and d < best_max:
                        update_best([a, b, c, d])

    # 5-term search
    for a in range(7, 32):
        if check_time(): break
        if a >= best_max: break
        rem_num = 5 * a - 31
        rem_den = 31 * a
        if rem_num <= 0: continue
        
        min_b = max(a + 1, rem_den // rem_num + 1)
        max_b = min((4 * rem_den) // rem_num, best_max - 1)
        for b in range(min_b, max_b + 1):
            if check_time(): break
            rem2_num = rem_num * b - rem_den
            rem2_den = rem_den * b
            if rem2_num <= 0: continue
            
            min_c = max(b + 1, rem2_den // rem2_num + 1)
            max_c = min((3 * rem2_den) // rem2_num, best_max - 1)
            for c in range(min_c, max_c + 1):
                if check_time(): break
                rem3_num = rem2_num * c - rem2_den
                rem3_den = rem2_den * c
                if rem3_num <= 0: continue
                
                min_d = max(c + 1, rem3_den // rem3_num + 1)
                max_d = min((2 * rem3_den) // rem3_num, best_max - 1)
                for d in range(min_d, max_d + 1):
                    if check_time(): break
                    rem4_num = rem3_num * d - rem3_den
                    rem4_den = rem3_den * d
                    if rem4_num <= 0: continue
                    
                    if rem4_den % rem4_num == 0:
                        e = rem4_den // rem4_num
                        if e > d and e < best_max:
                            update_best([a, b, c, d, e])

    if best_result is not None:
        return best_result

    # Fallback if no solution found (should not happen for 5/31)
    return [7, 8, 9, 10000000]

# EVOLVE-BLOCK-END
