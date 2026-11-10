import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
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

    # 3-term search
    # 1/a + 1/b + 1/c = 5/31
    # a in [7, 18]
    for a in range(7, 19):
        if time.time() > deadline:
            break
        # rem = 5/31 - 1/a = (5a - 31) / (31a)
        rem_num = 5 * a - 31
        rem_den = 31 * a
        if rem_num <= 0:
            continue
        
        # 1/b + 1/c = rem_num / rem_den
        # b > a
        # 1/b < rem => b > rem_den / rem_num
        # 2/b >= rem => b <= 2 * rem_den / rem_num
        min_b = a + 1
        max_b = int(2 * rem_den / rem_num)
        
        for b in range(min_b, max_b + 1):
            # rem2 = rem - 1/b = (rem_num*b - rem_den) / (rem_den*b)
            rem2_num = rem_num * b - rem_den
            rem2_den = rem_den * b
            if rem2_num <= 0:
                continue
            # 1/c = rem2 => c = rem2_den / rem2_num
            if rem2_den % rem2_num != 0:
                continue
            c = rem2_den // rem2_num
            if c <= b:
                continue
            update_best([a, b, c])

    # 4-term search
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem_num = 5 * a - 31
        rem_den = 31 * a
        if rem_num <= 0:
            continue
        
        min_b = a + 1
        max_b = int(3 * rem_den / rem_num)
        
        for b in range(min_b, max_b + 1):
            rem2_num = rem_num * b - rem_den
            rem2_den = rem_den * b
            if rem2_num <= 0:
                continue
            
            min_c = b + 1
            max_c = int(2 * rem2_den / rem2_num)
            
            for c in range(min_c, max_c + 1):
                rem3_num = rem2_num * c - rem2_den
                rem3_den = rem2_den * c
                if rem3_num <= 0:
                    continue
                if rem3_den % rem3_num != 0:
                    continue
                d = rem3_den // rem3_num
                if d <= c:
                    continue
                if d >= best_max:
                    continue
                update_best([a, b, c, d])

    if best_result is not None:
        return best_result

    return [7, 8, 9, 10000000]
