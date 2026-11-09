import time

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):
        val_a = 2 ** a
        for b in range(0, 5):
            val_ab = val_a * (3 ** b)
            if val_ab > 128:
                break
            for c in range(0, 4):
                val_abc = val_ab * (5 ** c)
                if val_abc > 128:
                    break
                smooth.add(val_abc)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # Precompute divisibility
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_list[i], smooth_list[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    divs[i][j] = True
                    divs[j][i] = True
    
    min_k_for_sum = {}
    
    # k=1
    for i in range(n_smooth):
        s = smooth_list[i]
        if s <= 128 and s not in min_k_for_sum:
            min_k_for_sum[s] = (1, [s])
    
    # k=2
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if divs[i][j]:
                continue
            s = smooth_list[i] + smooth_list[j]
            if s <= 128 and s not in min_k_for_sum:
                min_k_for_sum[s] = (2, [smooth_list[i], smooth_list[j]])
    
    # k=3
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if divs[i][j]:
                continue
            for k in range(j+1, n_smooth):
                if divs[i][k] or divs[j][k]:
                    continue
                s = smooth_list[i] + smooth_list[j] + smooth_list[k]
                if s <= 128 and s not in min_k_for_sum:
                    min_k_for_sum[s] = (3, [smooth_list[i], smooth_list[j], smooth_list[k]])
    
    # k=4
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if divs[i][j]:
                continue
            for k in range(j+1, n_smooth):
                if divs[i][k] or divs[j][k]:
                    continue
                for l in range(k+1, n_smooth):
                    if divs[i][l] or divs[j][l] or divs[k][l]:
                        continue
                    s = smooth_list[i] + smooth_list[j] + smooth_list[k] + smooth_list[l]
                    if s <= 128 and s not in min_k_for_sum:
                        min_k_for_sum[s] = (4, [smooth_list[i], smooth_list[j], smooth_list[k], smooth_list[l]])
    
    # k=5
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if divs[i][j]:
                continue
            for k in range(j+1, n_smooth):
                if divs[i][k] or divs[j][k]:
                    continue
                for l in range(k+1, n_smooth):
                    if divs[i][l] or divs[j][l] or divs[k][l]:
                        continue
                    for m in range(l+1, n_smooth):
                        if divs[i][m] or divs[j][m] or divs[k][m] or divs[l][m]:
                            continue
                        s = smooth_list[i] + smooth_list[j] + smooth_list[k] + smooth_list[l] + smooth_list[m]
                        if s <= 128 and s not in min_k_for_sum:
                            min_k_for_sum[s] = (5, [smooth_list[i], smooth_list[j], smooth_list[k], smooth_list[l], smooth_list[m]])
    
    # k=6
    if time.time() - start_time < DEADLINE - 5:
        for i in range(n_smooth):
            for j in range(i+1, n_smooth):
                if divs[i][j]:
                    continue
                for k in range(j+1, n_smooth):
                    if divs[i][k] or divs[j][k]:
                        continue
                    for l in range(k+1, n_smooth):
                        if divs[i][l] or divs[j][l] or divs[k][l]:
                            continue
                        for m in range(l+1, n_smooth):
                            if divs[i][m] or divs[j][m] or divs[k][m] or divs[l][m]:
                                continue
                            for n in range(m+1, n_smooth):
                                if divs[i][n] or divs[j][n] or divs[k][n] or divs[l][n] or divs[m][n]:
                                    continue
                                s = smooth_list[i] + smooth_list[j] + smooth_list[k] + smooth_list[l] + smooth_list[m] + smooth_list[n]
                                if s <= 128 and s not in min_k_for_sum:
                                    min_k_for_sum[s] = (6, [smooth_list[i], smooth_list[j], smooth_list[k], smooth_list[l], smooth_list[m], smooth_list[n]])
    
    # Fallback for uncovered numbers
    def find_antichain_bt(n, k):
        cand = [idx for idx in range(n_smooth) if smooth_list[idx] <= n]
        if len(cand) < k:
            return None
        if k == 1:
            if n in val_to_idx:
                return [n]
            return None
        path = []
        def bt(start, rem_sum, rem_k):
            if rem_k == 0:
                if rem_sum == 0:
                    return [smooth_list[idx] for idx in path]
                return None
            if len(cand) - start < rem_k:
                return None
            for i in range(start, len(cand)):
                idx = cand[i]
                num = smooth_list[idx]
                if num > rem_sum:
                    continue
                ok = True
                for pi in path:
                    if divs[pi][idx]:
                        ok = False
                        break
                if ok:
                    path.append(idx)
                    r = bt(i+1, rem_sum-num, rem_k-1)
                    if r:
                        return r
                    path.pop()
            return None
        return bt(0, n, k)
    
    answers = []
    for n in range(1, 129):
        if n in min_k_for_sum:
            answers.append(min_k_for_sum[n][1])
        else:
            found = None
            for k in range(1, 31):
                if time.time() - start_time > DEADLINE:
                    break
                r = find_antichain_bt(n, k)
                if r:
                    found = r
                    break
            if found:
                answers.append(found)
            else:
                answers.append([n])
    
    return answers
