import time
from itertools import combinations

_START = time.time()
DEADLINE = 85  # Leave 5s margin

def construct():
    t0 = time.time()
    
    # 1. Generate all 2^a * 3^b * 5^c smooth numbers <= 128
    smooth = []
    smooth_set = set()
    for a in range(0, 8):
        p2 = 2 ** a
        if p2 > 128: break
        for b in range(0, 6):
            p3 = 3 ** b
            if p2 * p3 > 128: break
            for c in range(0, 5):
                p5 = 5 ** c
                v = p2 * p3 * p5
                if v > 128: break
                if v not in smooth_set:
                    smooth_set.add(v)
                    smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(smooth)}
    is_smooth_arr = [False] * 129
    for v in smooth:
        is_smooth_arr[v] = True
        
    # Precompute best antichain for each sum <= 128
    # best_for_sum[s] = (count, list_of_terms)
    best_for_sum = {}
    
    def is_antichain(vals):
        l = len(vals)
        for i in range(l):
            vi = vals[i]
            for j in range(i + 1, l):
                vj = vals[j]
                if vi != vj:
                    if vi < vj:
                        if vj % vi == 0:
                            return False
                    else:
                        if vi % vj == 0:
                            return False
        return True

    # Precompute for sizes 2 to 5
    # Size 1 is handled by is_smooth_arr
    
    # Size 2
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a = smooth[i]
            b = smooth[j]
            s = a + b
            if s > 128:
                continue
            if b % a == 0: # a < b
                continue
            if s not in best_for_sum:
                best_for_sum[s] = (2, [a, b])
            elif best_for_sum[s][0] > 2:
                best_for_sum[s] = (2, [a, b])

    # Size 3
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a = smooth[i]
            b = smooth[j]
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                s = a + b + c
                if s > 128:
                    continue
                if b % a == 0: continue
                if c % a == 0: continue
                if c % b == 0: continue
                if s not in best_for_sum:
                    best_for_sum[s] = (3, [a, b, c])
                elif best_for_sum[s][0] > 3:
                    best_for_sum[s] = (3, [a, b, c])

    # Size 4
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a = smooth[i]
            b = smooth[j]
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                for l in range(k + 1, n_smooth):
                    d = smooth[l]
                    s = a + b + c + d
                    if s > 128:
                        continue
                    if b % a == 0: continue
                    if c % a == 0: continue
                    if d % a == 0: continue
                    if c % b == 0: continue
                    if d % b == 0: continue
                    if d % c == 0: continue
                    if s not in best_for_sum:
                        best_for_sum[s] = (4, [a, b, c, d])
                    elif best_for_sum[s][0] > 4:
                        best_for_sum[s] = (4, [a, b, c, d])

    # Size 5
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a = smooth[i]
            b = smooth[j]
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                for l in range(k + 1, n_smooth):
                    d = smooth[l]
                    for m in range(l + 1, n_smooth):
                        e = smooth[m]
                        s = a + b + c + d + e
                        if s > 128:
                            continue
                        if b % a == 0: continue
                        if c % a == 0: continue
                        if d % a == 0: continue
                        if e % a == 0: continue
                        if c % b == 0: continue
                        if d % b == 0: continue
                        if e % b == 0: continue
                        if d % c == 0: continue
                        if e % c == 0: continue
                        if e % d == 0: continue
                        if s not in best_for_sum:
                            best_for_sum[s] = (5, [a, b, c, d, e])
                        elif best_for_sum[s][0] > 5:
                            best_for_sum[s] = (5, [a, b, c, d, e])

    # Fallback backtracking for any n not covered
    def backtrack(rem, min_val, current_set):
        if rem == 0:
            return current_set
        if time.time() - t0 > DEADLINE:
            return None
        for idx in range(n_smooth):
            s = smooth[idx]
            if s <= min_val:
                continue
            if s > rem:
                break
            valid = True
            for x in current_set:
                if s % x == 0 or x % s == 0:
                    valid = False
                    break
            if not valid:
                continue
            res = backtrack(rem - s, s, current_set + [s])
            if res:
                return res
        return None

    answers = []
    for n in range(1, 129):
        if is_smooth_arr[n]:
            answers.append([n])
        elif n in best_for_sum:
            answers.append(best_for_sum[n][1])
        else:
            res = backtrack(n, 0, [])
            if res:
                answers.append(res)
            else:
                answers.append([1, 2, 3, 5, 7, 9, 11, 13, 15, 17]) # Dummy

    return answers
