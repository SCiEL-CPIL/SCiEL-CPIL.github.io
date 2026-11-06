import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Generate 2-3-5 smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    smooth.sort()
    n_smooth = len(smooth)
    
    # Precompute compatibility
    compat = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            a, b = smooth[i], smooth[j]
            if a % b == 0 or b % a == 0:
                compat[i][j] = False
                compat[j][i] = False
    
    answers = []
    
    for target in range(1, 129):
        if time.time() > deadline:
            # Emergency fallback
            if target in smooth_set:
                answers.append([target])
            else:
                # Quick search
                ans = None
                for cnt in range(2, 11):
                    cands = [x for x in smooth if x <= target]
                    if len(cands) < cnt:
                        break
                    from itertools import combinations
                    for terms in combinations(cands, cnt):
                        if sum(terms) == target:
                            idxs = [smooth.index(t) for t in terms]
                            ok = True
                            for ii in range(len(idxs)):
                                for jj in range(ii+1, len(idxs)):
                                    if not compat[idxs[ii]][idxs[jj]]:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if ok:
                                ans = list(terms)
                                break
                    if ans:
                        break
                answers.append(ans)
            continue
        
        answer = None
        candidates = [x for x in smooth if x <= target]
        
        for count in range(1, 15):
            if time.time() > deadline:
                break
            if len(candidates) < count:
                break
            min_sum = sum(candidates[:count])
            max_sum = sum(candidates[-count:])
            if min_sum > target or max_sum < target:
                continue
            
            if count == 1:
                if target in smooth_set:
                    answer = [target]
                    break
                continue
            
            elif count == 2:
                found = False
                for a in candidates:
                    if a >= target - a:
                        break
                    b = target - a
                    if b in smooth_set and b > a:
                        i = smooth.index(a)
                        j = smooth.index(b)
                        if compat[i][j]:
                            answer = [a, b]
                            found = True
                            break
                if found:
                    break
            else:
                # Backtracking search
                from itertools import combinations
                found = False
                for terms in combinations(candidates, count):
                    if sum(terms) == target:
                        idxs = [smooth.index(t) for t in terms]
                        ok = True
                        for ii in range(len(idxs)):
                            for jj in range(ii+1, len(idxs)):
                                if not compat[idxs[ii]][idxs[jj]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            answer = list(terms)
                            found = True
                            break
                if found:
                    break
        
        if answer is None:
            for count in range(15, 31):
                if time.time() > deadline:
                    break
                if len(candidates) < count:
                    break
                min_sum = sum(candidates[:count])
                max_sum = sum(candidates[-count:])
                if min_sum > target or max_sum < target:
                    continue
                from itertools import combinations
                found = False
                for terms in combinations(candidates, count):
                    if sum(terms) == target:
                        idxs = [smooth.index(t) for t in terms]
                        ok = True
                        for ii in range(len(idxs)):
                            for jj in range(ii+1, len(idxs)):
                                if not compat[idxs[ii]][idxs[jj]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            answer = list(terms)
                            found = True
                            break
                if found:
                    break
        
        if answer is None:
            answer = [target] if target in smooth_set else None
        
        answers.append(answer)
    
    return answers
