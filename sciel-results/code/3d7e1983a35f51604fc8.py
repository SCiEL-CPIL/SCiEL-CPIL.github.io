import time

_START = time.time()
DEADLINE = 85

def construct():
    t0 = time.time()
    
    # Generate all 2^a * 3^b * 5^c smooth numbers <= 128
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
                smooth_set.add(v)
    smooth = sorted(smooth_set)
    n_smooth = len(smooth)
    
    # Index mapping
    val_to_idx = {v: i for i, v in enumerate(smooth)}
    
    # Precompute divisibility relations
    # For each i, divisors_of[i] = set of j where smooth[j] divides smooth[i]
    divisors_of = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        vi = smooth[i]
        for j in range(i):
            if vi % smooth[j] == 0:
                divisors_of[i].add(j)
    
    min_terms = [100] * 129
    reps = [None] * 129
    
    def record(target, terms):
        if terms and target > 0 and target <= 128:
            if len(terms) < min_terms[target]:
                min_terms[target] = len(terms)
                reps[target] = list(terms)
    
    # Phase 1: Single terms
    for v in smooth:
        if v <= 128:
            record(v, [v])
    
    # Phase 2: Two-term antichains
    for i in range(n_smooth):
        vi = smooth[i]
        for j in range(i + 1, n_smooth):
            vj = smooth[j]
            s = vi + vj
            if s > 128:
                break
            if vj % vi != 0:
                record(s, [vi, vj])
    
    # Phase 3: Three-term antichains
    for i in range(n_smooth):
        vi = smooth[i]
        for j in range(i + 1, n_smooth):
            vj = smooth[j]
            if vj % vi == 0:
                continue
            for k in range(j + 1, n_smooth):
                vk = smooth[k]
                s = vi + vj + vk
                if s > 128:
                    break
                if vk % vi == 0 or vk % vj == 0:
                    continue
                record(s, [vi, vj, vk])
    
    # Phase 4: Four-term antichains
    for i in range(n_smooth):
        vi = smooth[i]
        for j in range(i + 1, n_smooth):
            vj = smooth[j]
            if vj % vi == 0:
                continue
            for k in range(j + 1, n_smooth):
                vk = smooth[k]
                if vk % vi == 0 or vk % vj == 0:
                    continue
                for l in range(k + 1, n_smooth):
                    vl = smooth[l]
                    s = vi + vj + vk + vl
                    if s > 128:
                        break
                    if vl % vi == 0 or vl % vj == 0 or vl % vk == 0:
                        continue
                    record(s, [vi, vj, vk, vl])
    
    # Phase 5: For any targets still not covered, do a targeted DFS
    missing = [n for n in range(1, 129) if reps[n] is None]
    
    if missing:
        def dfs_target(target, start_idx, current_sum, current_list, max_terms):
            if current_sum > target:
                return False
            if len(current_list) >= max_terms:
                return False
            if current_sum == target:
                if current_list:
                    record(target, current_list)
                    return True
                return False
            
            if time.time() - t0 > DEADLINE - 2:
                return False
            
            for i in range(start_idx, n_smooth):
                new_val = smooth[i]
                new_sum = current_sum + new_val
                if new_sum > target:
                    break
                valid = True
                for val in current_list:
                    if new_val % val == 0:
                        valid = False
                        break
                if not valid:
                    continue
                current_list.append(new_val)
                if dfs_target(target, i + 1, new_sum, current_list, max_terms):
                    current_list.pop()
                    return True
                current_list.pop()
            return False
        
        for target in missing:
            found = False
            for max_t in range(5, 31):
                if time.time() - t0 > DEADLINE - 2:
                    break
                if dfs_target(target, 0, 0, [], max_t):
                    found = True
                    break
            if not found and reps[target] is None:
                if target in smooth_set:
                    record(target, [target])
    
    # Phase 6: Refinement - try to find shorter representations for targets
    # that currently have longer ones, using a more exhaustive search
    # Focus on targets with min_terms[target] > 4, try to find 4 or fewer
    for target in range(1, 129):
        if reps[target] is None:
            continue
        current_len = min_terms[target]
        if current_len <= 4:
            continue
        # Try to find a representation with fewer terms
        # Try 4 terms first, then 3, then 2, then 1
        for max_t in range(1, min(current_len, 5)):
            if time.time() - t0 > DEADLINE - 3:
                break
            if max_t == 1:
                if target in smooth_set:
                    record(target, [target])
                    break
            elif max_t == 2:
                found = False
                for i in range(n_smooth):
                    vi = smooth[i]
                    if vi >= target:
                        break
                    vj = target - vi
                    if vj <= vi:
                        continue
                    if vj in smooth_set and vj % vi != 0:
                        record(target, [vi, vj])
                        found = True
                        break
                if found:
                    break
            elif max_t == 3:
                found = False
                for i in range(n_smooth):
                    vi = smooth[i]
                    if vi >= target:
                        break
                    for j in range(i + 1, n_smooth):
                        vj = smooth[j]
                        if vj % vi == 0:
                            continue
                        vk = target - vi - vj
                        if vk <= vj:
                            continue
                        if vk in smooth_set and vk % vi != 0 and vk % vj != 0:
                            record(target, [vi, vj, vk])
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            elif max_t == 4:
                found = False
                for i in range(n_smooth):
                    vi = smooth[i]
                    if vi >= target:
                        break
                    for j in range(i + 1, n_smooth):
                        vj = smooth[j]
                        if vj % vi == 0:
                            continue
                        for k in range(j + 1, n_smooth):
                            vk = smooth[k]
                            if vk % vi == 0 or vk % vj == 0:
                                continue
                            vl = target - vi - vj - vk
                            if vl <= vk:
                                continue
                            if vl in smooth_set and vl % vi != 0 and vl % vj != 0 and vl % vk != 0:
                                record(target, [vi, vj, vk, vl])
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
    
    # Build answers
    answers = []
    for n in range(1, 129):
        if reps[n] is not None:
            answers.append(reps[n])
        else:
            if n in smooth_set:
                answers.append([n])
            else:
                def slow_find(target):
                    def dfs2(start_idx, cur_sum, cur_list):
                        if cur_sum > target:
                            return None
                        if len(cur_list) >= 30:
                            return None
                        if cur_sum == target:
                            return list(cur_list) if cur_list else None
                        if time.time() - t0 > DEADLINE - 1:
                            return None
                        for i in range(start_idx, n_smooth):
                            nv = smooth[i]
                            ns = cur_sum + nv
                            if ns > target:
                                break
                            ok = True
                            for v in cur_list:
                                if nv % v == 0:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            cur_list.append(nv)
                            res = dfs2(i + 1, ns, cur_list)
                            cur_list.pop()
                            if res:
                                return res
                        return None
                    return dfs2(0, 0, [])
                res = slow_find(n)
                if res:
                    answers.append(res)
                else:
                    answers.append([1])
    
    return answers
