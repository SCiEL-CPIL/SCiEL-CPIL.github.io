import time
import random

def _validate(colors):
    n = len(colors)
    for i in range(n):
        for d in range(1, (n - 1 - i) // 3 + 1):
            j = i + d
            k = i + 2 * d
            l = i + 3 * d
            if l >= n:
                break
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _get_aps_and_map(n):
    aps = []
    pos_to_aps = {i: [] for i in range(n)}
    for i in range(n):
        for d in range(1, (n - 1 - i) // 3 + 1):
            j = i + d
            k = i + 2 * d
            l = i + 3 * d
            if l < n:
                idx = len(aps)
                aps.append((i, j, k, l))
                pos_to_aps[i].append(idx)
                pos_to_aps[j].append(idx)
                pos_to_aps[k].append(idx)
                pos_to_aps[l].append(idx)
    return aps, pos_to_aps

def _local_search_csp(n, K, aps, pos_to_aps, time_limit):
    start_time = time.time()
    
    while time.time() < start_time + time_limit:
        colors = [random.randint(0, K - 1) for _ in range(n)]
        
        is_violated = [False] * len(aps)
        violations_count = 0
        for idx, ap in enumerate(aps):
            i, j, k, l = ap
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                is_violated[idx] = True
                violations_count += 1
                
        if violations_count == 0:
            return colors
            
        no_improve_count = 0
        while time.time() < start_time + time_limit:
            if violations_count == 0:
                return colors
                
            # Find violated APs
            violated_indices = [idx for idx, v in enumerate(is_violated) if v]
            if not violated_indices:
                return colors
            ap_idx = random.choice(violated_indices)
            i, j, k, l = aps[ap_idx]
            
            pos_idx = random.choice([i, j, k, l])
            old_color = colors[pos_idx]
            
            new_color = random.randint(0, K - 1)
            if new_color == old_color:
                if K > 1:
                    new_color = (new_color + 1) % K
                else:
                    continue
                    
            colors[pos_idx] = new_color
            
            old_count = violations_count
            for idx in pos_to_aps[pos_idx]:
                old_v = is_violated[idx]
                ii, jj, kk, ll = aps[idx]
                new_v = len({colors[ii], colors[jj], colors[kk], colors[ll]}) < 3
                is_violated[idx] = new_v
                if old_v and not new_v:
                    violations_count -= 1
                elif not old_v and new_v:
                    violations_count += 1
                    
            if violations_count < old_count:
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count > 1000:
                break
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128
    aps, pos_to_aps = _get_aps_and_map(n)
    
    for K in range(3, 10):
        time_remaining = deadline - time.time()
        if time_remaining < 1.0:
            break
            
        if K == 3:
            t_limit = min(30.0, time_remaining)
        elif K == 4:
            t_limit = min(30.0, time_remaining)
        else:
            t_limit = min(10.0, time_remaining)
            
        if t_limit <= 0:
            continue
            
        result = _local_search_csp(n, K, aps, pos_to_aps, t_limit)
        
        if result is not None:
            return result
            
    # Fallback
    colors = [0] * n
    for i in range(n):
        pos = i + 1
        for c in range(128):
            ok = True
            max_d = (pos - 1) // 3
            for d in range(1, max_d + 1):
                i1 = pos - 3 * d - 1
                i2 = pos - 2 * d - 1
                i3 = pos - d - 1
                if len({colors[i1], colors[i2], colors[i3], c}) < 3:
                    ok = False
                    break
            if not ok:
                continue
            max_d2 = (n - pos) // 3
            for d in range(1, max_d2 + 1):
                i1 = pos + d - 1
                i2 = pos + 2 * d - 1
                i3 = pos + 3 * d - 1
                if len({c, colors[i1], colors[i2], colors[i3]}) < 3:
                    ok = False
                    break
            if not ok:
                continue
            max_d3 = (pos - 1) // 3
            for d in range(1, max_d3 + 1):
                i1 = pos - 3 * d - 1
                i2 = pos - 2 * d - 1
                i4 = pos + d - 1
                if pos + d - 1 < n:
                    if len({colors[i1], colors[i2], c, colors[i4]}) < 3:
                        ok = False
                        break
            if ok:
                colors[i] = c
                break
                
    return colors
