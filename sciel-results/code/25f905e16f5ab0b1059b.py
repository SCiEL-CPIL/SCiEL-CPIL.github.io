import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # 1. Initialization
    S = list(range(129, 257))
    S_set = set(S)
    
    def is_valid(S_list, S_set):
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                min_k = (2 * b + a - 1) // a
                max_k = (256 + b) // a
                for k in range(min_k, max_k + 1):
                    c = k * a - b
                    if c >= b and c in S_set:
                        return False
        return True

    def count_conflicts(S_list, S_set):
        conflicts = {x: 0 for x in S_list}
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                min_k = (2 * b + a - 1) // a
                max_k = (256 + b) // a
                for k in range(min_k, max_k + 1):
                    c = k * a - b
                    if c >= b and c in S_set:
                        conflicts[a] += 1
                        conflicts[b] += 1
                        if c in conflicts:
                            conflicts[c] += 1
        return conflicts

    # 2. Initial Pruning
    for _ in range(200):
        if is_valid(S, S_set):
            break
        conflicts = count_conflicts(S, S_set)
        if not conflicts:
            break
        max_conf = -1
        max_elem = None
        for elem, cnt in conflicts.items():
            if cnt > max_conf:
                max_conf = cnt
                max_elem = elem
        if max_elem is None:
            break
        S.remove(max_elem)
        S_set.remove(max_elem)
        
    # 3. Local Search
    def can_add(x, S_list, S_set):
        # Check x as a
        for b in S_list:
            if b <= x: continue
            min_k = (2 * b + x - 1) // x
            max_k = (256 + b) // x
            for k in range(min_k, max_k + 1):
                c = k * x - b
                if c >= b and c in S_set:
                    return False
        # Check x as b
        for a in S_list:
            if a >= x: continue
            min_k = (2 * x + a - 1) // a
            max_k = (256 + x) // a
            for k in range(min_k, max_k + 1):
                c = k * a - x
                if c >= x and c in S_set:
                    return False
        # Check x as c
        for b in S_list:
            if b > x: continue
            for a in S_list:
                if a >= b: continue
                if (b + x) % a == 0:
                    return False
        return True

    def can_swap_out_in(y, x, S_list, S_set):
        # Check if x is compatible with S \ {y}
        # x as a
        for b in S_list:
            if b == y: continue
            if b <= x: continue
            min_k = (2 * b + x - 1) // x
            max_k = (256 + b) // x
            for k in range(min_k, max_k + 1):
                c = k * x - b
                if c >= b and (c in S_set and c != y):
                    return False
        # x as b
        for a in S_list:
            if a == y: continue
            if a >= x: continue
            min_k = (2 * x + a - 1) // a
            max_k = (256 + x) // a
            for k in range(min_k, max_k + 1):
                c = k * a - x
                if c >= x and (c in S_set and c != y):
                    return False
        # x as c
        for b in S_list:
            if b == y: continue
            if b > x: continue
            for a in S_list:
                if a == y: continue
                if a >= b: continue
                if (b + x) % a == 0:
                    return False
        return True

    while time.time() < deadline:
        # Phase 1: Additions
        added_any = False
        candidates = [x for x in range(1, 257) if x not in S_set]
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S, S_set):
                S.append(x)
                S_set.add(x)
                added_any = True
        S.sort()
        
        if added_any:
            continue

        # Phase 2: Swaps (Limited)
        swap_done = False
        candidates_out = [x for x in range(1, 257) if x not in S_set]
        random.shuffle(candidates_out)
        
        # Try a limited number of swaps to avoid O(N^3) blowup
        swap_attempts = 0
        max_swaps_per_iter = 50
        while swap_attempts < max_swaps_per_iter:
            if time.time() > deadline:
                break
            if not S or not candidates_out:
                break
            y = random.choice(S)
            x = random.choice(candidates_out)
            if can_swap_out_in(y, x, S, S_set):
                S.remove(y)
                S_set.remove(y)
                S.append(x)
                S_set.add(x)
                S.sort()
                swap_done = True
                break
            swap_attempts += 1
            
        if swap_done:
            continue

        # Phase 3: Remove one element to break deadlock
        if S:
            # Remove the smallest element, as it is the most restrictive
            S.sort()
            y = S.pop(0)
            S_set.remove(y)
        else:
            break

    return S
