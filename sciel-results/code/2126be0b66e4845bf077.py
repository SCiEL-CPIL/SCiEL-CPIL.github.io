import time
import random

def construct():
    deadline = time.time() + 80
    n = 256
    best_size = 0
    best_set = None
    
    def can_add_fast(x, S_set, S_list):
        # x as 'a': b > x, c >= b in S, check x !| (b+c)
        for b in S_list:
            if b <= x:
                continue
            for k in range(3, 513 // x + 1):
                c = k * x - b
                if c < b:
                    continue
                if c > 256:
                    break
                if c in S_set:
                    return False
        # x as 'b': a < x, c >= x in S, check a !| (x+c)
        for a in S_list:
            if a >= x:
                break
            m_min = (2 * x + a - 1) // a
            m_max = (x + 256) // a
            for m in range(m_min, m_max + 1):
                c = m * a - x
                if c < x or c > 256:
                    continue
                if c in S_set:
                    return False
        # x as 'c': a < b <= x in S, check a !| (b+x)
        for a in S_list:
            if a >= x:
                break
            m_min = x // a + 2
            m_max = (2 * x) // a
            for m in range(m_min, m_max + 1):
                b = m * a - x
                if b <= a or b > x:
                    continue
                if b in S_set:
                    return False
        return True
    
    # Strategy A: Upper tail greedy from various starts
    for start in [129, 135, 140, 150, 160, 170, 180, 192]:
        if time.time() > deadline:
            break
        S_set = set()
        S_list = []
        for x in range(start, 257):
            if can_add_fast(x, S_set, S_list):
                S_set.add(x)
                S_list.append(x)
        for x in range(1, start):
            if can_add_fast(x, S_set, S_list):
                S_set.add(x)
                S_list.append(x)
        S_list.sort()
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
    
    # Strategy B: Random permutation greedy
    rng = random.Random(42)
    for trial in range(3000):
        if time.time() > deadline:
            break
        perm = list(range(1, 257))
        rng.shuffle(perm)
        S_set = set()
        S_list = []
        for x in perm:
            if can_add_fast(x, S_set, S_list):
                S_set.add(x)
                S_list.append(x)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
    
    # Strategy C: Local search
    if best_set is not None:
        S_set = set(best_set)
        S_list = sorted(S_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, 257):
                if x in S_set:
                    continue
                if can_add_fast(x, S_set, S_list):
                    S_set.add(x)
                    S_list.append(x)
                    S_list.sort()
                    improved = True
                    break
            if not improved:
                for i, rem in enumerate(S_list):
                    if time.time() > deadline:
                        break
                    S2_set = S_set - {rem}
                    S2_list = S_list[:i] + S_list[i+1:]
                    for x in range(1, 257):
                        if x in S2_set:
                            continue
                        if can_add_fast(x, S2_set, S2_list):
                            S_set = S2_set | {x}
                            S_list = S2_list + [x]
                            S_list.sort()
                            improved = True
                            break
                    if improved:
                        break
    
    if best_set is None:
        return list(range(192, 257))
    return sorted(best_set)
