import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # safety margin
    
    n = 512
    
    # Precompute all 4-APs
    ap_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # element -> list of AP indices containing it
    element_aps = [[] for _ in range(n + 1)]
    for idx, ap in enumerate(ap_list):
        for elem in ap:
            element_aps[elem].append(idx)
    
    # Initial: base-4 digits {0,1,2}
    current = set()
    for x in range(1, n + 1):
        t = x
        ok = True
        while t > 0:
            if t % 4 == 3:
                ok = False
                break
            t //= 4
        if ok:
            current.add(x)
    
    best = set(current)
    best_size = len(current)
    rng = random.Random(12345)
    
    def can_add(x, s):
        for ap_idx in element_aps[x]:
            ap = ap_list[ap_idx]
            if all(e in s or e == x for e in ap):
                return False
        return True
    
    def try_add(x, s):
        if can_add(x, s):
            s.add(x)
            return True
        return False
    
    while time.time() < deadline:
        # Greedy add pass
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            candidates = [x for x in range(1, n + 1) if x not in current]
            rng.shuffle(candidates)
            for x in candidates:
                if try_add(x, current):
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
        
        # 2-exchange pass
        if time.time() < deadline - 5:
            in_set = list(current)
            rng.shuffle(in_set)
            for rem in in_set:
                if time.time() >= deadline - 2:
                    break
                current.discard(rem)
                # Try to add up to 2 elements
                added = []
                cands = [x for x in range(1, n + 1) if x not in current and x != rem]
                rng.shuffle(cands)
                for x in cands:
                    if len(added) >= 2:
                        break
                    if try_add(x, current):
                        added.append(x)
                
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
                
                if len(added) < 1:
                    # Revert
                    current.add(rem)
                elif len(added) == 1 and len(current) <= best_size:
                    # Revert
                    current.discard(added[0])
                    current.add(rem)
        
        # Perturbation if stuck
        if time.time() < deadline - 5:
            stuck = True
            for x in range(1, n + 1):
                if x not in current and can_add(x, current):
                    stuck = False
                    break
            if stuck:
                elems = list(current)
                rng.shuffle(elems)
                for rem in elems[:4]:
                    current.discard(rem)
                # Re-greedy
                cands = [x for x in range(1, n + 1) if x not in current]
                rng.shuffle(cands)
                for x in cands:
                    if try_add(x, current):
                        pass
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
    
    result = sorted(best)
    # Final check
    for ap in ap_list:
        if all(e in best for e in ap):
            # Fallback
            fb = set()
            for x in range(1, n + 1):
                t = x
                ok = True
                while t > 0:
                    if t % 4 == 3:
                        ok = False
                        break
                    t //= 4
                if ok:
                    fb.add(x)
            return sorted(fb)
    return result
