import time
import random
import sys

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # Safety margin for 90s limit

    N = 256
    FULL = (1 << N) - 1

    # conflict_mask[i]: bitmask of j where (i+1)*(j+1) % ((i+1)+(j+1)) == 0
    cm = [0] * N
    for i in range(N):
        a = i + 1
        m = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                m |= (1 << j)
        cm[i] = m

    odds = [i for i in range(N) if (i + 1) % 2 == 1]
    evens = [i for i in range(N) if (i + 1) % 2 == 0]
    
    # Precompute v2 for potential ordering strategies
    def v2(n):
        c = 0
        while n % 2 == 0 and n > 0:
            c += 1
            n //= 2
        return c

    evens_by_v2 = {}
    for i in evens:
        val = v2(i + 1)
        if val not in evens_by_v2:
            evens_by_v2[val] = []
        evens_by_v2[val].append(i)

    # Group elements by v2 for smarter swap strategies
    all_by_v2 = {}
    for i in range(N):
        val = v2(i + 1)
        if val not in all_by_v2:
            all_by_v2[val] = []
        all_by_v2[val].append(i)

    def popcount(x):
        return x.bit_count()

    def is_valid(m):
        for i in range(N):
            if (m >> i) & 1:
                if cm[i] & m:
                    return False
        return True

    def greedy_add(start, order):
        m = start
        for e in order:
            if (m >> e) & 1:
                continue
            if cm[e] & m:
                continue
            m |= (1 << e)
        return m

    def local_search(m, iters, deadline):
        best = m
        bestsz = popcount(m)
        
        for _ in range(iters):
            if time.time() > deadline:
                break
            
            if random.random() < 0.7:
                # Try to add a random element
                e = random.randrange(N)
                if (m >> e) & 1:
                    continue
                if not (cm[e] & m):
                    m |= (1 << e)
                    s = popcount(m)
                    if s > bestsz:
                        best, bestsz = m, s
                else:
                    # Try to swap: remove a conflict c, add e
                    conf = cm[e] & m
                    if conf:
                        # Smart swap: prefer removing elements with same v2 class
                        e_v2 = v2(e + 1)
                        # Try same v2 first
                        same_v2_conf = 0
                        for c_idx in range(N):
                            if (conf >> c_idx) & 1:
                                if v2(c_idx + 1) == e_v2:
                                    same_v2_conf |= (1 << c_idx)
                        
                        if same_v2_conf:
                            c = (same_v2_conf & -same_v2_conf).bit_length() - 1
                        else:
                            c = (conf & -conf).bit_length() - 1
                        
                        newm = (m & ~(1 << c)) | (1 << e)
                        # Check validity of newm
                        if not (cm[e] & (newm & ~(1 << e))):
                            s = popcount(newm)
                            if s >= bestsz:
                                m = newm
                                if s > bestsz:
                                    best, bestsz = m, s
            else:
                # Try to remove a random element and add 1-2 others
                if m == 0:
                    continue
                c = random.randrange(N)
                if not ((m >> c) & 1):
                    continue
                
                trial = m & ~(1 << c)
                added = 0
                # Try to add random elements
                for _ in range(20):
                    e = random.randrange(N)
                    if (trial >> e) & 1:
                        continue
                    if not (cm[e] & trial):
                        trial |= (1 << e)
                        added += 1
                        if added >= 2:
                            break
                s = popcount(trial)
                if s > popcount(m):
                    m = trial
                    if s > bestsz:
                        best, bestsz = m, s
        return best

    def polish(m, deadline):
        """Systematic single-element additions and swaps"""
        best = m
        bestsz = popcount(m)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any missing element
            for e in range(N):
                if (m >> e) & 1:
                    continue
                if not (cm[e] & m):
                    m |= (1 << e)
                    s = popcount(m)
                    if s > bestsz:
                        best, bestsz = m, s
                    improved = True
        return best

    def final_polish(m, deadline):
        """More aggressive final polish with systematic swaps"""
        best = m
        bestsz = popcount(m)
        
        # First, do standard polish
        m = polish(m, min(deadline - 5, time.time() + 10))
        bestsz = popcount(m)
        
        # Then try systematic swaps
        if time.time() < deadline:
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                # Try to find a beneficial swap
                # For each element not in m, check if we can swap it in
                for e in range(N):
                    if (m >> e) & 1:
                        continue
                    conf = cm[e] & m
                    if not conf:
                        # Can add directly
                        m |= (1 << e)
                        bestsz = popcount(m)
                        continue
                    
                    # Try swapping: remove one conflict, add e
                    # Prefer removing elements that are "less connected"
                    best_swap = None
                    best_swap_score = -1
                    
                    for c in range(N):
                        if not ((conf >> c) & 1):
                            continue
                        
                        # Score: prefer removing elements with fewer conflicts in m
                        c_conf_count = popcount(cm[c] & (m & ~(1 << c)))
                        
                        if c_conf_count < best_swap_score or best_swap is None:
                            best_swap = c
                            best_swap_score = c_conf_count
                    
                    if best_swap is not None:
                        newm = (m & ~(1 << best_swap)) | (1 << e)
                        if not (cm[e] & (newm & ~(1 << e))):
                            newsz = popcount(newm)
                            if newsz > bestsz:
                                m = newm
                                bestsz = newsz
        
        return m

    best = 0
    bestsz = 0

    def consider(m):
        nonlocal best, bestsz
        if not is_valid(m):
            return
        s = popcount(m)
        if s > bestsz:
            best, bestsz = m, s

    # 1) All odds baseline
    base = 0
    for i in odds:
        base |= (1 << i)
    consider(base)

    # 2) Greedy evens on odds, multiple strategies
    # Strategy A: Random orders
    for _ in range(50):
        if time.time() > DEADLINE - 20:
            break
        ro = evens[:]
        random.shuffle(ro)
        consider(greedy_add(base, ro))

    # Strategy B: v2 ordered (low to high, high to low)
    low_v2 = []
    high_v2 = []
    for k in sorted(evens_by_v2.keys()):
        low_v2.extend(evens_by_v2[k])
    for k in sorted(evens_by_v2.keys(), reverse=True):
        high_v2.extend(evens_by_v2[k])
    
    consider(greedy_add(base, low_v2))
    consider(greedy_add(base, high_v2))

    # 3) Random cores + Greedy
    for _ in range(30):
        if time.time() > DEADLINE - 20:
            break
        core = 0
        # Pick a few random odds and evens to start
        for i in random.sample(range(N), random.randint(1, 5)):
            if not (cm[i] & core):
                core |= (1 << i)
        ro = evens[:]
        random.shuffle(ro)
        consider(greedy_add(core, ro))

    # 4) Local Search from best candidates
    remaining = DEADLINE - time.time()
    ls_time = remaining * 0.5
    polish_time = remaining * 0.3
    
    # Run local search
    if time.time() < DEADLINE - 10:
        consider(local_search(best, 1000000, DEADLINE - polish_time - 5))

    # 5) Final polish
    if time.time() < DEADLINE - 2:
        consider(final_polish(best, DEADLINE - 2))

    if best == 0:
        best = 1

    # Final validation
    if not is_valid(best):
        # Fallback
        base = 0
        for i in odds:
            base |= (1 << i)
        best = base

    result = sorted([i + 1 for i in range(N) if (best >> i) & 1])
    return result
