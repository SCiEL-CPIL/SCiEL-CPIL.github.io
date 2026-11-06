import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    N = 512
    
    # Precompute compatibility bitmasks
    # compat[i] is a bitmask where bit j is 1 if lcm(i, j) <= N
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if (i * j) // gcd(i, j) <= N:
                mask |= (1 << j)
        compat[i] = mask
        
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_mask(mask):
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if mask & ~compat[i]:
                return False
            m ^= lsb
        return True

    def complete(mask):
        # Greedily add all compatible elements
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if time.time() > deadline - 1.0:
                    return mask
                bit = 1 << x
                if mask & bit:
                    continue
                if mask & ~compat[x]:
                    continue
                mask |= bit
                changed = True
        return mask

    def local_search(mask):
        mask = complete(mask)
        best_mask = mask
        best_size = popcount(mask)
        improved = True
        rng = random.Random(42)
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            m = mask
            elements = []
            while m:
                lsb = m & -m
                elements.append(lsb.bit_length() - 1)
                m ^= lsb
            
            # 1-for-1 swaps
            # Precompute union of incompatibilities
            union_incompat = [0] * (N + 1)
            for u in elements:
                ui = 0
                for w in elements:
                    if w != u:
                        ui |= ~compat[w]
                union_incompat[u] = ui
            
            for u in elements:
                if time.time() > deadline - 2.0:
                    break
                ui = union_incompat[u]
                mask_minus_u = mask & ~(1 << u)
                for v in range(1, N + 1):
                    if time.time() > deadline - 2.0:
                        break
                    if v == u or (mask & (1 << v)):
                        continue
                    # Check if v is compatible with all in mask_minus_u
                    if mask_minus_u & ~compat[v]:
                        continue
                    new_mask = complete(mask_minus_u | (1 << v))
                    new_size = popcount(new_mask)
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        mask = new_mask
                        improved = True
                        break
            
            if improved:
                continue
            
            # 1-for-2 swaps
            for u in elements:
                if time.time() > deadline - 2.0:
                    break
                mask_minus_u = mask & ~(1 << u)
                new_mask = mask_minus_u
                added = 0
                for v in range(1, N + 1):
                    if time.time() > deadline - 2.0:
                        break
                    if added >= 2:
                        break
                    if new_mask & (1 << v):
                        continue
                    if new_mask & ~compat[v]:
                        continue
                    new_mask |= (1 << v)
                    added += 1
                if added >= 2:
                    new_mask = complete(new_mask)
                    new_size = popcount(new_mask)
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        mask = new_mask
                        improved = True
                        break
            
            if improved:
                continue

        # Final polish
        polish_rounds = 0
        while polish_rounds < 100 and time.time() < deadline - 1.0:
            polish_rounds += 1
            m = best_mask
            elems = []
            while m:
                lsb = m & -m
                elems.append(lsb.bit_length() - 1)
                m ^= lsb
            found = False
            for u in elems:
                if time.time() > deadline - 1.0:
                    break
                candidate = best_mask & ~(1 << u)
                candidate = complete(candidate)
                sz = popcount(candidate)
                if sz > best_size:
                    best_size = sz
                    best_mask = candidate
                    found = True
                    break
            if not found:
                break
        
        return best_mask

    candidates = []
    
    # 1. Divisors of highly composite numbers
    for M in range(1, N + 1):
        divs = []
        for i in range(1, M + 1):
            if M % i == 0:
                divs.append(i)
        mask = 0
        for d in divs:
            mask |= (1 << d)
        candidates.append(mask)
        
    # 2. Intervals
    for k in range(1, 30):
        mask = 0
        for i in range(1, k + 1):
            mask |= (1 << i)
        candidates.append(mask)
        
    # 3. Random restarts
    rng = random.Random(123)
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        mask = 1 << 1
        order = list(range(2, N + 1))
        rng.shuffle(order)
        for x in order:
            if mask & ~compat[x] == 0:
                mask |= (1 << x)
        candidates.append(mask)
    
    best_overall_mask = 0
    best_overall_size = 0
    
    # Sort candidates by initial size
    candidates.sort(key=lambda m: popcount(m), reverse=True)
    
    for mask in candidates:
        if time.time() > deadline - 3.0:
            break
        if not is_valid_mask(mask):
            m = mask
            while m:
                lsb = m & -m
                x = lsb.bit_length() - 1
                if mask & ~compat[x]:
                    mask &= ~lsb
                m ^= lsb
        
        optimized_mask = local_search(mask)
        size = popcount(optimized_mask)
        if size > best_overall_size:
            best_overall_size = size
            best_overall_mask = optimized_mask
            
    # Final polish
    m = best_overall_mask
    for x in range(1, N + 1):
        if m & (1 << x):
            continue
        if m & ~compat[x] == 0:
            m |= (1 << x)
    
    result = []
    mm = m
    while mm:
        lsb = mm & -mm
        result.append(lsb.bit_length() - 1)
        mm ^= lsb
        
    if not result:
        return list(range(1, 23))
        
    return sorted(result)
