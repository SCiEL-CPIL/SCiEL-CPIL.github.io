import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set
    
    # Precompute LCMs and Conflicts
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
        
    lcms = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcms[i][j] = lcm(i, j)
            
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if lcms[a][b] == lcms[a][c] == lcms[b][c]:
                    conflict[a][b] |= (1 << c)
                    conflict[b][a] |= (1 << c)
                    conflict[a][c] |= (1 << b)
                    conflict[c][a] |= (1 << b)
                    conflict[b][c] |= (1 << a)
                    conflict[c][b] |= (1 << a)
                    
    best_size = 0
    best_set = []
    
    def greedy_random(order):
        selected = []
        avail = full_mask
        forb = 0
        
        for x in order:
            if not (avail & (1 << x)):
                continue
            
            # Add x
            selected.append(x)
            
            # Update forb
            for s in selected[:-1]:
                forb |= conflict[s][x]
                
            # Update avail
            avail &= ~forb
            avail &= ~(1 << x)
            
        return selected

    # 1. Natural order
    s = greedy_random(list(range(1, N + 1)))
    if len(s) > best_size:
        best_size = len(s)
        best_set = s[:]
        
    # 2. Random orders
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 15.0:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_random(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    def improve_set(s):
        s = s[:]
        s_set = set(s)
        changed = True
        while changed:
            changed = False
            # 1. Add-back
            forb = 0
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    forb |= conflict[s[i]][s[j]]
            
            for x in range(1, N + 1):
                if x in s_set:
                    continue
                if not (forb & (1 << x)):
                    s.append(x)
                    s_set.add(x)
                    changed = True
                    break
            
            # 2. Swap (optional, if add-back stalls)
            # Try to swap u in s with x not in s
            # This is expensive, so only do it if we have time and add-back failed
            if not changed and time.time() - start_time < deadline - 5.0:
                # Try simple swaps
                for u in s:
                    if changed: break
                    s_temp = [y for y in s if y != u]
                    forb_temp = 0
                    for i in range(len(s_temp)):
                        for j in range(i+1, len(s_temp)):
                            forb_temp |= conflict[s_temp[i]][s_temp[j]]
                    
                    for x in range(1, N + 1):
                        if x in s_set or x == u:
                            continue
                        if not (forb_temp & (1 << x)):
                            # Swap u with x
                            s.remove(u)
                            s_set.remove(u)
                            s.append(x)
                            s_set.add(x)
                            changed = True
                            break
                            
        return s

    best_set = improve_set(best_set)
    best_size = len(best_set)

    def solve(selected, avail_mask, forb_mask):
        nonlocal best_size, best_set
        
        if time.time() - start_time > deadline:
            return
            
        if len(selected) + avail_mask.bit_count() <= best_size:
            return
            
        if not avail_mask:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
            return
            
        # Variable selection: lowest set bit
        x_bit = avail_mask & -avail_mask
        x = x_bit.bit_length() - 1
        
        # Branch 1: Include x
        new_forb = forb_mask
        for s in selected:
            new_forb |= conflict[s][x]
            
        new_avail = avail_mask & ~(1 << x) & ~new_forb
        
        solve(selected + [x], new_avail, new_forb)
        
        # Branch 2: Exclude x
        solve(selected, avail_mask & ~(1 << x), forb_mask)
        
    # Start B&B
    initial_avail = full_mask
    initial_forb = 0
    solve([], initial_avail, initial_forb)
    
    return best_set
