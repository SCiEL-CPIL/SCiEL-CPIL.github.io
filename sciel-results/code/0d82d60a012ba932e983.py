import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    m = 40
    total_sets = 1 << n
    
    # Precompute comparability bitmasks
    # comp_bits[a] has bit b set if a and b are comparable
    comp_bits = [0] * total_sets
    for a in range(total_sets):
        mask = 0
        for b in range(total_sets):
            if a == b: continue
            # Check if a is subset of b or b is subset of a
            if (a & b) == a or (a & b) == b:
                mask |= (1 << b)
        comp_bits[a] = mask

    # Greedy Initialization
    S = set()
    in_S = [False] * total_sets
    G = [0] * total_sets # Gain for adding y to current S
    
    # Start with empty set and full set? Or just greedy from empty.
    # Greedy from empty:
    # G is initially 0 for all.
    # Pick any? Let's pick 0.
    S.add(0)
    in_S[0] = True
    # Update G
    for v in range(total_sets):
        if not in_S[v]:
            # G[v] += comp_bits[v] & (1<<0) ? No, comp is symmetric.
            # G[v] is sum of comp[v][u] for u in S.
            # Since S={0}, G[v] = 1 if v comparable to 0.
            # 0 is subset of everything. So G[v] = 1 for all v != 0.
            pass # We can just initialize G properly or update.
            
    # Better: Initialize G for S={0}
    for v in range(total_sets):
        if v != 0:
            G[v] = 1 # 0 is subset of v
    
    # Add remaining m-1 sets
    while len(S) < m:
        # Find v not in S with max G[v]
        best_v = -1
        best_g = -1
        for v in range(total_sets):
            if not in_S[v] and G[v] > best_g:
                best_g = G[v]
                best_v = v
        
        if best_v == -1: break
        
        # Add best_v to S
        S.add(best_v)
        in_S[best_v] = True
        
        # Update G for all u not in S
        # G[u] += 1 if u comparable to best_v
        # This is equivalent to G[u] += (comp_bits[u] >> best_v) & 1
        # Or just check comparability
        for u in range(total_sets):
            if not in_S[u]:
                # Check if u and best_v are comparable
                # (u & best_v) == u or (u & best_v) == best_v
                if (u & best_v) == u or (u & best_v) == best_v:
                    G[u] += 1
                    
    # Now S has m sets.
    # Initialize C and G for local search
    C = [0] * total_sets
    G = [0] * total_sets
    
    def recalc_C_G():
        # C[u] for u in S: sum of comp[u][v] for v in S, v!=u
        # G[v] for v not in S: sum of comp[v][u] for u in S
        # Using bitmasks
        mask_S = 0
        for u in S:
            mask_S |= (1 << u)
            
        for u in S:
            # C[u] is popcount(comp_bits[u] & mask_S) - 1 (exclude self)
            C[u] = (comp_bits[u] & mask_S).bit_count() - 1
            
        for v in range(total_sets):
            if not in_S[v]:
                G[v] = (comp_bits[v] & mask_S).bit_count()
                
    recalc_C_G()
    
    best_score = -1
    best_S = S.copy()
    
    # Calculate current score
    def calc_score(S_set):
        score = 0
        for u in S_set:
            for v in S_set:
                if u < v and ((u & v) == u or (u & v) == v):
                    score += 1
        return score
    
    current_score = calc_score(S)
    best_score = current_score
    best_S = S.copy()
    
    # ILS Loop
    while time.time() < deadline:
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_x = -1
            best_y = -1
            
            # Find best swap
            # x in S, y not in S
            # Delta = -C[x] + G[y] - (1 if comparable(x,y) else 0)
            for x in S:
                cx = C[x]
                for y in range(total_sets):
                    if in_S[y]: continue
                    gy = G[y]
                    # Check comparability
                    comp_xy = 1 if ((x & y) == x or (x & y) == y) else 0
                    delta = -cx + gy - comp_xy
                    if delta > best_delta:
                        best_delta = delta
                        best_x = x
                        best_y = y
            
            if best_delta > 0:
                # Apply swap
                S.remove(best_x)
                in_S[best_x] = False
                S.add(best_y)
                in_S[best_y] = True
                
                # Update Score
                current_score += best_delta
                
                # Recalculate C and G (Fast enough)
                recalc_C_G()
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
            else:
                improved = False
        
        # Perturbation
        if time.time() > deadline: break
        
        # Randomly swap 2-5 pairs
        num_swaps = random.randint(2, 5)
        for _ in range(num_swaps):
            # Pick random x in S
            x = random.choice(list(S))
            # Pick random y not in S
            # Efficiently pick y
            while True:
                y = random.randint(0, total_sets - 1)
                if not in_S[y]:
                    break
            
            # Swap
            S.remove(x)
            in_S[x] = False
            S.add(y)
            in_S[y] = True
            
        # Recalculate Score and C/G
        current_score = calc_score(S)
        recalc_C_G()
        
        if current_score > best_score:
            best_score = current_score
            best_S = S.copy()
            
    # Convert best_S to lists
    result = []
    for mask in best_S:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
        
    return result
