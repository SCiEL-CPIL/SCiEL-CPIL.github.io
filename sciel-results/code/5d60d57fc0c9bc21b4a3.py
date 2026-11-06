import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    
    # Precompute conflict masks
    # conflict_masks[i] has bits set for j where i and j conflict
    conflict_masks = [0] * N
    
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask |= (1 << j)
        conflict_masks[i] = mask
        
    best_size = 0
    best_set_mask = 0
    
    # Greedy initialization
    def greedy_init():
        S = 0
        U = (1 << N) - 1
        while U:
            # Pick vertex with max degree in U
            max_deg = -1
            max_v = -1
            u_tmp = U
            while u_tmp:
                lsb = u_tmp & -u_tmp
                v = lsb.bit_length() - 1
                deg = (conflict_masks[v] & U).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    max_v = v
                u_tmp ^= lsb
            if max_v == -1:
                break
            S |= (1 << max_v)
            U &= ~(conflict_masks[max_v] | (1 << max_v))
        return S.bit_count(), S

    g_size, g_mask = greedy_init()
    best_size = g_size
    best_set_mask = g_mask
    
    def solve(U, S, k):
        nonlocal best_size, best_set_mask
        if time.time() > deadline:
            return
        
        # Pruning
        if k + U.bit_count() <= best_size:
            return
            
        if U == 0:
            if k > best_size:
                best_size = k
                best_set_mask = S
            return
            
        # Find v with max degree in U
        max_deg = -1
        max_v = -1
        u_tmp = U
        while u_tmp:
            lsb = u_tmp & -u_tmp
            v = lsb.bit_length() - 1
            deg = (conflict_masks[v] & U).bit_count()
            if deg > max_deg:
                max_deg = deg
                max_v = v
            u_tmp ^= lsb
            
        if max_v == -1:
            return

        # Branch 1: Include max_v
        U1 = U & ~(conflict_masks[max_v] | (1 << max_v))
        S1 = S | (1 << max_v)
        solve(U1, S1, k + 1)
        
        # Branch 2: Exclude max_v
        U2 = U & ~(1 << max_v)
        solve(U2, S, k)

    solve((1 << N) - 1, 0, 0)
    
    # Convert best_set_mask to list of values
    result = []
    mask = best_set_mask
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        mask ^= lsb
        
    return sorted(result)
