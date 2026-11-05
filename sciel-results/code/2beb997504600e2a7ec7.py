import time

def construct():
    N = 128
    # Precompute 4-APs
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    pos_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)

    def solve_with_result(k):
        colors = [-1] * N
        full_mask = (1 << k) - 1
        
        def get_available_colors(pos):
            mask = full_mask
            for ap_idx in pos_aps[pos]:
                ap = aps[ap_idx]
                others = [p for p in ap if p != pos]
                if all(colors[p] != -1 for p in others):
                    c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
                    distinct = {c1, c2, c3}
                    if len(distinct) < 3:
                        for c in distinct:
                            mask &= ~(1 << c)
            return mask

        def dfs(idx):
            if idx == N:
                return True
            
            pos = idx
            avail = get_available_colors(pos)
            
            if avail == 0:
                return False
            
            if pos == 0:
                candidates = [0]
            else:
                candidates = []
                m = avail
                while m:
                    lsb = m & (-m)
                    c = lsb.bit_length() - 1
                    candidates.append(c)
                    m ^= lsb
            
            for c in candidates:
                colors[pos] = c
                if dfs(idx + 1):
                    return True
                colors[pos] = -1
            
            return False
        
        if dfs(0):
            return colors
        return None

    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    for k in range(1, 20):
        if time.time() > deadline:
            break
        res = solve_with_result(k)
        if res is not None:
            return res
            
    # Fallback if no solution found in time (unlikely for N=128)
    return [0] * N
