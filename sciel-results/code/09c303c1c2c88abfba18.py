import time
import random

N = 16
TOTAL_POINTS = N * N
ALL_MASK = (1 << TOTAL_POINTS) - 1

iso = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]

def precompute():
    dist = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    for i in range(TOTAL_POINTS):
        xi = i // N
        yi = i % N
        for j in range(i + 1, TOTAL_POINTS):
            xj = j // N
            yj = j % N
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist[i][j] = d
            dist[j][i] = d

    for p in range(TOTAL_POINTS):
        for s in range(TOTAL_POINTS):
            if p == s:
                continue
            d_ps = dist[p][s]
            mask = 0
            for q in range(TOTAL_POINTS):
                if q == p or q == s:
                    continue
                d_pq = dist[p][q]
                d_sq = dist[s][q]
                if d_pq == d_sq or d_pq == d_ps or d_sq == d_ps:
                    mask |= (1 << q)
            iso[p][s] = mask

def greedy_once(seed):
    random.seed(seed)
    S = 0
    Candidates = ALL_MASK
    
    while Candidates:
        # Get candidate indices
        cand_list = []
        temp = Candidates
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            cand_list.append(idx)
            temp ^= lsb
        
        gains = []
        for p in cand_list:
            gain_mask = 0
            temp_s = S
            while temp_s:
                lsb = temp_s & -temp_s
                s = lsb.bit_length() - 1
                gain_mask |= iso[p][s]
                temp_s ^= lsb
            
            effective_gain = gain_mask & Candidates
            count = effective_gain.bit_count()
            gains.append((count, p))
            
        if not gains:
            break
            
        max_count = max(g[0] for g in gains)
        bests = [p for c, p in gains if c == max_count]
        p = random.choice(bests)
        
        S |= (1 << p)
        
        # Update Candidates
        new_blocked = 0
        temp_s = S & ~(1 << p)
        while temp_s:
            lsb = temp_s & -temp_s
            s = lsb.bit_length() - 1
            new_blocked |= iso[p][s]
            temp_s ^= lsb
            
        newly_blocked = new_blocked & Candidates
        Candidates ^= newly_blocked
        Candidates &= ~(1 << p)
        
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    precompute()
    
    best_S = 0
    best_size = TOTAL_POINTS
    
    for seed in range(500):
        if time.time() > deadline:
            break
        S = greedy_once(seed)
        size = S.bit_count()
        if size < best_size:
            best_size = size
            best_S = S
            
    # Convert bitmask to list of (x, y)
    result = []
    for i in range(TOTAL_POINTS):
        if (best_S >> i) & 1:
            result.append((i // N, i % N))
            
    return result
