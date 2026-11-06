import time
import random

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N
FULL_MASK = (1 << NUM_POINTS) - 1

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # Precomputation
    dist2 = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        for j in range(i+1, NUM_POINTS):
            dx = POINTS[i][0] - POINTS[j][0]
            dy = POINTS[i][1] - POINTS[j][1]
            d = dx*dx + dy*dy
            dist2[i][j] = d
            dist2[j][i] = d
            
    iso_pair_masks = [[] for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        for j in range(i+1, NUM_POINTS):
            for k in range(j+1, NUM_POINTS):
                d1 = dist2[i][j]
                d2 = dist2[i][k]
                d3 = dist2[j][k]
                if d1 == d2 or d1 == d3 or d2 == d3:
                    mask_jk = (1 << j) | (1 << k)
                    iso_pair_masks[i].append(mask_jk)
                    
                    mask_ik = (1 << i) | (1 << k)
                    iso_pair_masks[j].append(mask_ik)
                    
                    mask_ij = (1 << i) | (1 << j)
                    iso_pair_masks[k].append(mask_ij)

    def can_add(p, S_bitset):
        for mask in iso_pair_masks[p]:
            if (S_bitset & mask) == mask:
                return False
        return True

    def is_maximal(S_bitset):
        comp = FULL_MASK ^ S_bitset
        while comp:
            lsb = comp & -comp
            p = lsb.bit_length() - 1
            if can_add(p, S_bitset):
                return False
            comp ^= lsb
        return True

    def greedy_construct(rng):
        S_bitset = 0
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        for p in order:
            if can_add(p, S_bitset):
                S_bitset |= (1 << p)
        return S_bitset

    def local_search(S_bitset, rng, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            points_in_S = []
            temp = S_bitset
            while temp:
                lsb = temp & -temp
                p = lsb.bit_length() - 1
                points_in_S.append(p)
                temp ^= lsb
                
            rng.shuffle(points_in_S)
            
            for p in points_in_S:
                if time.time() >= deadline:
                    return S_bitset
                new_S = S_bitset ^ (1 << p)
                if is_maximal(new_S):
                    S_bitset = new_S
                    improved = True
                    break
                    
        return S_bitset

    best_S = 0
    best_size = float('inf')
    
    restart = 0
    while time.time() < deadline - 5:
        rng = random.Random(restart * 7919 + 12345)
        S = greedy_construct(rng)
        S = local_search(S, rng, deadline - 2)
        
        size = bin(S).count('1')
        if size < best_size:
            best_size = size
            best_S = S
            
        restart += 1
        
    result = []
    temp = best_S
    while temp:
        lsb = temp & -temp
        p = lsb.bit_length() - 1
        result.append(POINTS[p])
        temp ^= lsb
        
    return result
