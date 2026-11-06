import time
import random
from itertools import combinations

N = 16
K = 4
ALL_SETS = []
MASKS = []
INDEX_BY_MASK = {}

# Precompute masks
for comb in combinations(range(1, N+1), K):
    mask = 0
    for x in comb:
        mask |= (1 << (x-1))
    MASKS.append(mask)
    ALL_SETS.append(comb)
    INDEX_BY_MASK[mask] = len(MASKS) - 1

M = len(MASKS)

def get_conflicts_on_fly(i, j):
    mi = MASKS[i]
    mj = MASKS[j]
    s_mask = mi & mj
    u_mask = ((1 << N) - 1) & ~(mi | mj)
    
    s_size = bin(s_mask).count('1')
    u_size = bin(u_mask).count('1')
    
    need = K - s_size
    
    if need < 0 or need > u_size:
        return []
        
    u_elems = [x+1 for x in range(N) if (u_mask >> x) & 1]
    
    conflicts = []
    for comb in combinations(u_elems, need):
        k_mask = s_mask
        for x in comb:
            k_mask |= (1 << (x-1))
        if k_mask in INDEX_BY_MASK:
            conflicts.append(INDEX_BY_MASK[k_mask])
            
    return conflicts

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_family = []
    
    while time.time() < deadline:
        order = list(range(M))
        random.shuffle(order)
        
        current_family = []
        conf_count = [0] * M
        
        for k in order:
            if conf_count[k] > 0:
                continue
            
            current_family.append(k)
            
            for i in current_family[:-1]:
                conflicts = get_conflicts_on_fly(i, k)
                for m in conflicts:
                    conf_count[m] += 1
                    
        if len(current_family) > len(best_family):
            best_family = current_family[:]
            
        if len(best_family) >= 250:
            break
            
    result = []
    for idx in best_family:
        result.append(ALL_SETS[idx])
        
    return result
