import time
import random
from itertools import combinations

# Precompute all 4-subsets of 16
N = 16
K = 4
ALL_COMBOS = list(combinations(range(1, N + 1), K))
M = len(ALL_COMBOS) # 1820

# Convert to bitmasks
ALL_SETS = [0] * M
for i, c in enumerate(ALL_COMBOS):
    mask = 0
    for x in c:
        mask |= (1 << (x - 1))
    ALL_SETS[i] = mask

# Precompute intersections
# inter[i][j] = ALL_SETS[i] & ALL_SETS[j]
# To save memory and time, we can compute on the fly in the inner loop?
# No, precomputing is better.
# 1820x1820 matrix.
# Use a flat list or list of lists.
# List of lists is fine.
INTER = [[0]*M for _ in range(M)]
for i in range(M):
    mi = ALL_SETS[i]
    for j in range(i+1, M):
        mj = ALL_SETS[j]
        inter_val = mi & mj
        INTER[i][j] = inter_val
        INTER[j][i] = inter_val

def can_add_fast(c_idx, family_indices, in_family):
    """
    Check if c_idx can be added to family_indices.
    Uses grouping optimization.
    """
    c_mask = ALL_SETS[c_idx]
    groups = {}
    
    for i in family_indices:
        s = ALL_SETS[i] & c_mask
        if s not in groups:
            groups[s] = []
        groups[s].append(i)
        
    for s, idxs in groups.items():
        n_g = len(idxs)
        if n_g < 2:
            continue
        # Check pairs within this group
        for u in range(n_g):
            i = idxs[u]
            for v in range(u+1, n_g):
                j = idxs[v]
                if INTER[i][j] == s:
                    return False
    return True

def construct():
    start_time = time.time()
    timeout = 50.0
    
    best_family = []
    best_size = 0
    
    # Initial Greedy
    random.seed(0)
    indices = list(range(M))
    random.shuffle(indices)
    
    current_family = []
    in_family = [False] * M
    
    for idx in indices:
        if can_add_fast(idx, current_family, in_family):
            current_family.append(idx)
            in_family[idx] = True
    
    best_size = len(current_family)
    best_family = current_family[:]
    
    # Local Search / Kick
    # We try to increase size by removing some sets and adding more.
    # Or just random restarts if stuck.
    
    # Better: Try to add a set. If fail, remove a random set and try again.
    
    # Let's do random restarts with the fast check.
    # The fast check allows more iterations.
    
    seed_counter = 1
    while time.time() - start_time < timeout:
        random.seed(seed_counter)
        indices = list(range(M))
        random.shuffle(indices)
        
        fam = []
        in_f = [False] * M
        
        for idx in indices:
            if can_add_fast(idx, fam, in_f):
                fam.append(idx)
                in_f[idx] = True
                
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam[:]
            
        # Try to improve current best by adding more?
        # If we have time, try to extend best_family
        if best_size < 250:
             # Try to find a set that can be added to best_family
             # This is O(M * Cost(can_add))
             # M=1820. Cost(can_add) ~ 1ms. Total ~ 1.8s.
             # Do this every few iterations?
             pass
             
        seed_counter += 1
        
    # Convert result
    result = []
    for idx in best_family:
        mask = ALL_SETS[idx]
        elems = []
        for i in range(N):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
