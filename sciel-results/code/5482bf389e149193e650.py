"""Rewritten program: maximize 4-uniform family on [16] with no 3-sunflower."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(1, n + 1), k):
        mask = 0
        for x in combo:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    total = len(all_sets)  # 1820
    
    # Precompute intersection masks
    inter_mask = [[0] * total for _ in range(total)]
    for i in range(total):
        si = all_sets[i]
        for j in range(i + 1, total):
            m = si & all_sets[j]
            inter_mask[i][j] = m
            inter_mask[j][i] = m
    
    def is_sunflower(i, j, l):
        return (inter_mask[i][j] == inter_mask[i][l]) and (inter_mask[i][j] == inter_mask[j][l])
    
    best_family = []
    best_size = 0
    
    def try_greedy(order):
        family = []
        
        for idx in order:
            if time.time() > deadline:
                return family
            valid = True
            for a in range(len(family)):
                fa = family[a]
                for b in range(a + 1, len(family)):
                    fb = family[b]
                    if is_sunflower(fa, fb, idx):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                family.append(idx)
        
        return family
    
    # Deterministic greedy
    order = list(range(total))
    family = try_greedy(order)
    if len(family) > best_size:
        best_size = len(family)
        best_family = family
    
    # Random restarts
    random.seed(42)
    while time.time() < deadline:
        order = list(range(total))
        random.shuffle(order)
        family = try_greedy(order)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
            if best_size >= 250:
                break
    
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elements = []
        for bit in range(n):
            if mask & (1 << bit):
                elements.append(bit + 1)
        result.append(sorted(elements))
    
    return result

# EVOLVE-BLOCK-END
