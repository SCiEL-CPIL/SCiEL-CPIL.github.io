import random
import time
from itertools import combinations

def is_sunflower(a, b, c):
    # a, b, c are distinct integers representing sets
    x = a & b & c
    pa = a ^ x
    pb = b ^ x
    pc = c ^ x
    return (pa & pb == 0) and (pa & pc == 0) and (pb & pc == 0)

def construct():
    start_time = time.time()
    time_limit = 85.0  # 85 seconds to be safe
    
    n = 16
    all_sets = list(combinations(range(n), 4))
    masks = []
    for s in all_sets:
        m = 0
        for i in s:
            m |= (1 << i)
        masks.append(m)
        
    best_family = []
    best_size = 0
    
    def can_add(mask, family):
        # Check if mask can be added to family without forming a sunflower
        for i in range(len(family)):
            a = family[i]
            for j in range(i+1, len(family)):
                b = family[j]
                if is_sunflower(a, b, mask):
                    return False
        return True

    def greedy_pass(seed):
        random.seed(seed)
        family = []
        candidates = masks[:]
        random.shuffle(candidates)
        
        while candidates:
            valids = []
            for c in candidates:
                if can_add(c, family):
                    valids.append(c)
            
            if not valids:
                break
                
            # Pick one randomly from valids
            chosen = random.choice(valids)
            family.append(chosen)
            candidates.remove(chosen)
            
            if time.time() - start_time > time_limit - 5:
                break
                
        return family

    seed = 0
    while time.time() - start_time < time_limit - 10:
        family = greedy_pass(seed)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
        seed += 1

    result = []
    for m in best_family:
        s = []
        for i in range(n):
            if m & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
