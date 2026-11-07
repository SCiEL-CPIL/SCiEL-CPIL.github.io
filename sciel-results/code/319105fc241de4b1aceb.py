import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    m = 40
    max_mask = (1 << n) - 1
    
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (b & a) == b

    def count_comparables(family):
        score = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if is_comparable(family[i], family[j]):
                    score += 1
        return score

    def greedy_init():
        family = []
        family.append(random.randint(0, max_mask))
        
        for _ in range(m - 1):
            best_candidate = -1
            best_score = -1
            candidates = []
            for _ in range(50):
                candidates.append(random.randint(0, max_mask))
            for s in family:
                for _ in range(2):
                    bit = random.randint(0, n-1)
                    candidates.append(s ^ (1 << bit))
                    
            for cand in candidates:
                if cand in family:
                    continue
                inc = 0
                for s in family:
                    if is_comparable(cand, s):
                        inc += 1
                if inc > best_score:
                    best_score = inc
                    best_candidate = cand
            
            if best_candidate != -1:
                family.append(best_candidate)
            else:
                while True:
                    r = random.randint(0, max_mask)
                    if r not in family:
                        family.append(r)
                        break
        return family

    best_family = None
    best_score = -1
    
    family = greedy_init()
    current_score = count_comparables(family)
    current_set = set(family)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99999
    
    while time.time() < deadline:
        k = random.randint(0, m - 1)
        old_mask = family[k]
        
        if random.random() < 0.5:
            new_mask = old_mask
            bit = random.randint(0, n - 1)
            new_mask ^= (1 << bit)
        else:
            new_mask = random.randint(0, max_mask)
            
        if new_mask == old_mask:
            continue
        if new_mask in current_set:
            continue
            
        old_contrib = 0
        for j in range(m):
            if j == k: continue
            if is_comparable(old_mask, family[j]):
                old_contrib += 1
                
        new_contrib = 0
        for j in range(m):
            if j == k: continue
            if is_comparable(new_mask, family[j]):
                new_contrib += 1
                
        delta = new_contrib - old_contrib
        
        if delta > 0:
            accept = True
        else:
            if T > 0:
                prob = math.exp(delta / T)
            else:
                prob = 0
            accept = random.random() < prob
            
        if accept:
            current_set.remove(old_mask)
            current_set.add(new_mask)
            family[k] = new_mask
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_family = family.copy()
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    if best_family is None:
        best_family = family
        
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result
