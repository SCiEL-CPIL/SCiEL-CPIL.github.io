import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 16
    k = 4
    
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
        
    def can_add(F, C):
        m = len(F)
        # Local reference for speed
        f_list = F
        for i in range(m):
            a = f_list[i]
            for j in range(i + 1, m):
                b = f_list[j]
                if (a & b) == (a & C) == (b & C):
                    return False
        return True

    best_family = []
    best_size = 0
    rng = random.Random(12345)
    
    while time.time() < deadline:
        shuffled_sets = all_sets[:]
        rng.shuffle(shuffled_sets)
        
        current_family = []
        
        for C in shuffled_sets:
            if time.time() > deadline:
                break
            
            if can_add(current_family, C):
                current_family.append(C)
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
            
    # Convert masks back to sorted lists of elements
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if (mask >> i) & 1:
                elems.append(i + 1) # 1-indexed as per task [1, n]
        result.append(sorted(elems))
        
    return result
