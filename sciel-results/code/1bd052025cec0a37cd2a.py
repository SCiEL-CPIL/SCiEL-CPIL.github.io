import random
import math
import time

def construct():
    max_mod = 128
    base_moduli = list(range(2, max_mod + 1))
    
    # Precompute gcds
    gcds = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            g = math.gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    best_set = []
    best_count = 0
    
    start_time = time.time()
    deadline = start_time + 80
    
    while time.time() < deadline:
        moduli = base_moduli.copy()
        random.shuffle(moduli)
        current_set = []
        
        for m in moduli:
            # Check if m is coprime to any in current_set
            skip = False
            for m2, _ in current_set:
                if gcds[m][m2] == 1:
                    skip = True
                    break
            if skip:
                continue
                
            # Find valid residues
            forbidden = [False] * m
            for m2, r2 in current_set:
                g = gcds[m][m2]
                rem = r2 % g
                for r in range(rem, m, g):
                    forbidden[r] = True
                    
            # Find an unforbidden r
            valid_rs = [i for i in range(m) if not forbidden[i]]
            if valid_rs:
                r = random.choice(valid_rs)
                current_set.append((m, r))
                
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set.copy()
            
    return [[r, m] for m, r in best_set]
