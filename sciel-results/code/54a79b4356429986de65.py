"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 256

def max_discrepancy_fast(signs):
    n = N
    best = 0
    for d in range(1, n):
        for s in range(d):
            total = 0
            for i in range(s, n, d):
                total += signs[i]
                if total > best:
                    best = total
                elif -total > best:
                    best = -total
    return best

def find_worst_ap(signs):
    n = N
    best_abs = 0
    best_s, best_d, best_len, best_sum = 0, 1, 1, 0
    for d in range(1, n):
        for s in range(d):
            total = 0
            length = 0
            for i in range(s, n, d):
                total += signs[i]
                length += 1
                if abs(total) > best_abs:
                    best_abs = abs(total)
                    best_s, best_d, best_len, best_sum = s, d, length, total
    return best_s, best_d, best_len, best_sum

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    best_signs = None
    best_disc = float('inf')
    
    rng = random.Random(42)
    
    # Try several initializations
    inits = []
    inits.append([rng.choice((-1, 1)) for _ in range(N)])
    inits.append([1 if i % 2 == 0 else -1 for i in range(N)])
    inits.append([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(N)])
    
    for init_signs in inits:
        if time.time() > deadline - 10:
            break
        
        signs = init_signs[:]
        disc = max_discrepancy_fast(signs)
        
        # Limited greedy repair
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            
            s, d, length, s_val = find_worst_ap(signs)
            if abs(s_val) <= 1:
                break
            
            target = -1 if s_val > 0 else 1
            candidates = [i for i in range(s, s + length * d, d) if signs[i] == target]
            if not candidates:
                candidates = list(range(s, s + length * d, d))
            
            best_flip = None
            best_new_disc = disc
            
            for idx in candidates[:3]:
                old = signs[idx]
                signs[idx] = -old
                new_disc = max_discrepancy_fast(signs)
                signs[idx] = old
                if new_disc < best_new_disc:
                    best_new_disc = new_disc
                    best_flip = idx
            
            if best_flip is None:
                break
            signs[best_flip] = -signs[best_flip]
            disc = best_new_disc
        
        if disc < best_disc:
            best_disc = disc
            best_signs = signs[:]
    
    if best_signs is None:
        best_signs = [1 if i % 2 == 0 else -1 for i in range(N)]
    
    return best_signs
# EVOLVE-BLOCK-END
