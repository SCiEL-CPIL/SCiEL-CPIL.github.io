import time
import random
from itertools import combinations

N = 16
K_SIZE = 4
ALL_MASK = (1 << N) - 1
ALL_SETS = []
MASK_TO_IDX = {}

for comb in combinations(range(N), K_SIZE):
    mask = 0
    for i in comb:
        mask |= (1 << i)
    ALL_SETS.append(mask)
    MASK_TO_IDX[mask] = len(ALL_SETS) - 1

NUM_SETS = len(ALL_SETS)

def construct():
    best_family = []
    best_size = 0
    start_time = time.time()
    deadline = start_time + 55.0
    
    def run_greedy_v2(seed):
        random.seed(seed)
        conflict_counts = [0] * NUM_SETS
        chosen = []
        available_set = set(range(NUM_SETS))
        
        while available_set:
            min_conf = float('inf')
            min_candidates = []
            for idx in available_set:
                c = conflict_counts[idx]
                if c < min_conf:
                    min_conf = c
                    min_candidates = [idx]
                elif c == min_conf:
                    min_candidates.append(idx)
            
            next_idx = random.choice(min_candidates)
            
            chosen.append(next_idx)
            available_set.remove(next_idx)
            
            next_mask = ALL_SETS[next_idx]
            
            for prev_idx in chosen:
                if prev_idx == next_idx:
                    continue
                prev_mask = ALL_SETS[prev_idx]
                
                K = next_mask & prev_mask
                U = ALL_MASK & ~(next_mask | prev_mask)
                
                k_pop = bin(K).count('1')
                r = K_SIZE - k_pop
                
                if r < 0:
                    continue
                
                u_bits = [i for i in range(N) if (U >> i) & 1]
                
                for comb in combinations(u_bits, r):
                    V = 0
                    for bit in comb:
                        V |= (1 << bit)
                    C_mask = K | V
                    c_idx = MASK_TO_IDX.get(C_mask)
                    if c_idx is not None and c_idx in available_set:
                        conflict_counts[c_idx] += 1
                        
        return [ALL_SETS[i] for i in chosen]

    for i in range(1000):
        if time.time() > deadline:
            break
        
        family = run_greedy_v2(i)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
            
    result = []
    for mask in best_family:
        s = []
        for i in range(N):
            if (mask >> i) & 1:
                s.append(i + 1)
        result.append(s)
        
    return result[:250]
