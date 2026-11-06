"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

N = 9
TOTAL = 1 << N

# EVOLVE-BLOCK-START
def _vectorized_validity_check(v, path_array, used_set):
    if v in used_set:
        return False
    if len(path_array) == 0:
        return True
    diffs = path_array ^ v
    unpacked = np.unpackbits(diffs.astype(np.uint8), axis=1, bitorder='little')
    popcounts = np.sum(unpacked[:, :N], axis=1)
    if np.any(popcounts == 1):
        return False
    return True

def _fast_greedy_restart(rng, time_limit=1.0):
    start_t = time.time()
    start_v = rng.randint(0, TOTAL - 1)
    path = [start_v]
    used = {start_v}
    path_array = np.array([], dtype=np.uint16)
    
    while True:
        if time.time() - start_t > time_limit:
            break
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if _vectorized_validity_check(v, path_array, used):
                candidates.append(v)
        if not candidates:
            break
        
        # Heuristic: Maximize number of unused neighbors
        scores = []
        for v in candidates:
            score = 0
            for b in range(N):
                if (v ^ (1 << b)) not in used:
                    score += 1
            scores.append(score)
            
        max_score = max(scores)
        if max_score == 0:
            best_v = rng.choice(candidates)
        else:
            max_candidates = [c for c, s in zip(candidates, scores) if s == max_score]
            best_v = rng.choice(max_candidates)
            
        path.append(best_v)
        used.add(best_v)
        path_array = np.array(path[:-1], dtype=np.uint16)
        
    return path

def construct():
    best_path = []
    best_len = 0
    start_time = time.time()
    time_limit = 115
    
    trial = 0
    while time.time() - start_time < time_limit:
        rng = random.Random(trial)
        path = _fast_greedy_restart(rng, time_limit=1.0)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            
        trial += 1
        
    if not best_path:
        # Fallback
        best_path = [0]
        
    return best_path
# EVOLVE-BLOCK-END
