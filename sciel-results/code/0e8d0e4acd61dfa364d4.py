import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    n = len(signs)
    s = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub = s[r::d]
            if len(sub) < 2:
                continue
            cum = np.concatenate(([0], np.cumsum(sub)))
            min_p = np.minimum.accumulate(cum)
            max_p = np.maximum.accumulate(cum)
            msub = np.max(cum[1:] - min_p[:-1])
            nsub = np.min(cum[1:] - max_p[:-1])
            cm = max(abs(int(msub)), abs(int(nsub)))
            if cm > max_abs:
                max_abs = cm
    return max_abs

def _build_ap_structure(n):
    aps = []
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            idxs = list(range(r, n, d))
            if len(idxs) >= 2:
                aps.append(np.array(idxs, dtype=np.int32))
    n_aps = len(aps)
    
    pos_aps = [[] for _ in range(n)]
    for a, idxs in enumerate(aps):
        for i in idxs:
            pos_aps[int(i)].append(a)
    
    return aps, pos_aps, n_aps

def _init_ap_sums(aps, signs):
    n_aps = len(aps)
    max_sums = np.zeros(n_aps, dtype=np.int32)
    min_sums = np.zeros(n_aps, dtype=np.int32)
    s = signs
    for a, idxs in enumerate(aps):
        sub = s[idxs]
        cum = np.concatenate(([0], np.cumsum(sub)))
        min_p = np.minimum.accumulate(cum)
        max_p = np.maximum.accumulate(cum)
        max_sums[a] = np.max(cum[1:] - min_p[:-1])
        min_sums[a] = np.min(cum[1:] - max_p[:-1])
    return max_sums, min_sums

def _recompute_ap(a, aps, signs, max_sums, min_sums):
    idxs = aps[a]
    sub = signs[idxs]
    cum = np.concatenate(([0], np.cumsum(sub)))
    min_p = np.minimum.accumulate(cum)
    max_p = np.maximum.accumulate(cum)
    max_sums[a] = np.max(cum[1:] - min_p[:-1])
    min_sums[a] = np.min(cum[1:] - max_p[:-1])

def _compute_disc(max_sums, min_sums):
    return max(int(np.max(max_sums)), int(-np.min(min_sums)), 1)

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 256
    rng = random.Random(12345)
    
    aps, pos_aps, n_aps = _build_ap_structure(n)
    
    # Candidates
    candidates = []
    candidates.append([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)])
    candidates.append([1 if i % 2 == 0 else -1 for i in range(n)])
    for seed in range(10):
        rng_c = random.Random(seed * 1000 + 7)
        candidates.append([rng_c.choice((-1, 1)) for _ in range(n)])
    
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    for c in candidates:
        if time.time() > deadline - 10:
            break
        d = _max_ap_discrepancy(c)
        if d < best_disc:
            best_disc = d
            best_signs = c[:]
    
    signs = best_signs[:]
    max_sums, min_sums = _init_ap_sums(aps, signs)
    current_disc = _compute_disc(max_sums, min_sums)
    
    # SA
    temp = 2.5
    cool_rate = 0.999
    min_temp = 0.05
    
    while time.time() < deadline - 3 and temp > min_temp:
        remaining = deadline - 3 - (time.time() - start_time)
        moves = max(50, int(remaining * 500))
        moves = min(moves, 1500)
        
        for _ in range(moves):
            if time.time() > deadline - 3:
                break
            
            idx = rng.randint(0, n - 1)
            old_val = signs[idx]
            signs[idx] = -old_val
            
            affected = pos_aps[idx]
            for a in affected:
                _recompute_ap(a, aps, signs, max_sums, min_sums)
            
            new_disc = _compute_disc(max_sums, min_sums)
            
            if new_disc < current_disc:
                current_disc = new_disc
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            else:
                dd = new_disc - current_disc
                if dd > 0:
                    if temp > min_temp and rng.random() < math.exp(-dd / temp):
                        current_disc = new_disc
                    else:
                        signs[idx] = old_val
                        for a in affected:
                            _recompute_ap(a, aps, signs, max_sums, min_sums)
                else:
                    current_disc = new_disc
        
        temp *= cool_rate
    
    # Polish
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for idx in range(n):
            if time.time() > deadline - 1:
                break
            old_val = signs[idx]
            signs[idx] = -old_val
            
            affected = pos_aps[idx]
            for a in affected:
                _recompute_ap(a, aps, signs, max_sums, min_sums)
            
            new_disc = _compute_disc(max_sums, min_sums)
            
            if new_disc < current_disc:
                current_disc = new_disc
                improved = True
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            else:
                signs[idx] = old_val
                for a in affected:
                    _recompute_ap(a, aps, signs, max_sums, min_sums)
    
    final_disc = _max_ap_discrepancy(best_signs)
    if final_disc < best_disc:
        best_disc = final_disc
    
    return best_signs
