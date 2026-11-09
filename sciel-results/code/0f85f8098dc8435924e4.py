import time
import random
import numpy as np
import math

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin before 90s kill

    # 1. Precompute APs and affected-AP lists
    ap_indices = []
    for d in range(1, n + 1):
        for a in range(n):
            idx = np.arange(a, n, d)
            if len(idx) > 0:
                ap_indices.append(idx)
    
    M = len(ap_indices)
    
    affected_aps = [[] for _ in range(n)]
    for k, idx in enumerate(ap_indices):
        for pos in idx:
            affected_aps[pos].append(k)
    
    affected_aps_np = [np.array(ks, dtype=np.int32) if ks else np.array([], dtype=np.int32) for ks in affected_aps]

    # 2. Initialize with Thue-Morse
    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    s = thue_morse_signs(n)
    
    ap_sums = np.zeros(M, dtype=np.int32)
    for k in range(M):
        ap_sums[k] = np.sum(s[ap_indices[k]])
        
    current_max = np.max(np.abs(ap_sums))
    
    best_s = s.copy()
    best_max = current_max

    rng = random.Random(42)

    # Helper: compute current max from ap_sums
    def get_max():
        return int(np.max(np.abs(ap_sums)))

    # Helper: flip position i, update ap_sums, return delta_max info
    def flip(i):
        old_sign = s[i]
        s[i] = -old_sign
        delta = -2 * old_sign
        ks = affected_aps_np[i]
        if len(ks) > 0:
            ap_sums[ks] += delta
        return old_sign, ks, delta

    def unflip(i, old_sign, ks, delta):
        s[i] = old_sign
        if len(ks) > 0:
            ap_sums[ks] -= delta

    # Phase 1: Greedy descent to local optimum
    improved = True
    while improved and time.time() < deadline - 40.0:
        improved = False
        for i in range(n):
            old_sign, ks, delta = flip(i)
            new_max = get_max()
            if new_max < current_max:
                current_max = new_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_s = s.copy()
            else:
                unflip(i, old_sign, ks, delta)
    
    # Phase 2: Simulated Annealing with restarts
    T_start = 5.0
    T_min = 0.05
    cooling = 0.9995
    
    while time.time() < deadline - 5.0:
        # Restart: go back to best and perturb
        s = best_s.copy()
        ap_sums = np.zeros(M, dtype=np.int32)
        for k in range(M):
            ap_sums[k] = np.sum(s[ap_indices[k]])
        current_max = best_max
        
        # Perturb
        num_flips = rng.randint(5, 15)
        for _ in range(num_flips):
            p = rng.randint(0, n - 1)
            old_sign, ks, delta = flip(p)
        
        current_max = get_max()
        
        T = T_start
        # SA loop
        while time.time() < deadline - 5.0 and T > T_min:
            i = rng.randint(0, n - 1)
            old_sign, ks, delta = flip(i)
            new_max = get_max()
            
            if new_max <= current_max:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_s = s.copy()
            else:
                delta_obj = new_max - current_max
                if delta_obj == 0:
                    if rng.random() < 0.5:
                        current_max = new_max
                    else:
                        unflip(i, old_sign, ks, delta)
                else:
                    prob = math.exp(-delta_obj / T)
                    if rng.random() < prob:
                        current_max = new_max
                    else:
                        unflip(i, old_sign, ks, delta)
            
            T *= cooling
    
    # Phase 3: Final greedy polish on best
    s = best_s.copy()
    ap_sums = np.zeros(M, dtype=np.int32)
    for k in range(M):
        ap_sums[k] = np.sum(s[ap_indices[k]])
    current_max = best_max
    
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for i in range(n):
            old_sign, ks, delta = flip(i)
            new_max = get_max()
            if new_max < current_max:
                current_max = new_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_s = s.copy()
            else:
                unflip(i, old_sign, ks, delta)

    return [int(x) for x in best_s]
