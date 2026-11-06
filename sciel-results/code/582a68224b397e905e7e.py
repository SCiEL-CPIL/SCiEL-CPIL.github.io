import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    
    # Precompute all APs and position-to-AP mapping
    aps = []
    pos_to_aps = [[] for _ in range(n)]
    
    for s in range(n):
        for d in range(1, n - s + 1):
            positions = []
            pos = s
            while pos < n:
                positions.append(pos)
                pos += d
            if positions:
                ap_idx = len(aps)
                aps.append(positions)
                for p in positions:
                    pos_to_aps[p].append(ap_idx)
    
    num_aps = len(aps)
    
    # Precompute affected APs per position as numpy arrays
    pos_to_aps_arr = [np.array(pos_to_aps[i], dtype=np.int32) for i in range(n)]
    
    sum_offset = n
    sum_range = 2 * n + 1
    
    def get_max_abs(sum_hist):
        for k in range(n, 0, -1):
            if sum_hist[k + sum_offset] > 0 or sum_hist[-k + sum_offset] > 0:
                return k
        return 0
    
    def run_local_search(signs, ap_sums, sum_hist, current_max, rng, time_limit):
        best_signs = signs.copy()
        best_max = current_max
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            order = list(range(n))
            rng.shuffle(order)
            
            for i in order:
                if time.time() >= time_limit:
                    break
                
                old_sign = int(signs[i])
                new_sign = -old_sign
                delta = new_sign - old_sign
                
                affected = pos_to_aps_arr[i]
                old_sums = ap_sums[affected].copy()
                
                ap_sums[affected] += delta
                sum_hist[old_sums + sum_offset] -= 1
                sum_hist[ap_sums[affected] + sum_offset] += 1
                
                new_max = get_max_abs(sum_hist)
                
                if new_max < current_max:
                    current_max = new_max
                    signs[i] = new_sign
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                else:
                    ap_sums[affected] = old_sums
                    sum_hist[old_sums + sum_offset] += 1
                    sum_hist[ap_sums[affected] + sum_offset] -= 1
        
        return best_signs, best_max
    
    best_overall = None
    best_overall_max = float('inf')
    
    rng_master = random.Random(999)
    
    while time.time() < deadline - 10.0:
        rng = random.Random(rng_master.randint(0, 10**9))
        signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)
        
        ap_sums = np.zeros(num_aps, dtype=np.int32)
        for ap_idx, positions in enumerate(aps):
            ap_sums[ap_idx] = int(np.sum(signs[positions]))
        
        sum_hist = np.zeros(sum_range, dtype=np.int32)
        for s in ap_sums:
            sum_hist[s + sum_offset] += 1
        
        current_max = get_max_abs(sum_hist)
        
        remaining = deadline - 10.0 - time.time()
        if remaining < 5.0:
            break
        
        best_signs, best_max = run_local_search(signs, ap_sums, sum_hist, current_max, rng, time.time() + remaining)
        
        if best_max < best_overall_max:
            best_overall_max = best_max
            best_overall = best_signs.copy()
    
    if best_overall is None:
        rng = random.Random(42)
        return [rng.choice((-1, 1)) for _ in range(n)]
    
    return [int(x) for x in best_overall]
