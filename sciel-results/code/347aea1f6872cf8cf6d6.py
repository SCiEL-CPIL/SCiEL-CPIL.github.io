"""Colorings with three colors in every four-term progression."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    
    # Precompute all 4-term arithmetic progressions
    taps_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            taps_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    taps_arr = np.array(taps_list, dtype=np.int32)
    num_taps = len(taps_arr)
    
    def count_violations(colors):
        c0 = colors[taps_arr[:, 0]]
        c1 = colors[taps_arr[:, 1]]
        c2 = colors[taps_arr[:, 2]]
        c3 = colors[taps_arr[:, 3]]
        has3 = ((c0 != c1) & (c0 != c2) & (c1 != c2)) | \
               ((c0 != c1) & (c0 != c3) & (c1 != c3)) | \
               ((c0 != c2) & (c0 != c3) & (c2 != c3)) | \
               ((c1 != c2) & (c1 != c3) & (c2 != c3))
        return int(np.sum(~has3))
    
    best_obj = float('inf')
    best_colors = None
    
    rng = random.Random(42)
    
    for max_k in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        
        for init in range(5):
            if time.time() > deadline:
                break
            
            colors = np.array([rng.randint(0, max_k - 1) for _ in range(n)], dtype=np.int32)
            
            cur_viol = count_violations(colors)
            cur_dc = len(set(colors.tolist()))
            cur_obj = cur_viol * 1000 + cur_dc
            
            current_temp = 50.0
            min_temp = 0.01
            cooling = 0.985
            
            for it in range(4000):
                if time.time() > deadline:
                    break
                
                pos = rng.randint(0, n - 1)
                old_color = colors[pos]
                new_color = rng.randint(0, max_k - 1)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                new_viol = count_violations(colors)
                new_dc = len(set(colors.tolist()))
                delta = (new_viol * 1000 + new_dc) - cur_obj
                
                if delta < 0 or (current_temp > min_temp and rng.random() < np.exp(-delta / current_temp)):
                    cur_viol = new_viol
                    cur_dc = new_dc
                    cur_obj = cur_viol * 1000 + cur_dc
                else:
                    colors[pos] = old_color
                
                current_temp *= cooling
                
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_colors = colors.copy()
                
                if cur_viol == 0:
                    break
            
            if cur_viol == 0:
                for _ in range(500):
                    if time.time() > deadline:
                        break
                    used = set(colors.tolist())
                    if len(used) <= 3:
                        break
                    target = rng.choice(list(used))
                    positions = np.where(colors == target)[0]
                    if len(positions) == 0:
                        continue
                    pos = rng.choice(positions.tolist())
                    new_c = rng.randint(0, max_k - 1)
                    if new_c == colors[pos]:
                        continue
                    colors[pos] = new_c
                    new_viol = count_violations(colors)
                    if new_viol == 0 and len(set(colors.tolist())) < cur_dc:
                        cur_dc = len(set(colors.tolist()))
                        cur_obj = cur_viol * 1000 + cur_dc
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best_colors = colors.copy()
                    else:
                        colors[pos] = target
    
    if best_colors is not None:
        return best_colors.tolist()
    return list(range(128))
