import random
import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    
    # Precompute all 4-APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    ap_a = np.array([ap[0] for ap in aps], dtype=np.int32)
    ap_b = np.array([ap[1] for ap in aps], dtype=np.int32)
    ap_c = np.array([ap[2] for ap in aps], dtype=np.int32)
    ap_d = np.array([ap[3] for ap in aps], dtype=np.int32)
    
    def count_violations_fast(colors):
        c1 = colors[ap_a]
        c2 = colors[ap_b]
        c3 = colors[ap_c]
        c4 = colors[ap_d]
        
        all_eq = (c1 == c2) & (c2 == c3) & (c3 == c4)
        
        t31_1 = (c1 == c2) & (c2 == c3) & (c3 != c4)
        t31_2 = (c1 == c2) & (c2 == c4) & (c2 != c3)
        t31_3 = (c1 == c3) & (c3 == c4) & (c1 != c2)
        t31_4 = (c2 == c3) & (c3 == c4) & (c2 != c1)
        
        t22_1 = (c1 == c2) & (c3 == c4) & (c1 != c3)
        t22_2 = (c1 == c3) & (c2 == c4) & (c1 != c2)
        t22_3 = (c1 == c4) & (c2 == c3) & (c1 != c2)
        
        two_distinct = t31_1 | t31_2 | t31_3 | t31_4 | t22_1 | t22_2 | t22_3
        
        invalid = all_eq | two_distinct
        return int(np.sum(invalid))
    
    best_colors = None
    best_distinct = n + 1
    
    def reduce_colors(colors, deadline):
        colors = colors.copy()
        changed = True
        while changed and time.time() < deadline:
            changed = False
            distinct = sorted(set(colors.tolist()))
            for i in range(len(distinct)):
                if time.time() > deadline:
                    return colors, len(set(colors.tolist()))
                for j in range(i + 1, len(distinct)):
                    if time.time() > deadline:
                        return colors, len(set(colors.tolist()))
                    c1 = distinct[i]
                    c2 = distinct[j]
                    merged = colors.copy()
                    merged[merged == c2] = c1
                    viol = count_violations_fast(merged)
                    if viol == 0:
                        colors = merged
                        changed = True
                        break
                if changed:
                    break
        return colors, len(set(colors.tolist()))
    
    # Allocate time: more for smaller color counts
    # 3 colors: 20s, 4: 15s, 5: 12s, 6: 10s, 7+: 5s each
    time_alloc = {
        3: 20.0, 4: 15.0, 5: 12.0, 6: 10.0, 7: 8.0, 8: 6.0,
        9: 5.0, 10: 5.0, 11: 5.0, 12: 5.0, 13: 5.0, 14: 5.0
    }
    
    for max_colors in range(3, 15):
        if time.time() > deadline:
            break
        
        alloc = time_alloc.get(max_colors, 5.0)
        sub_deadline = min(deadline, time.time() + alloc)
        
        if max_colors == 3:
            num_restarts = 30
            iters_per_restart = 15000
        elif max_colors == 4:
            num_restarts = 25
            iters_per_restart = 12000
        else:
            num_restarts = 15
            iters_per_restart = 10000
        
        for restart in range(num_restarts):
            if time.time() > sub_deadline:
                break
            
            # Initialize
            if restart % 4 == 0:
                colors = np.zeros(n, dtype=np.int32)
                for i in range(n):
                    colors[i] = i % max_colors
            elif restart % 4 == 1:
                colors = np.random.randint(0, max_colors, size=n, dtype=np.int32)
            elif restart % 4 == 2:
                # Two-periodic
                colors = np.zeros(n, dtype=np.int32)
                for i in range(n):
                    colors[i] = (i * 7) % max_colors
            else:
                colors = np.random.randint(0, max_colors, size=n, dtype=np.int32)
            
            current_viol = count_violations_fast(colors)
            if current_viol == 0:
                colors, distinct = reduce_colors(colors, sub_deadline)
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = colors.copy()
                    print(f"Valid from init: {best_distinct} colors (max={max_colors})", file=sys.stderr)
                    if best_distinct == 3:
                        return best_colors.tolist()
                continue
            
            # Simulated annealing
            T = 1.0
            cooling = 0.998
            best_local = colors.copy()
            best_local_viol = current_viol
            
            for it in range(iters_per_restart):
                if time.time() > sub_deadline:
                    break
                
                idx = random.randint(0, n - 1)
                old_color = colors[idx]
                new_color = random.randint(0, max_colors - 1)
                if new_color == old_color:
                    continue
                
                colors[idx] = new_color
                new_viol = count_violations_fast(colors)
                delta = new_viol - current_viol
                
                if delta <= 0 or (T > 0.01 and random.random() < np.exp(-delta / T)):
                    current_viol = new_viol
                    if current_viol < best_local_viol:
                        best_local = colors.copy()
                        best_local_viol = current_viol
                else:
                    colors[idx] = old_color
                
                T *= cooling
            
            if best_local_viol == 0:
                colors, distinct = reduce_colors(best_local, sub_deadline)
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = colors.copy()
                    print(f"Found valid: {best_distinct} colors (max={max_colors}, r={restart})", file=sys.stderr)
                    if best_distinct == 3:
                        return best_colors.tolist()
    
    if best_colors is not None:
        return best_colors.tolist()
    
    # Fallback greedy
    colors = []
    for x in range(128):
        for color in range(128):
            valid = True
            for d in range(1, (x + 1) // 3 + 1):
                if x - 3 * d >= 0:
                    c1 = colors[x - 3 * d]
                    c2 = colors[x - 2 * d]
                    c3 = colors[x - d]
                    if len({c1, c2, c3, color}) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
    return colors
