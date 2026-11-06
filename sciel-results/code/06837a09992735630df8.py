import random
import time

def construct():
    n = 128
    # Precompute all 4-APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map index to list of AP indices it belongs to
    ap_idx_map = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_idx_map[pos].append(idx)

    def is_violation(c0, c1, c2, c3):
        # Returns True if distinct count < 3
        # 1 distinct
        if c0 == c1 and c0 == c2 and c0 == c3:
            return True
        # 2 distinct
        # 3 same, 1 diff
        if c0 == c1 and c0 == c2 and c0 != c3: return True
        if c0 == c1 and c0 != c2 and c0 == c3: return True
        if c0 == c2 and c0 != c1 and c0 == c3: return True
        if c1 == c2 and c1 != c0 and c1 == c3: return True
        if c1 == c3 and c1 != c0 and c1 == c2: return True
        if c2 == c3 and c2 != c0 and c2 == c1: return True
        # 2 pairs
        if c0 == c1 and c2 == c3 and c0 != c2: return True
        if c0 == c2 and c1 == c3 and c0 != c1: return True
        if c0 == c3 and c1 == c2 and c0 != c1: return True
        return False

    def count_violations(colors):
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if is_violation(c0, c1, c2, c3):
                v += 1
        return v

    def delta_violations(colors, idx, new_color):
        old_color = colors[idx]
        if old_color == new_color:
            return 0
        
        delta = 0
        for ap_idx in ap_idx_map[idx]:
            ap = aps[ap_idx]
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            
            old_viol = 1 if is_violation(c0, c1, c2, c3) else 0
            
            new_c0, new_c1, new_c2, new_c3 = c0, c1, c2, c3
            if ap[0] == idx: new_c0 = new_color
            elif ap[1] == idx: new_c1 = new_color
            elif ap[2] == idx: new_c2 = new_color
            elif ap[3] == idx: new_c3 = new_color
            
            new_viol = 1 if is_violation(new_c0, new_c1, new_c2, new_c3) else 0
            
            delta += (new_viol - old_viol)
            
        return delta

    random.seed(42)
    best_colors = None
    best_viol = float('inf')
    
    start_time = time.time()
    timeout = 50.0 # Safe margin below 60s
    
    while time.time() - start_time < timeout:
        colors = [random.randint(0, 2) for _ in range(n)]
        current_viol = count_violations(colors)
        
        if current_viol < best_viol:
            best_viol = current_viol
            best_colors = colors[:]
            if best_viol == 0:
                return best_colors
        
        T = 5.0
        cooling = 0.995
        steps = 0
        max_steps = 50000 # Increased steps since each step is faster
        
        while steps < max_steps:
            if time.time() - start_time > timeout:
                break
            
            idx = random.randint(0, n - 1)
            old_color = colors[idx]
            
            new_color = random.randint(0, 2)
            if new_color == old_color:
                steps += 1
                continue
                
            d_viol = delta_violations(colors, idx, new_color)
            
            if d_viol < 0:
                colors[idx] = new_color
                current_viol += d_viol
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_colors = colors[:]
                    if best_viol == 0:
                        return best_colors
            elif d_viol == 0:
                if random.random() < 0.8:
                    colors[idx] = new_color
            else:
                import math
                prob = math.exp(-d_viol / T)
                if random.random() < prob:
                    colors[idx] = new_color
                    current_viol += d_viol
            
            T *= cooling
            steps += 1

    return best_colors
