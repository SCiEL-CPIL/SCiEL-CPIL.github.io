import random
import numpy as np

def construct():
    n = 128
    
    # Precompute all 4-APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap = (a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1)
            aps.append(ap)
            
    index_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for idx in ap:
            index_to_aps[idx].append(i)

    def is_bad(ap, colors):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) < 3

    def solve_sa(k, max_iters=100000, seed=42):
        random.seed(seed)
        colors = [random.randint(0, k - 1) for _ in range(n)]
        
        ap_bad = [is_bad(ap, colors) for ap in aps]
        total_bad = sum(ap_bad)
        
        if total_bad == 0:
            return colors
        
        best_colors = colors[:]
        best_total_bad = total_bad
        
        temp = 1.0
        cooling = 0.9995
        
        for _ in range(max_iters):
            i = random.randint(0, n - 1)
            old_color = colors[i]
            new_color = random.randint(0, k - 1)
            
            if new_color == old_color:
                temp *= cooling
                continue
                
            colors[i] = new_color
            
            # Calculate delta
            delta = 0
            affected = index_to_aps[i]
            for ap_idx in affected:
                was_bad = ap_bad[ap_idx]
                is_now_bad = is_bad(aps[ap_idx], colors)
                if was_bad and not is_now_bad:
                    delta -= 1
                elif not was_bad and is_now_bad:
                    delta += 1
                ap_bad[ap_idx] = is_now_bad
            
            new_total_bad = total_bad + delta
            
            accept = False
            if new_total_bad < total_bad:
                accept = True
            elif delta > 0:
                p = np.exp(-delta / temp)
                if random.random() < p:
                    accept = True
            
            if accept:
                total_bad = new_total_bad
                if total_bad == 0:
                    return colors
                if total_bad < best_total_bad:
                    best_total_bad = total_bad
                    best_colors = colors[:]
            else:
                # Revert
                colors[i] = old_color
                # Recalculate bad status for affected APs
                for ap_idx in affected:
                    ap_bad[ap_idx] = is_bad(aps[ap_idx], colors)
            
            temp *= cooling
            
        return best_colors if best_total_bad == 0 else None

    for k in range(3, 8):
        sol = solve_sa(k, max_iters=50000)
        if sol is not None:
            return sol
            
    # Fallback
    return [0] * n
