"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 128
    start_time = time.time()
    random.seed(42)
    
    # Precompute all 4-APs
    aps = []
    for a in range(n):
        for d in range(1, (n - a - 1)//3 + 1):
            e = a + 3*d
            if e < n:
                aps.append((a, a+d, a+2*d, e))
    
    # For each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(i)
    
    def ap_violation(colors, ap):
        a, b, c, e = ap
        return len(set([colors[a], colors[b], colors[c], colors[e]])) < 3
    
    def try_solve(num_colors, max_iters=1000000):
        colors = [random.randrange(num_colors) for _ in range(n)]
        ap_status = [ap_violation(colors, ap) for ap in aps]
        v = sum(ap_status)
        
        if v == 0:
            return colors
        
        for it in range(max_iters):
            if time.time() - start_time > 50:
                return None
            if v == 0:
                return colors
            
            pos = random.randrange(n)
            old_c = colors[pos]
            new_c = random.randrange(num_colors)
            if new_c == old_c:
                continue
            
            colors[pos] = new_c
            
            delta = 0
            for ap_idx in pos_to_aps[pos]:
                old_s = ap_status[ap_idx]
                new_s = ap_violation(colors, aps[ap_idx])
                ap_status[ap_idx] = new_s
                if new_s and not old_s:
                    delta += 1
                elif old_s and not new_s:
                    delta -= 1
            
            v += delta
            
            if delta > 0:
                colors[pos] = old_c
                for ap_idx in pos_to_aps[pos]:
                    ap_status[ap_idx] = ap_violation(colors, aps[ap_idx])
                v -= delta
        
        return None
    
    for nc in [3, 4, 5, 6, 7, 8, 9, 10]:
        result = try_solve(nc)
        if result is not None:
            return result
    
    # Fallback: greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x-1)//3 + 1):
                a = x - 3*d
                if a < 1:
                    continue
                ap = [a, a+d, a+2*d, a+3*d]
                if all(p <= x for p in ap):
                    ap_colors = [color if p == x else colors[p-1] for p in ap]
                    if len(set(ap_colors)) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                break
    return colors

# EVOLVE-BLOCK-END
