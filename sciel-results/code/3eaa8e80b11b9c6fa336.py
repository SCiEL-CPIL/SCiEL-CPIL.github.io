import random
import time

def construct():
    n = 128
    random.seed(42)
    start_time = time.time()
    
    # Precompute all 4-term APs (0-indexed)
    aps = []
    for a in range(n):
        for d in range(1, (n - a) // 3):
            if a + 3 * d < n:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Initial coloring: mod 3
    colors = [i % 3 for i in range(n)]
    
    def count_violations():
        v = 0
        for ap in aps:
            c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(c) < 3:
                v += 1
        return v
    
    cur_v = count_violations()
    
    # Local search
    while time.time() - start_time < 55:
        if cur_v == 0:
            break
        
        # Find a random violating AP
        viol_ap = None
        for _ in range(100):
            vi = random.randint(0, num_aps - 1)
            ap = aps[vi]
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                viol_ap = ap
                break
        
        if viol_ap is None:
            break
        
        # Try to fix by changing one position in the AP
        improved = False
        for pos_idx in range(4):
            pos = viol_ap[pos_idx]
            old_c = colors[pos]
            for new_c in range(3):
                if new_c == old_c:
                    continue
                colors[pos] = new_c
                new_v = count_violations()
                if new_v < cur_v:
                    cur_v = new_v
                    improved = True
                    break
                colors[pos] = old_c
            if improved:
                break
    
    return colors
