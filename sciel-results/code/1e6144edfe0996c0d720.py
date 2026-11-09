import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    n = 128
    aps = []
    # ap_indices: list of lists of AP indices for each position
    ap_indices = [[] for _ in range(n)]
    
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap = (a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1)
            idx = len(aps)
            aps.append(ap)
            for pos in ap:
                ap_indices[pos].append(idx)
                
    def is_bad_ap(ap, colors):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) < 3

    best_colors = None
    best_distinct = 129
    
    for k in range(3, 8):
        if time.time() - start_time > time_limit:
            break
            
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Initialize violation status
            is_bad = [is_bad_ap(ap, colors) for ap in aps]
            bad_count = sum(is_bad)
            
            steps = 50000
            for step in range(steps):
                if time.time() - start_time > time_limit:
                    break
                
                if bad_count == 0:
                    distinct_colors = len(set(colors))
                    if distinct_colors < best_distinct:
                        mapping = {}
                        new_colors = []
                        next_color = 0
                        for c in colors:
                            if c not in mapping:
                                mapping[c] = next_color
                                next_color += 1
                            new_colors.append(mapping[c])
                        best_colors = new_colors
                        best_distinct = distinct_colors
                    break
                
                # Pick a random bad AP
                # We need a list of bad AP indices
                # To avoid scanning all APs, we can maintain a list of bad AP indices
                # But updating it is tricky.
                # Alternatively, just pick a random position, and if it's in a bad AP, fix it.
                # Or pick a random AP, if it's bad, fix it.
                
                # Let's pick a random AP index
                ap_idx = random.randint(0, len(aps) - 1)
                if not is_bad[ap_idx]:
                    continue
                    
                ap = aps[ap_idx]
                # Pick a random position in the AP
                pos = random.choice(ap)
                old_color = colors[pos]
                new_color = random.randint(0, k - 1)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                # Update violations for APs containing pos
                for idx in ap_indices[pos]:
                    old_bad = is_bad[idx]
                    new_bad = is_bad_ap(aps[idx], colors)
                    if old_bad != new_bad:
                        is_bad[idx] = new_bad
                        bad_count += 1 if new_bad else -1
                        
            if best_distinct == 3:
                return best_colors

    if best_colors and all(not is_bad_ap(ap, best_colors) for ap in aps):
        return best_colors
        
    # Fallback
    if time.time() - start_time < time_limit - 2:
        colors=[]
        for x in range(1,129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color);break
        return colors
    else:
        return [0]*128
