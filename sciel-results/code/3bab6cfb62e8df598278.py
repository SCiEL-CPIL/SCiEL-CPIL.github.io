import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    
    # Precompute APs
    aps = []
    # ap_to_pos: for each position, list of AP indices involving it
    pos_to_aps = [[] for _ in range(n)]
    
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            idx1 = a - 1
            idx2 = a + d - 1
            idx3 = a + 2 * d - 1
            idx4 = a + 3 * d - 1
            ap_idx = len(aps)
            aps.append((idx1, idx2, idx3, idx4))
            pos_to_aps[idx1].append(ap_idx)
            pos_to_aps[idx2].append(ap_idx)
            pos_to_aps[idx3].append(ap_idx)
            pos_to_aps[idx4].append(ap_idx)
            
    def is_ap_valid(colors, ap_idx):
        i1, i2, i3, i4 = aps[ap_idx]
        c1 = colors[i1]
        c2 = colors[i2]
        c3 = colors[i3]
        c4 = colors[i4]
        return len({c1, c2, c3, c4}) >= 3

    def count_violations(colors):
        v = 0
        for i in range(len(aps)):
            if not is_ap_valid(colors, i):
                v += 1
        return v

    def local_search(colors, k, max_iter=10000):
        # Initial violation count
        viol_count = count_violations(colors)
        if viol_count == 0:
            return colors, True
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return colors, False
            
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Get APs affected by pos
            affected_aps = pos_to_aps[pos]
            
            # Count violations in affected APs with old color
            old_viol = 0
            for ap_idx in affected_aps:
                if not is_ap_valid(colors, ap_idx):
                    old_viol += 1
            
            best_color = old_color
            best_new_viol = old_viol
            
            for new_color in range(k):
                if new_color == old_color:
                    continue
                colors[pos] = new_color
                new_viol = 0
                for ap_idx in affected_aps:
                    if not is_ap_valid(colors, ap_idx):
                        new_viol += 1
                
                if new_viol < best_new_viol:
                    best_new_viol = new_viol
                    best_color = new_color
                    if new_viol == 0:
                        break
                # Note: total violations change by (new_viol - old_viol)
                # We just want to minimize total, so minimizing local violations is good.
                
            colors[pos] = best_color
            
            # Update global violation count
            viol_count += best_new_viol - old_viol
            
            if viol_count == 0:
                return colors, True
                
        return colors, False

    # Try k=3
    for attempt in range(500):
        if time.time() > deadline:
            break
        colors = [random.randint(0, 2) for _ in range(n)]
        colors, valid = local_search(colors, 3, max_iter=5000)
        if valid:
            return colors

    # Try k=4
    for attempt in range(500):
        if time.time() > deadline:
            break
        colors = [random.randint(0, 3) for _ in range(n)]
        colors, valid = local_search(colors, 4, max_iter=5000)
        if valid:
            return colors

    # Try k=5
    for attempt in range(500):
        if time.time() > deadline:
            break
        colors = [random.randint(0, 4) for _ in range(n)]
        colors, valid = local_search(colors, 5, max_iter=5000)
        if valid:
            return colors

    # Fallback
    colors = []
    for x in range(1, n + 1):
        found = False
        for c in range(8):
            is_valid = True
            for d in range(1, (x-1)//3 + 1):
                a = x - 3*d
                if a < 1: continue
                c1 = colors[a-1]
                c2 = colors[a+d-1]
                c3 = colors[a+2*d-1]
                if len({c, c1, c2, c3}) < 3:
                    is_valid = False
                    break
            if is_valid:
                colors.append(c)
                found = True
                break
        if not found:
            colors.append(0)
            
    return colors
