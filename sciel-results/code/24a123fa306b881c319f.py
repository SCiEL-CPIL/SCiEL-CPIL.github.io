"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for i in range(n):
        for d in range(1, (n - i) // 3 + 1):
            i3 = i + 3 * d
            if i3 < n:
                aps.append((i, i + d, i + 2 * d, i3))
    
    num_aps = len(aps)
    
    # For each position, which AP indices contain it?
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)
    
    # Track violation status for each AP
    # ap_violation[idx] = 1 if AP idx is a violation, 0 otherwise
    # total_violations = sum of ap_violation
    
    def is_ap_violation(colors, ap_idx):
        ap = aps[ap_idx]
        return len(set(colors[p] for p in ap)) < 3
    
    best_colors = None
    best_num_colors = 128
    
    for num_colors in [3, 4, 5, 6, 8, 10, 12, 16]:
        if time.time() > deadline - 5:
            break
        
        max_attempts = 50
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
            
            colors = [random.randint(0, num_colors - 1) for _ in range(n)]
            
            # Initialize violation tracking
            ap_violation = [is_ap_violation(colors, i) for i in range(num_aps)]
            total_viol = sum(ap_violation)
            
            if total_viol == 0:
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_colors = colors[:]
                break
            
            # Simulated annealing with efficient delta computation
            temp = 1.0
            cooling = 0.995
            min_temp = 0.001
            
            for step in range(20000):
                if time.time() > deadline - 2:
                    break
                
                if total_viol == 0:
                    if num_colors < best_num_colors:
                        best_num_colors = num_colors
                        best_colors = colors[:]
                    break
                
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                new_color = random.randint(0, num_colors - 1)
                
                if old_color == new_color:
                    continue
                
                # Compute delta by checking only APs containing pos
                delta = 0
                for ap_idx in pos_aps[pos]:
                    ap = aps[ap_idx]
                    # Was it a violation?
                    was_viol = ap_violation[ap_idx]
                    
                    # Check after hypothetical change
                    colors[pos] = new_color
                    new_viol = len(set(colors[p] for p in ap)) < 3
                    colors[pos] = old_color
                    
                    if was_viol and not new_viol:
                        delta -= 1
                    elif not was_viol and new_viol:
                        delta += 1
                
                if delta <= 0 or random.random() < pow(temp, delta):
                    colors[pos] = new_color
                    # Update violation tracking
                    for ap_idx in pos_aps[pos]:
                        ap = aps[ap_idx]
                        new_viol = len(set(colors[p] for p in ap)) < 3
                        if ap_violation[ap_idx] != new_viol:
                            total_viol += 1 if new_viol else -1
                            ap_violation[ap_idx] = new_viol
                
                temp = max(temp * cooling, min_temp)
            
            if total_viol == 0:
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_colors = colors[:]
    
    if best_colors is not None:
        return best_colors
    
    # Fallback
    return list(range(128))
# EVOLVE-BLOCK-END
