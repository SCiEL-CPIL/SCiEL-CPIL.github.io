import time
import random
import math
import numpy as np

def construct():
    n = 128
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute all nonconstant 4-APs (0-indexed)
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    num_aps = len(aps)

    # For each position, precompute which APs it participates in
    pos_aps = [[] for _ in range(n)]
    for idx, (i, j, k, l) in enumerate(aps):
        pos_aps[i].append(idx)
        pos_aps[j].append(idx)
        pos_aps[k].append(idx)
        pos_aps[l].append(idx)

    def ap_violates(colors, idx):
        i, j, k, l = aps[idx]
        c = {colors[i], colors[j], colors[k], colors[l]}
        return len(c) < 3

    def count_violations(colors):
        return sum(1 for idx in range(num_aps) if ap_violates(colors, idx))

    def get_violation_set(colors):
        vs = set()
        for idx in range(num_aps):
            if ap_violates(colors, idx):
                vs.add(idx)
        return vs

    def try_merge_colors(colors, deadline):
        """Iteratively try to merge color pairs while maintaining validity."""
        current = list(colors)
        distinct = set(current)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            color_list = sorted(distinct)
            for i in range(len(color_list)):
                if time.time() > deadline - 5:
                    return current
                for j in range(i + 1, len(color_list)):
                    c1, c2 = color_list[i], color_list[j]
                    test = list(current)
                    for idx in range(n):
                        if test[idx] == c2:
                            test[idx] = c1
                    if count_violations(test) == 0:
                        current = test
                        distinct = set(current)
                        improved = True
                        break
                if improved:
                    break
        
        return current

    def local_search(colors, max_k, deadline):
        """Local search with violation-biased moves and SA acceptance."""
        colors = list(colors)
        viol_set = get_violation_set(colors)
        cur_viol = len(viol_set)
        
        best_local = list(colors)
        best_viol = cur_viol
        no_improve = 0
        t = 0.5
        
        max_steps = 8000
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if cur_viol == 0:
                break
            
            # Pick position biased toward violations
            if random.random() < 0.75 and viol_set:
                viol_ap = random.choice(list(viol_set))
                i, j, k, l = aps[viol_ap]
                pos = random.choice([i, j, k, l])
            else:
                pos = random.randint(0, n - 1)
            
            old_color = colors[pos]
            new_color = random.randint(0, max_k - 1)
            
            if new_color == old_color:
                continue
            
            # Compute delta in violation count
            delta = 0
            for ap_idx in pos_aps[pos]:
                was_viol = ap_idx in viol_set
                colors[pos] = new_color
                is_viol = ap_violates(colors, ap_idx)
                colors[pos] = old_color
                if was_viol and not is_viol:
                    delta -= 1
                elif not was_viol and is_viol:
                    delta += 1
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.4
            else:
                if t > 0.01:
                    accept = random.random() < math.exp(-delta / t)
            
            if accept:
                colors[pos] = new_color
                for ap_idx in pos_aps[pos]:
                    was_viol = ap_idx in viol_set
                    is_viol = ap_violates(colors, ap_idx)
                    if was_viol and not is_viol:
                        viol_set.discard(ap_idx)
                    elif not was_viol and is_viol:
                        viol_set.add(ap_idx)
                cur_viol = len(viol_set)
                
                if cur_viol < best_viol:
                    best_viol = cur_viol
                    best_local = list(colors)
                    no_improve = 0
                else:
                    no_improve += 1
                
                t = max(0.005, t * 0.9995)
            else:
                no_improve += 1
            
            if no_improve > 1500:
                break
        
        if best_viol < cur_viol:
            return best_local, best_viol
        return colors, cur_viol

    best_colors = None
    best_distinct = float('inf')

    # Phase 1: Find valid colorings with various color counts
    for max_k in range(6, 16):
        if time.time() > deadline - 15:
            break
        
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            colors = [random.randint(0, max_k - 1) for _ in range(n)]
            colors, viol = local_search(colors, max_k, deadline)
            
            if viol == 0:
                merged = try_merge_colors(colors, deadline)
                distinct = len(set(merged))
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = merged
                    if best_distinct <= 4:
                        break
        
        if best_distinct <= 4:
            break

    # Fallback: modular coloring
    if best_colors is None:
        best_colors = [x % 43 for x in range(n)]

    return best_colors
