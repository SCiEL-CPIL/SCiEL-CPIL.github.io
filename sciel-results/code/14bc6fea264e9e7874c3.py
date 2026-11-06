"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    
    # Precompute all 4-APs and 3-APs
    four_aps = []
    for d in range(1, 86):
        for a in range(1, n - 3*d + 2):
            four_aps.append((a, a+d, a+2*d, a+3*d))
    
    three_aps = []
    for d in range(1, 129):
        for a in range(1, n - 2*d + 2):
            three_aps.append((a, a+d, a+2*d))
    
    # Build index: for each element, which 4-APs and 3-APs it belongs to
    elem_to_4ap = {x: [] for x in range(1, n+1)}
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4ap[x].append(i)
    
    elem_to_3ap = {x: [] for x in range(1, n+1)}
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3ap[x].append(i)
    
    # Initialize with greedy 4-AP-free set
    selected = set()
    for x in range(1, n+1):
        if all(not all(x-j*d in selected for j in (1,2,3)) for d in range(1, (x-1)//3+1)):
            selected.add(x)
    
    # Track which 4-APs are violated and which 3-APs are complete
    four_ap_violated = [False] * len(four_aps)
    three_ap_complete = [False] * len(three_aps)
    
    four_viol_count = 0
    for i, ap in enumerate(four_aps):
        if all(x in selected for x in ap):
            four_ap_violated[i] = True
            four_viol_count += 1
    
    three_count = 0
    for i, ap in enumerate(three_aps):
        if all(x in selected for x in ap):
            three_ap_complete[i] = True
            three_count += 1
    
    best_set = set(selected)
    best_three = three_count
    
    def update_after_add(x):
        """Add x to selected, update counts. Returns (delta_three, delta_viol)."""
        nonlocal four_viol_count, three_count
        old_viol = four_viol_count
        old_three = three_count
        for i in elem_to_4ap[x]:
            if not four_ap_violated[i] and all(v in selected for v in four_aps[i]):
                four_ap_violated[i] = True
                four_viol_count += 1
        for i in elem_to_3ap[x]:
            if not three_ap_complete[i] and all(v in selected for v in three_aps[i]):
                three_ap_complete[i] = True
                three_count += 1
        return three_count - old_three, four_viol_count - old_viol
    
    def update_after_remove(x):
        """Remove x from selected, update counts. Returns (delta_three, delta_viol)."""
        nonlocal four_viol_count, three_count
        old_viol = four_viol_count
        old_three = three_count
        for i in elem_to_4ap[x]:
            if four_ap_violated[i]:
                if not all(v in selected for v in four_aps[i]):
                    four_ap_violated[i] = False
                    four_viol_count -= 1
        for i in elem_to_3ap[x]:
            if three_ap_complete[i]:
                if not all(v in selected for v in three_aps[i]):
                    three_ap_complete[i] = False
                    three_count -= 1
        return three_count - old_three, four_viol_count - old_viol
    
    temp = 50.0
    cooling = 0.9995
    min_temp = 0.1
    step = 0
    
    while time.time() < deadline and temp > min_temp:
        step += 1
        move_type = random.random()
        
        if move_type < 0.35:
            # Add an element
            candidates = [x for x in range(1, n+1) if x not in selected]
            if not candidates:
                continue
            x = random.choice(candidates)
            selected.add(x)
            dt, dv = update_after_add(x)
            delta_score = dt - 1000 * dv
            if delta_score >= 0 or random.random() < min(1.0, temp / max(1, abs(delta_score))):
                if four_viol_count == 0 and three_count > best_three:
                    best_set = set(selected)
                    best_three = three_count
            else:
                selected.discard(x)
                update_after_remove(x)
        elif move_type < 0.65:
            # Remove an element
            if not selected:
                continue
            x = random.choice(list(selected))
            selected.discard(x)
            dt, dv = update_after_remove(x)
            delta_score = dt - 1000 * dv
            if delta_score >= 0 or random.random() < min(1.0, temp / max(1, abs(delta_score))):
                if four_viol_count == 0 and three_count > best_three:
                    best_set = set(selected)
                    best_three = three_count
            else:
                selected.add(x)
                update_after_add(x)
        else:
            # Swap
            if not selected:
                continue
            x_remove = random.choice(list(selected))
            candidates = [x for x in range(1, n+1) if x not in selected]
            if not candidates:
                continue
            x_add = random.choice(candidates)
            if x_add == x_remove:
                continue
            selected.discard(x_remove)
            selected.add(x_add)
            dt1, dv1 = update_after_remove(x_remove)
            dt2, dv2 = update_after_add(x_add)
            dt = dt1 + dt2
            dv = dv1 + dv2
            delta_score = dt - 1000 * dv
            if delta_score >= 0 or random.random() < min(1.0, temp / max(1, abs(delta_score))):
                if four_viol_count == 0 and three_count > best_three:
                    best_set = set(selected)
                    best_three = three_count
            else:
                # Reject: revert
                selected.discard(x_add)
                selected.add(x_remove)
                update_after_remove(x_add)
                update_after_add(x_remove)
        
        temp *= cooling
        
        # Periodic greedy repair
        if four_viol_count > 0 and step % 100 == 0:
            for i in range(len(four_aps)):
                if four_ap_violated[i]:
                    x = random.choice(four_aps[i])
                    selected.discard(x)
                    update_after_remove(x)
    
    if not best_set:
        return sorted(selected) if selected else [1]
    return sorted(best_set)
# EVOLVE-BLOCK-END
