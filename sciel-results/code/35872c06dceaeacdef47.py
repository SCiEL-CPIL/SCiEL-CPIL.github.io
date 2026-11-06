import time
import random

def construct():
    N = 512
    deadline = time.time() + 55
    
    # Precompute all 4-APs
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    num_to_aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(aps):
        for x in ap:
            num_to_aps[x].append(idx)
            
    # Greedy initialization
    selected = set()
    for x in range(1, N + 1):
        can_add = True
        for d in range(1, (x - 1) // 3 + 1):
            if x - 3 * d in selected and x - 2 * d in selected and x - d in selected:
                can_add = False
                break
        if can_add:
            selected.add(x)
            
    unselected = [x for x in range(1, N + 1) if x not in selected]
    
    # Track ap_counts
    ap_counts = [0] * len(aps)
    for x in selected:
        for idx in num_to_aps[x]:
            ap_counts[idx] += 1
            
    best_size = len(selected)
    best_set = set(selected)
    
    # Local Search
    while time.time() < deadline:
        # Check for violations
        violated = [i for i, c in enumerate(ap_counts) if c == 4]
        
        if not violated:
            if not unselected:
                break
            tried = 0
            added = False
            while tried < 5 and not added:
                tried += 1
                # Pick random index
                idx_u = random.randint(0, len(unselected) - 1)
                x = unselected[idx_u]
                ok = True
                for idx in num_to_aps[x]:
                    if ap_counts[idx] == 3:
                        ok = False
                        break
                if ok:
                    # Add x
                    selected.add(x)
                    # Remove x from unselected (swap with last)
                    unselected[idx_u] = unselected[-1]
                    unselected.pop()
                    for idx in num_to_aps[x]:
                        ap_counts[idx] += 1
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_set = set(selected)
                    added = True
            
            if not added:
                if selected:
                    x = random.choice(list(selected))
                    selected.remove(x)
                    unselected.append(x)
                    for idx in num_to_aps[x]:
                        ap_counts[idx] -= 1
        else:
            idx = random.choice(violated)
            ap = aps[idx]
            x = random.choice(ap)
            selected.remove(x)
            unselected.append(x)
            for idx2 in num_to_aps[x]:
                ap_counts[idx2] -= 1
            
            if unselected:
                tried = 0
                while tried < 5:
                    tried += 1
                    idx_u = random.randint(0, len(unselected) - 1)
                    y = unselected[idx_u]
                    ok = True
                    for idx2 in num_to_aps[y]:
                        if ap_counts[idx2] == 3:
                            ok = False
                            break
                    if ok:
                        selected.add(y)
                        unselected[idx_u] = unselected[-1]
                        unselected.pop()
                        for idx2 in num_to_aps[y]:
                            ap_counts[idx2] += 1
                        if len(selected) > best_size:
                            best_size = len(selected)
                            best_set = set(selected)
                        break

    return sorted(best_set)
