import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 128
    
    aps = []
    ap_indices = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                ap_indices[idx].append(ap_idx)
                
    num_aps = len(aps)
    
    def is_violation(ap, colors):
        c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c1, c2, c3, c4}) < 3
    
    def run_seed_greedy():
        colors = []
        for x in range(N):
            for color in range(128):
                test_colors = colors + [color]
                valid = True
                for ai in ap_indices[x]:
                    ap = aps[ai]
                    if all(p < x + 1 for p in ap):
                        if is_violation(ap, test_colors):
                            valid = False
                            break
                if valid:
                    colors.append(color)
                    break
            else:
                colors.append(0)
        return colors
    
    seed_colors = run_seed_greedy()
    best_colors = seed_colors[:]
    best_k = len(set(seed_colors))
    
    for target_k in range(2, best_k):
        if time.time() > DEADLINE - 3:
            break
            
        k_start = time.time()
        k_budget = min(2.0, DEADLINE - 3 - k_start)
        if k_budget <= 0:
            break
        
        for restart in range(500):
            if time.time() > k_start + k_budget:
                break
            
            colors = [random.randint(0, target_k - 1) for _ in range(N)]
            
            violation_flags = [False] * num_aps
            num_violations = 0
            for ai in range(num_aps):
                if is_violation(aps[ai], colors):
                    violation_flags[ai] = True
                    num_violations += 1
            
            max_steps = 2000
            step = 0
            while num_violations > 0 and step < max_steps:
                step += 1
                if time.time() > k_start + k_budget:
                    break
                
                violated = [i for i in range(num_aps) if violation_flags[i]]
                if not violated:
                    break
                ai = random.choice(violated)
                ap = aps[ai]
                
                positions = [0, 1, 2, 3]
                random.shuffle(positions)
                
                improved = False
                for pos in positions:
                    idx = ap[pos]
                    old_c = colors[idx]
                    
                    affected = ap_indices[idx]
                    old_flags = {aff_ai: violation_flags[aff_ai] for aff_ai in affected}
                    
                    for new_c in range(target_k):
                        if new_c == old_c:
                            continue
                        
                        colors[idx] = new_c
                        
                        new_num_violations = num_violations
                        for aff_ai in affected:
                            was_v = old_flags[aff_ai]
                            is_v = is_violation(aps[aff_ai], colors)
                            if was_v and not is_v:
                                new_num_violations -= 1
                            elif not was_v and is_v:
                                new_num_violations += 1
                            violation_flags[aff_ai] = is_v
                        
                        if new_num_violations < num_violations:
                            num_violations = new_num_violations
                            improved = True
                            break
                        else:
                            colors[idx] = old_c
                            for aff_ai in affected:
                                violation_flags[aff_ai] = old_flags[aff_ai]
                    
                    if improved:
                        break
                
                if not improved:
                    idx = random.randint(0, N - 1)
                    new_c = random.randint(0, target_k - 1)
                    if new_c != colors[idx]:
                        colors[idx] = new_c
                        num_violations = 0
                        for ai2 in range(num_aps):
                            if is_violation(aps[ai2], colors):
                                violation_flags[ai2] = True
                                num_violations += 1
                            else:
                                violation_flags[ai2] = False
                    if step % 100 == 0:
                        break
            
            if num_violations == 0:
                k_used = len(set(colors))
                if k_used < best_k:
                    best_colors = colors[:]
                    best_k = k_used
                break
    
    # Final check
    if time.time() < DEADLINE - 1:
        final_violations = sum(1 for ai in range(num_aps) if is_violation(aps[ai], best_colors))
        if final_violations > 0:
            best_colors = seed_colors
    
    return best_colors
