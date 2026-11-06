import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin
    
    N = 128
    
    # Pre-compute all 4-term APs
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
    
    def check_ap_valid(ap, colors):
        c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c1, c2, c3, c4}) >= 3

    def run_seed():
        colors=[]
        for x in range(1,129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color);break
        return colors

    seed_colors = run_seed()
    best_colors = seed_colors
    best_k = len(set(seed_colors))
    
    # Local Search for better k
    for target_k in range(3, best_k):
        if time.time() > DEADLINE - 2: # Keep 2s for finalization
            break
            
        k_start = time.time()
        k_budget = 4.0 # 4 seconds per k
        
        for restart in range(500):
            if time.time() > k_start + k_budget:
                break
                
            colors = [random.randint(0, target_k - 1) for _ in range(N)]
            
            # Initialize violation state
            violation_flags = [False] * num_aps
            num_violations = 0
            for ai, ap in enumerate(aps):
                if not check_ap_valid(ap, colors):
                    violation_flags[ai] = True
                    num_violations += 1
                    
            # Local search
            while num_violations > 0:
                if time.time() > k_start + k_budget:
                    break
                    
                # Pick a random violated AP
                violated_indices = [i for i, v in enumerate(violation_flags) if v]
                if not violated_indices:
                    break
                ai = random.choice(violated_indices)
                ap = aps[ai]
                
                # Pick a random element in the AP to change
                pos = random.randint(0, 3)
                idx = ap[pos]
                old_c = colors[idx]
                new_c = random.randint(0, target_k - 1)
                
                if new_c == old_c:
                    continue
                
                # Calculate delta violations
                old_num_violations = num_violations
                
                # Temporarily apply change
                colors[idx] = new_c
                
                # Update violations for affected APs
                for affected_ai in ap_indices[idx]:
                    was_violated = violation_flags[affected_ai]
                    is_violated = not check_ap_valid(aps[affected_ai], colors)
                    
                    if was_violated and not is_violated:
                        num_violations -= 1
                    elif not was_violated and is_violated:
                        num_violations += 1
                    violation_flags[affected_ai] = is_violated
                
                # If move did not reduce violations, revert
                if num_violations >= old_num_violations:
                    colors[idx] = old_c
                    # Revert violation flags
                    for affected_ai in ap_indices[idx]:
                        # Re-calculate or save old state?
                        # Since we only changed one color, we can just re-check
                        # But we already updated flags.
                        # Easier: just re-check and update flags back?
                        # Or save old flags?
                        # Let's just re-check and set flags correctly.
                        is_violated = not check_ap_valid(aps[affected_ai], colors)
                        # But we need to adjust num_violations back
                        # This is getting messy.
                        
                        # Alternative: Only accept if strictly better.
                        # If we revert, we need to restore num_violations.
                        # Let's just re-compute num_violations from scratch? No, too slow.
                        
                        # Let's use a simpler heuristic:
                        # Only try moves that fix the chosen AP?
                        # Or just accept any move, and rely on randomness?
                        # Random walk might get stuck.
                        
                        # Let's try: Accept if num_violations < old_num_violations.
                        # If not, revert color and re-compute flags for affected APs?
                        # We can just re-check affected APs and update flags and count.
                        
                        # Actually, if we revert, the state is exactly as before.
                        # So we can just restore old_c and the flags/count to old values.
                        # But we modified flags/count in the loop.
                        # We didn't save them.
                        
                        # Solution: Save old flags for affected APs?
                        # Or just re-compute num_violations?
                        # Given N=128, re-computing all violations is ~5000 checks.
                        # This is fast enough to do every 100 steps?
                        # Or just do it if revert is needed?
                        
                        # Let's just re-compute the affected APs' status and adjust count.
                        # But we already adjusted it.
                        
                        # Let's change strategy:
                        # 1. Save old_c.
                        # 2. Save old_violation_flags for affected APs? No.
                        # 3. Just re-compute the entire violation state if we want to revert?
                        # No, that's $O(N^2)$.
                        
                        # Let's try a different approach:
                        # Only pick moves that fix the current violation.
                        # i.e., new color must make the chosen AP valid.
                        # This guarantees that at least one violation is fixed (the chosen one).
                        # It might break others.
                        # If net change is positive, revert.
                        
                        # To revert safely:
                        # We can just set colors[idx] = old_c.
                        # Then re-evaluate the affected APs and update flags/count.
                        
                        # Let's implement this revert logic.
                        
                        pass # Placeholder for revert logic
                    
                    if num_violations >= old_num_violations:
                        colors[idx] = old_c
                        # Revert flags and count
                        # Re-compute status of affected APs
                        # We need to know the OLD status to adjust count?
                        # We can just re-compute the count for affected APs and adjust.
                        # But we don't know the old count contribution.
                        
                        # Easiest: Just re-compute ALL violations?
                        # 5000 APs. 5000 * 4 checks = 20000 ops.
                        # If we do this every revert, and we have many reverts, it might be slow.
                        # But local search steps are fast.
                        
                        # Let's just re-compute the whole thing if we revert.
                        # It's a safe fallback.
                        
                        new_num_violations = 0
                        for ai2 in range(num_aps):
                            if not check_ap_valid(aps[ai2], colors):
                                violation_flags[ai2] = True
                                new_num_violations += 1
                            else:
                                violation_flags[ai2] = False
                        num_violations = new_num_violations
                    else:
                        # Accepted move
                        pass # Flags and count already updated

            if num_violations == 0:
                k_used = len(set(colors))
                if k_used < best_k:
                    best_colors = colors
                    best_k = k_used
                return best_colors

    return best_colors
