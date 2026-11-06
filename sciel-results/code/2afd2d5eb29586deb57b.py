"""
Combined approach: Use deterministic Hamming-weight construction as a strong
initialization, then aggressively minimize colors via simulated annealing
with incremental violation counting.
"""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    N = 128

    # Precompute all 4-APs (0-indexed): (a, a+d, a+2d, a+3d) with a+3d < N
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            i0, i1, i2, i3 = a, a + d, a + 2 * d, a + 3 * d
            ap_id = len(aps)
            aps.append((i0, i1, i2, i3))
            ap_indices[i0].append(ap_id)
            ap_indices[i1].append(ap_id)
            ap_indices[i2].append(ap_id)
            ap_indices[i3].append(ap_id)

    def count_violations(coloring):
        """Count number of 4-APs with fewer than 3 distinct colors."""
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations

    def validate(coloring):
        """Check every nonconstant 4-term AP has >= 3 distinct colors."""
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                return False
        return True

    def relabel(colors):
        mapping = {}
        new_colors = []
        next_id = 0
        for c in colors:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new_colors.append(mapping[c])
        return new_colors, next_id

    def count_colors(colors):
        return len(set(colors))

    def hamming_weight_coloring():
        """Deterministic coloring by Hamming weight of position index."""
        return [bin(i).count('1') for i in range(N)]

    def modular_coloring(mod):
        return [i % mod for i in range(N)]

    def try_minimize_colors(initial, k_max, time_budget, rng):
        """
        Start from initial coloring, try to reduce distinct colors to <= k_max
        using simulated annealing that allows recoloring to any of k_max colors.
        Returns best coloring found.
        """
        colors = initial[:]
        n_colors = count_colors(colors)
        current_violations = count_violations(colors)
        
        if current_violations == 0 and n_colors <= k_max:
            return colors, n_colors
        
        best_colors = colors[:]
        best_count = n_colors
        best_violations = current_violations
        
        # We want to end up with <= k_max colors. 
        # Strategy: SA where we can recolor to any of the first k_max colors (0..k_max-1).
        # But we also want to *remove* colors. 
        # A good heuristic: pick a random color from current palette, and try to map it to another.
        # Or simpler: just allow recoloring to any of 0..k_max-1. If a color disappears, great.
        
        T = 5.0
        cooling = 0.9995
        steps = 0
        max_steps = 1000000
        start_t = time.time()
        
        # Pre-allocate for speed? No, just use lists.
        
        while current_violations > 0 or n_colors > k_max:
            if time.time() - start_t > time_budget:
                break
            if steps > max_steps:
                break
            
            steps += 1
            idx = rng.randint(0, N - 1)
            old_color = colors[idx]
            
            # Choose a new color. 
            # Bias towards reducing color count? 
            # If we pick from 0..k_max-1, we might introduce a new color if k_max > current distinct?
            # No, if we restrict to 0..k_max-1, and current colors are relabeled to 0..n-1, 
            # then if n > k_max, we are forcing some colors to merge.
            
            # Relabel current colors to 0..n-1 for the SA process?
            # Actually, it's easier to just keep the labels and pick a target.
            # Let's relabel the current best to 0..n-1 first, then SA with palette 0..k_max-1.
            # But relabeling every step is expensive.
            
            # Alternative: The SA state is the coloring. The "energy" is violations + penalty for colors > k_max.
            # Let's stick to: recolor idx to a random color in 0..k_max-1.
            # This implicitly merges colors.
            
            new_color = rng.randint(0, k_max - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate delta violations
            delta_violations = 0
            for ap_id in ap_indices[idx]:
                ap = aps[ap_id]
                c0 = colors[ap[0]]
                c1 = colors[ap[1]]
                c2 = colors[ap[2]]
                c3 = colors[ap[3]]
                
                old_set = {c0, c1, c2, c3}
                old_size = len(old_set)
                
                new_c0 = new_color if ap[0] == idx else c0
                new_c1 = new_color if ap[1] == idx else c1
                new_c2 = new_color if ap[2] == idx else c2
                new_c3 = new_color if ap[3] == idx else c3
                
                new_set = {new_c0, new_c1, new_c2, new_c3}
                new_size = len(new_set)
                
                if old_size < 3 and new_size >= 3:
                    delta_violations -= 1
                elif old_size >= 3 and new_size < 3:
                    delta_violations += 1
            
            # Acceptance
            if delta_violations < 0:
                colors[idx] = new_color
                current_violations += delta_violations
            else:
                if delta_violations == 0:
                    accept = rng.random() < 0.3
                else:
                    if T > 0.001:
                        accept = rng.random() < np.exp(-delta_violations / T)
                    else:
                        accept = False
                
                if accept:
                    colors[idx] = new_color
                    current_violations += delta_violations
            
            # Check for improvement in color count (only if valid)
            if current_violations == 0:
                new_count = count_colors(colors)
                if new_count < best_count:
                    best_count = new_count
                    best_colors = colors[:]
                    # If we hit the target k_max, we can stop or continue to see if we can go lower?
                    # The goal is to minimize. But the prompt says "Minimize the number of labels".
                    # So we should try to go below k_max if possible.
                    # However, the SA is restricted to 0..k_max-1.
                    # To go lower, we'd need to restart with a smaller k_max.
                    # For now, let's just find a valid one with <= k_max.
                    if best_count <= k_max:
                        # We can break early if we are confident, but let's run a bit longer to see if it stabilizes lower?
                        # Actually, if we break, we return best_colors.
                        # Let's continue for a few more steps to ensure stability.
                        pass
            
            T *= cooling
            
            # Periodic check
            if steps % 1000 == 0:
                # Optional: try to force merge a color?
                pass
        
        return best_colors, best_count

    # --- Main Strategy ---
    
    rng = random.Random(42)
    
    # 1. Get a valid initial coloring
    # Hamming weight usually gives 8 colors and is likely valid or close to valid.
    init_hw = hamming_weight_coloring()
    init_mod4 = modular_coloring(4)
    init_mod5 = modular_coloring(5)
    
    candidates = [init_hw, init_mod4, init_mod5]
    
    best_global = None
    best_global_count = N + 1
    
    # Evaluate initial candidates
    for c in candidates:
        if validate(c):
            cnt = count_colors(c)
            if cnt < best_global_count:
                best_global_count = cnt
                best_global = c[:]
    
    # If none valid, use SA to fix the best one (fewest violations)
    if best_global is None:
        # Find the one with fewest violations
        best_init = min(candidates, key=count_violations)
        # Run SA to make it valid, allowing up to 8 colors (Hamming weight max is 7, so 8 colors)
        # Actually, let's just run SA with k_max=8 on the best initial guess.
        fixed, cnt = try_minimize_colors(best_init, 8, 10.0, rng)
        if validate(fixed):
            best_global = fixed
            best_global_count = cnt
    
    # If we still have nothing (unlikely), fallback to greedy
    if best_global is None:
        # Greedy construction
        colors = [0] * N
        for i in range(N):
            forbidden = set()
            # 4th term
            for d in range(1, i // 3 + 1):
                a = i - 3 * d
                if a >= 0:
                    s = {colors[a], colors[a+d], colors[a+2*d]}
                    if len(s) <= 2:
                        forbidden |= s
            # 3rd term
            for d in range(1, min(i // 2, (N - 1 - i)) + 1):
                a = i - 2 * d
                if a >= 0:
                    b = a + d
                    e = a + 3 * d
                    if e < N:
                        s = {colors[a], colors[b], colors[e]}
                        if len(s) <= 2:
                            forbidden |= s
            # 2nd term
            for d in range(1, min(i, (N - 1 - i) // 2) + 1):
                a = i - d
                if a >= 0:
                    c = i + d
                    e = i + 2 * d
                    if e < N:
                        s = {colors[a], colors[c], colors[e]}
                        if len(s) <= 2:
                            forbidden |= s
            # 1st term
            for d in range(1, (N - 1 - i) // 3 + 1):
                b = i + d
                c = i + 2 * d
                e = i + 3 * d
                s = {colors[b], colors[c], colors[e]}
                if len(s) <= 2:
                    forbidden |= s
            
            for c in range(N):
                if c not in forbidden:
                    colors[i] = c
                    break
            else:
                colors[i] = 0
        best_global = colors
        best_global_count = count_colors(colors)

    # 2. Try to minimize colors further
    # We have a valid coloring with best_global_count colors.
    # Try to reduce it to k < best_global_count.
    
    # Order of k to try: 3, 4, 5... up to best_global_count - 1
    # But we also have time to try higher k if we start from a bad state? No, we start from valid.
    
    # Let's try k = 3, 4, 5, 6, 7
    # Allocate time based on remaining budget.
    remaining = deadline - time.time()
    
    # We'll try k=3 first. If it works, great. If not, try k=4, etc.
    # But we must ensure we always return a valid coloring.
    # So we keep best_global as the fallback.
    
    # Time allocation:
    # k=3: 15s
    # k=4: 10s
    # k=5: 8s
    # k=6: 5s
    # k=7: 3s
    # Fallback: rest
    
    k_targets = [3, 4, 5, 6, 7]
    time_alloc = [15.0, 10.0, 8.0, 5.0, 3.0]
    
    for idx, k in enumerate(k_targets):
        if k >= best_global_count:
            continue
        if time.time() > deadline - 5:
            break
        
        t_budget = min(time_alloc[idx], deadline - time.time() - 5)
        if t_budget < 1.0:
            break
        
        print(f"Trying to minimize to k={k}...", file=sys.stderr)
        
        # Use the current best_global as the starting point for SA
        # Note: try_minimize_colors assumes we want to fit into 0..k-1.
        # It will recolor positions to 0..k-1.
        # If it finds a valid one with <= k colors, it returns it.
        
        candidate, cnt = try_minimize_colors(best_global, k, t_budget, rng)
        
        if validate(candidate) and cnt <= k:
            if cnt < best_global_count:
                best_global = candidate
                best_global_count = cnt
                print(f"Success! Reduced to {cnt} colors", file=sys.stderr)
                # If we hit 3, we can't do better, so break.
                if cnt == 3:
                    break
    
    # Final relabel
    best_global, final_count = relabel(best_global)
    
    # Final validation
    if not validate(best_global):
        print("WARNING: Final validation failed, using Hamming weight fallback", file=sys.stderr)
        best_global = hamming_weight_coloring()
        # If HW is invalid, try mod 4
        if not validate(best_global):
            best_global = modular_coloring(4)
        # If mod 4 is invalid, try mod 5
        if not validate(best_global):
            best_global = modular_coloring(5)
        # If still invalid, greedy
        if not validate(best_global):
             # Re-run greedy quickly
             colors = [0] * N
             for i in range(N):
                 forbidden = set()
                 for d in range(1, i // 3 + 1):
                     a = i - 3 * d
                     if a >= 0:
                         s = {colors[a], colors[a+d], colors[a+2*d]}
                         if len(s) <= 2: forbidden |= s
                 for d in range(1, min(i // 2, (N - 1 - i)) + 1):
                     a = i - 2 * d
                     if a >= 0:
                         b = a + d; e = a + 3 * d
                         if e < N:
                             s = {colors[a], colors[b], colors[e]}
                             if len(s) <= 2: forbidden |= s
                 for d in range(1, min(i, (N - 1 - i) // 2) + 1):
                     a = i - d
                     if a >= 0:
                         c = i + d; e = i + 2 * d
                         if e < N:
                             s = {colors[a], colors[c], colors[e]}
                             if len(s) <= 2: forbidden |= s
                 for d in range(1, (N - 1 - i) // 3 + 1):
                     b = i + d; c = i + 2 * d; e = i + 3 * d
                     s = {colors[b], colors[c], colors[e]}
                     if len(s) <= 2: forbidden |= s
                 for c in range(N):
                     if c not in forbidden:
                         colors[i] = c
                         break
                 else:
                     colors[i] = 0
             best_global = colors
             
        best_global, final_count = relabel(best_global)

    return best_global
# EVOLVE-BLOCK-END
