import time
import random

def construct():
    N = 128
    # Precompute APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    M = len(aps)
    
    # Map position to list of AP indices
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Precompute bit masks for colors 0..127 (we might use up to 128 in fallback)
    # But for k=3..8, we only need 0..7
    # We'll compute masks on the fly for small k to save memory, or precompute for 0..127
    # Since k is small, 1 << c is fast.
    
    def check_ap(ap, colors):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        # Count distinct colors
        # Using bitmask is faster than set() for small ints
        mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << c3)
        # Count bits
        # bin(mask).count('1') is okay, but we can use a lookup table for 16-bit chunks if needed
        # For k<=8, mask is small.
        return bin(mask).count('1') >= 3

    def count_violations(colors):
        v = 0
        for ap in aps:
            if not check_ap(ap, colors):
                v += 1
        return v

    def solve_with_k(k, initial_colors, time_limit):
        start_time = time.time()
        colors = initial_colors[:]
        
        # Precompute masks for colors 0..k-1
        cm = [1 << i for i in range(k)]
        
        # Track bad APs
        is_bad = [False] * M
        bad_count = 0
        for idx, ap in enumerate(aps):
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            mask = cm[c0] | cm[c1] | cm[c2] | cm[c3]
            if bin(mask).count('1') < 3:
                is_bad[idx] = True
                bad_count += 1
        
        if bad_count == 0:
            return colors

        max_steps = 20000
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return None
            if bad_count == 0:
                return colors
            
            # Pick a random bad AP
            # Optimization: Keep a list of bad AP indices? Or just sample.
            # Sampling is O(1) but might miss if bad_count is small.
            # Since we start with many violations, sampling is fine.
            # If bad_count is small, we might need to scan.
            # Let's maintain a list of bad APs for efficiency.
            
            # Actually, maintaining a list of bad APs is tricky with updates.
            # Let's stick to random sampling but increase trials if bad_count is low.
            trials = 20 if bad_count > 10 else 100
            found_bad_ap = False
            for _ in range(trials):
                ap_idx = random.randint(0, M - 1)
                if is_bad[ap_idx]:
                    ap = aps[ap_idx]
                    pos = random.choice(ap)
                    found_bad_ap = True
                    break
            
            if not found_bad_ap:
                # Fallback: pick random pos
                pos = random.randint(0, N - 1)
            
            current_val = colors[pos]
            
            # Try to find a better value for pos
            best_val = current_val
            best_delta = 0 # We want to minimize bad_count, so delta < 0 is good
            
            # Calculate current bad status for APs involving pos
            # We need to check delta for each new_val
            
            # Optimization: Pre-compute the other 3 colors for each AP involving pos?
            # This is expensive. Let's just iterate.
            
            for new_val in range(k):
                if new_val == current_val:
                    continue
                
                delta = 0
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    # Get colors, replacing pos with new_val
                    c0 = colors[ap[0]] if ap[0] != pos else new_val
                    c1 = colors[ap[1]] if ap[1] != pos else new_val
                    c2 = colors[ap[2]] if ap[2] != pos else new_val
                    c3 = colors[ap[3]] if ap[3] != pos else new_val
                    
                    mask = cm[c0] | cm[c1] | cm[c2] | cm[c3]
                    is_bad_new = bin(mask).count('1') < 3
                    was_bad = is_bad[ap_idx]
                    
                    if was_bad and not is_bad_new:
                        delta -= 1
                    elif not was_bad and is_bad_new:
                        delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_val = new_val
            
            # If no improvement found (best_delta >= 0), try a random move to escape local min
            if best_delta >= 0:
                best_val = random.randint(0, k - 1)
                if best_val == current_val:
                    best_val = (best_val + 1) % k
                
                # Calculate delta for this random choice
                delta = 0
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    c0 = colors[ap[0]] if ap[0] != pos else best_val
                    c1 = colors[ap[1]] if ap[1] != pos else best_val
                    c2 = colors[ap[2]] if ap[2] != pos else best_val
                    c3 = colors[ap[3]] if ap[3] != pos else best_val
                    
                    mask = cm[c0] | cm[c1] | cm[c2] | cm[c3]
                    is_bad_new = bin(mask).count('1') < 3
                    was_bad = is_bad[ap_idx]
                    
                    if was_bad and not is_bad_new:
                        delta -= 1
                    elif not was_bad and is_bad_new:
                        delta += 1
                best_delta = delta
            
            # Apply change
            colors[pos] = best_val
            bad_count += best_delta
            
            # Update is_bad array
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                mask = cm[c0] | cm[c1] | cm[c2] | cm[c3]
                is_bad[ap_idx] = bin(mask).count('1') < 3

        return None

    global_deadline = time.time() + 55.0 # 5s margin
    
    # Strategy:
    # 1. Try k=3 with Hamming init
    # 2. Try k=4 with Hamming init
    # 3. Try k=5, 6, 7, 8
    # 4. Fallback to Hamming (8 colors)
    
    for k in [3, 4, 5, 6, 7, 8]:
        if time.time() > global_deadline:
            break
        
        # Initialize with Hamming weights mod k
        init_colors = []
        for i in range(N):
            # Hamming weight of i
            hw = bin(i).count('1')
            init_colors.append(hw % k)
        
        time_budget = 15.0 if k <= 4 else 5.0
        sol = solve_with_k(k, init_colors, time_budget)
        
        if sol is not None:
            # Verify solution just in case
            if count_violations(sol) == 0:
                return sol

    # Fallback: Return Hamming weights (8 colors)
    # This is valid? 
    # Let's check if Hamming weights 0..7 satisfy the condition.
    # 4-AP in Hamming weights:
    # i, i+d, i+2d, i+3d.
    # It is known that Hamming weight is not necessarily 3-distinct for all 4-APs?
    # Actually, the problem statement implies we need to *construct* it.
    # If the fallback fails, we return a known valid high-color solution.
    # list(range(N)) is always valid (128 colors).
    
    # Let's try to return the Hamming sequence, but verify it.
    hw_colors = [bin(i).count('1') for i in range(N)]
    if count_violations(hw_colors) == 0:
        return hw_colors
    
    # Absolute fallback
    return list(range(N))
