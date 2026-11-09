import random
import time

def construct():
    N = 128
    # Precompute all 4-APs
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for start in range(N - 3 * d):
            aps.append((start, start + d, start + 2 * d, start + 3 * d))
    
    num_aps = len(aps)
    
    # For each index, store the list of AP indices it belongs to
    index_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            index_to_aps[pos].append(idx)
            
    def count_violations(colors):
        """Count number of APs with < 3 distinct colors."""
        viol = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                viol += 1
        return viol

    def local_search_fix(colors, allowed_colors, time_limit=5.0):
        """
        Try to fix violations in 'colors' using only 'allowed_colors'.
        Returns (success, fixed_colors)
        """
        start_time = time.time()
        current_viol = count_violations(colors)
        if current_viol == 0:
            return True, colors
        
        best_viol = current_viol
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        temp = 5.0
        cooling = 0.95
        no_improve_counter = 0
        max_no_improve = 5000
        
        while time.time() - start_time < time_limit and no_improve_counter < max_no_improve:
            # Pick random index
            i = random.randint(0, N - 1)
            old_c = colors[i]
            
            # Pick new color from allowed set
            if len(allowed_colors) == 1:
                new_c = allowed_colors[0]
            else:
                # Prefer colors not equal to old_c
                candidates = [c for c in allowed_colors if c != old_c]
                if not candidates:
                    candidates = allowed_colors
                new_c = random.choice(candidates)
            
            if new_c == old_c:
                continue
                
            # Calculate delta
            affected_aps = index_to_aps[i]
            
            # Save old states
            old_states = []
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                old_states.append(1 if len({c0, c1, c2, c3}) < 3 else 0)
            
            # Apply change
            colors[i] = new_c
            
            # Calculate new states
            new_states = []
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                new_states.append(1 if len({c0, c1, c2, c3}) < 3 else 0)
            
            delta = sum(new_states) - sum(old_states)
            
            # Acceptance
            if delta < 0 or (delta == 0 and random.random() < 0.3) or (delta > 0 and random.random() < pow(2.718, -delta / temp)):
                current_viol += delta
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_colors = colors[:]
                    no_improve_counter = 0
                    if current_viol == 0:
                        return True, best_colors
                else:
                    no_improve_counter += 1
            else:
                # Revert
                colors[i] = old_c
                no_improve_counter += 1
                
            temp *= cooling
            if temp < 0.1:
                temp = 5.0 # Reheat
                
        return best_viol == 0, best_colors

    start_time = time.time()
    time_limit = 55.0
    
    # Step 1: Get a valid coloring with K=8
    K = 8
    colors = [random.randint(0, K - 1) for _ in range(N)]
    success, colors = local_search_fix(colors, list(range(K)), time_limit=2.0)
    if not success:
        # Try again with K=10
        K = 10
        colors = [random.randint(0, K - 1) for _ in range(N)]
        success, colors = local_search_fix(colors, list(range(K)), time_limit=2.0)
        if not success:
            # Fallback: return the best we have (might be invalid, but unlikely)
            pass

    # Step 2: Reduce colors
    current_colors = colors[:]
    used_labels = set(current_colors)
    K_current = len(used_labels)
    
    # We want to minimize K_current.
    # We will try to merge colors.
    
    # To ensure we explore well, we can try multiple rounds of merging attempts.
    # If we fail to merge in one pass, we might still be able to merge later?
    # No, if we can't merge any pair, we are stuck at this K.
    # Unless we allow "splitting" which we don't.
    
    # However, the order of merging matters.
    # Merging A and B might fail, but Merging B and C might succeed, leading to a state where A and C can be merged.
    # So we should try many random pairs.
    
    attempts = 0
    max_attempts = 200 # Limit total merge attempts to avoid timeout
    
    while K_current > 3 and time.time() - start_time < time_limit and attempts < max_attempts:
        attempts += 1
        unique_labels = sorted(list(used_labels))
        if len(unique_labels) <= 1:
            break
            
        # Pick two random distinct labels to merge
        c1, c2 = random.sample(unique_labels, 2)
        
        # Create new coloring
        new_colors = []
        for c in current_colors:
            if c == c2:
                new_colors.append(c1)
            else:
                new_colors.append(c)
                
        allowed = [c for c in unique_labels if c != c2]
        
        # Check if valid
        viol = count_violations(new_colors)
        
        if viol == 0:
            current_colors = new_colors
            used_labels = set(current_colors)
            K_current = len(used_labels)
        else:
            # Try to fix
            # Use short time limit for each merge attempt
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time <= 0:
                break
            time_per_attempt = min(1.0, remaining_time / (max_attempts - attempts))
            success, fixed_colors = local_search_fix(new_colors, allowed, time_limit=time_per_attempt)
            if success:
                current_colors = fixed_colors
                used_labels = set(current_colors)
                K_current = len(used_labels)

    return current_colors
