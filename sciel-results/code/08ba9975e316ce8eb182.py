import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    
    # Precompute all 4-term APs
    all_aps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                all_aps.append((a, b, c, e))
    
    # For each position, precompute which APs it participates in and its index in those APs
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(all_aps):
        pos_aps[a].append((idx, 0))
        pos_aps[b].append((idx, 1))
        pos_aps[c].append((idx, 2))
        pos_aps[e].append((idx, 3))
    
    def count_violations(colors):
        """Count number of 4-term APs with fewer than 3 distinct colors."""
        violations = 0
        for a, b, c, e in all_aps:
            vals = {colors[a], colors[b], colors[c], colors[e]}
            if len(vals) < 3:
                violations += 1
        return violations
    
    def get_violating_aps_for_position(colors, pos):
        """Get list of AP indices that are currently violating and involve position pos."""
        violating = []
        for ap_idx, role in pos_aps[pos]:
            a, b, c, e = all_aps[ap_idx]
            vals = {colors[a], colors[b], colors[c], colors[e]}
            if len(vals) < 3:
                violating.append((ap_idx, role))
        return violating
    
    def would_violate(colors, pos, new_color):
        """Check if setting colors[pos] = new_color would create any violation in APs involving pos."""
        for ap_idx, role in pos_aps[pos]:
            a, b, c, e = all_aps[ap_idx]
            # Build the set of colors in this AP with the hypothetical new color
            if role == 0:
                vals = {new_color, colors[b], colors[c], colors[e]}
            elif role == 1:
                vals = {colors[a], new_color, colors[c], colors[e]}
            elif role == 2:
                vals = {colors[a], colors[b], new_color, colors[e]}
            else:
                vals = {colors[a], colors[b], colors[c], new_color}
            if len(vals) < 3:
                return True
        return False
    
    def try_local_search(num_colors, max_steps, seed):
        """Simulated annealing local search."""
        rng = random.Random(seed)
        
        # Initialize with random coloring
        colors = [rng.randint(0, num_colors - 1) for _ in range(n)]
        cur_viol = count_violations(colors)
        
        if cur_viol == 0:
            return colors
        
        best_colors = colors[:]
        best_viol = cur_viol
        
        # Simulated annealing parameters
        T_start = 5.0
        T_end = 0.1
        cooling = (T_end / T_start) ** (1.0 / max(max_steps, 1))
        T = T_start
        
        step = 0
        no_improve = 0
        max_no_improve = 5000
        
        while step < max_steps and no_improve < max_no_improve:
            if time.time() > deadline:
                break
            
            # Pick a random position to reassign
            pos = rng.randint(0, n - 1)
            old_color = colors[pos]
            
            # Pick a new color different from old
            new_color = rng.randint(0, num_colors - 1)
            while new_color == old_color:
                new_color = rng.randint(0, num_colors - 1)
            
            # Check if this move would create new violations
            if would_violate(colors, pos, new_color):
                # Compute delta: how many violations would be added
                # We need to count violations in APs involving pos before and after
                # But since we know it would violate, let's compute the delta properly
                # Actually, let's just skip moves that create violations when we have few violations
                if cur_viol <= 1:
                    step += 1
                    T *= cooling
                    continue
                
                # For moves that create violations, only accept with low probability
                # Estimate delta: number of APs involving pos that would newly violate
                # minus number that would be fixed
                new_viol_count = 0
                for ap_idx, role in pos_aps[pos]:
                    a, b, c, e = all_aps[ap_idx]
                    if role == 0:
                        old_vals = {old_color, colors[b], colors[c], colors[e]}
                        new_vals = {new_color, colors[b], colors[c], colors[e]}
                    elif role == 1:
                        old_vals = {colors[a], old_color, colors[c], colors[e]}
                        new_vals = {colors[a], new_color, colors[c], colors[e]}
                    elif role == 2:
                        old_vals = {colors[a], colors[b], old_color, colors[e]}
                        new_vals = {colors[a], colors[b], new_color, colors[e]}
                    else:
                        old_vals = {colors[a], colors[b], colors[c], old_color}
                        new_vals = {colors[a], colors[b], colors[c], new_color}
                    
                    old_bad = 1 if len(old_vals) < 3 else 0
                    new_bad = 1 if len(new_vals) < 3 else 0
                    new_viol_count += new_bad - old_bad
                
                delta = new_viol_count
                if delta <= 0:
                    # This move doesn't increase violations, accept it
                    colors[pos] = new_color
                    cur_viol += delta
                    if cur_viol < best_viol:
                        best_viol = cur_viol
                        best_colors = colors[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Accept with probability exp(-delta/T)
                    import math
                    prob = math.exp(-delta / max(T, 1e-10))
                    if rng.random() < prob:
                        colors[pos] = new_color
                        cur_viol += delta
                        if cur_viol < best_viol:
                            best_viol = cur_viol
                            best_colors = colors[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
            else:
                # Move doesn't create any new violations in APs involving pos
                # But it might fix some. Compute delta.
                delta = 0
                for ap_idx, role in pos_aps[pos]:
                    a, b, c, e = all_aps[ap_idx]
                    if role == 0:
                        old_vals = {old_color, colors[b], colors[c], colors[e]}
                        new_vals = {new_color, colors[b], colors[c], colors[e]}
                    elif role == 1:
                        old_vals = {colors[a], old_color, colors[c], colors[e]}
                        new_vals = {colors[a], new_color, colors[c], colors[e]}
                    elif role == 2:
                        old_vals = {colors[a], colors[b], old_color, colors[e]}
                        new_vals = {colors[a], colors[b], new_color, colors[e]}
                    else:
                        old_vals = {colors[a], colors[b], colors[c], old_color}
                        new_vals = {colors[a], colors[b], colors[c], new_color}
                    
                    old_bad = 1 if len(old_vals) < 3 else 0
                    new_bad = 1 if len(new_vals) < 3 else 0
                    delta += new_bad - old_bad
                
                # delta <= 0 always here since no new violations
                colors[pos] = new_color
                cur_viol += delta
                if cur_viol < best_viol:
                    best_viol = cur_viol
                    best_colors = colors[:]
                    no_improve = 0
                else:
                    no_improve += 1
            
            step += 1
            T *= cooling
        
        if best_viol == 0:
            return best_colors
        return None
    
    # Try with 3 colors first (the minimum possible)
    # Use multiple random seeds for diversity
    best_result = None
    
    # First, try a few quick local searches with 3 colors
    num_attempts_3 = 20
    steps_per_attempt = 50000
    
    for attempt in range(num_attempts_3):
        if time.time() > deadline:
            break
        result = try_local_search(3, steps_per_attempt, attempt * 7919 + 1)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # If 3 colors didn't work, try 4 colors
    num_attempts_4 = 10
    steps_per_attempt_4 = 30000
    
    for attempt in range(num_attempts_4):
        if time.time() > deadline:
            break
        result = try_local_search(4, steps_per_attempt_4, attempt * 104729 + 100)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: try 5 colors
    for attempt in range(5):
        if time.time() > deadline:
            break
        result = try_local_search(5, 20000, attempt * 15485863 + 200)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Final fallback: all distinct colors
    return list(range(n))
