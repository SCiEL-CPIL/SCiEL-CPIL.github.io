def _local_search(colors, rng):
    current_violations = _count_violations(colors)
    if current_violations == 0:
        return colors, 0
    
    best_colors = list(colors)
    best_violations = current_violations
    
    stagnation_counter = 0
    max_stagnation = 1000
    
    while _time_left() > 0.1:
        if current_violations == 0:
            return colors, 0
            
        if stagnation_counter > max_stagnation:
            # Restart with random perturbation
            for _ in range(10):
                pos = rng.randint(0, 127)
                colors[pos] = rng.randint(0, 2)
            current_violations = _count_violations(colors)
            stagnation_counter = 0
            continue
            
        # Pick a random position
        pos = rng.randint(0, 127)
        old_color = colors[pos]
        
        # Try changing to a different color
        new_color = (old_color + 1 + rng.randint(0, 1)) % 3
        
        # Calculate change in violations
        delta = 0
        for p_idx in _POS_TO_PROGS[pos]:
            a, b, c, d = _PROGRESSIONS[p_idx]
            
            # Before
            old_set = {colors[a], colors[b], colors[c], colors[d]}
            old_viol = 1 if len(old_set) < 3 else 0
            
            # After
            if pos == a:
                new_set = {new_color, colors[b], colors[c], colors[d]}
            elif pos == b:
                new_set = {colors[a], new_color, colors[c], colors[d]}
            elif pos == c:
                new_set = {colors[a], colors[b], new_color, colors[d]}
            else:
                new_set = {colors[a], colors[b], colors[c], new_color}
                
            new_viol = 1 if len(new_set) < 3 else 0
            
            delta += new_viol - old_viol
        
        new_violations = current_violations + delta
        
        if new_violations < current_violations:
            colors[pos] = new_color
            current_violations = new_violations
            stagnation_counter = 0
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = list(colors)
        else:
            stagnation_counter += 1
            # Accept equal with small prob
            if new_violations == current_violations and rng.random() < 0.05:
                colors[pos] = new_color
                # stagnation_counter not reset? Or reset?
                # Let's not reset, it's a neutral move.
                
    return best_colors, best_violations
