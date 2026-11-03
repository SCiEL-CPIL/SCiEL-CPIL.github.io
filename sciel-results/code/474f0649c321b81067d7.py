<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios: w + h = 2 (perimeter 4)
    # Focus on a range of aspect ratios, with more emphasis on near-square
    aspect_configs = [
        (1.0, 1.0),      # Square
        (1.05, 0.95),
        (0.95, 1.05),
        (1.1, 0.9),
        (0.9, 1.1),
        (1.15, 0.85),
        (0.85, 1.15),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.25, 0.75),
        (0.75, 1.25),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.5, 0.5),
        (0.5, 1.5),
    ]
    
    # Allocate time: more for near-square, less for extreme aspects
    time_weights = {
        (1.0, 1.0): 25,
        (1.05, 0.95): 20,
        (0.95, 1.05): 20,
        (1.1, 0.9): 20,
        (0.9, 1.1): 20,
        (1.15, 0.85): 15,
        (0.85, 1.15): 15,
        (1.2, 0.8): 15,
        (0.8, 1.2): 15,
        (1.25, 0.75): 10,
        (0.75, 1.25): 10,
        (1.3, 0.7): 10,
        (0.7, 1.3): 10,
        (1.4, 0.6): 8,
        (0.6, 1.4): 8,
        (1.5, 0.5): 6,
        (0.5, 1.5): 6,
    }
    
    for w, h in aspect_configs:
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 12:
            break
        
        # Allocate time based on weight and remaining time
        weight = time_weights.get((w, h), 10)
        total_weight = sum(time_weights.values())
        time_per_aspect = min(time_remaining - 5, max(5, (weight / total_weight) * 100))
        
        num_starts = 4 if time_per_aspect > 20 else 3
        
        result, score = _pack_for_aspect(w, h, num_starts=num_starts, 
                                         time_limit=time_per_aspect, deadline=deadline)
        
        if result is not None and score > best_score:
            best_score = score
            best_result = result
    
    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        cols, rows = 5, 5
        positions = []
        radii = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pos = [(i + 0.5) * w / cols, (j + 0.5) * h / rows]
                r = 0.5 * min(w/cols, h/rows) * 0.95
                positions.append(pos)
                radii.append(r)
                count += 1
        best_result = np.column_stack([np.array(positions), np.array(radii)])
    
    return best_result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios: w + h = 2 (perimeter 4)
    # Focus on near-square with some elongation
    aspect_configs = [
        (1.0, 1.0),
        (1.05, 0.95),
        (0.95, 1.05),
        (1.1, 0.9),
        (0.9, 1.1),
        (1.15, 0.85),
        (0.85, 1.15),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.25, 0.75),
        (0.75, 1.25),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.5, 0.5),
        (0.5, 1.5),
    ]
    
    # Time allocation: more for promising aspects
    time_weights = {
        (1.0, 1.0): 30,
        (1.05, 0.95): 22,
        (0.95, 1.05): 22,
        (1.1, 0.9): 22,
        (0.9, 1.1): 22,
        (1.15, 0.85): 16,
        (0.85, 1.15): 16,
        (1.2, 0.8): 14,
        (0.8, 1.2): 14,
        (1.25, 0.75): 10,
        (0.75, 1.25): 10,
        (1.3, 0.7): 8,
        (0.7, 1.3): 8,
        (1.4, 0.6): 6,
        (0.6, 1.4): 6,
        (1.5, 0.5): 5,
        (0.5, 1.5): 5,
    }
    
    total_weight = sum(time_weights.values())
    
    for w, h in aspect_configs:
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            break
        
        weight = time_weights.get((w, h), 8)
        time_per_aspect = min(time_remaining - 5, max(5, (weight / total_weight) * 110))
        
        num_starts = 4 if time_per_aspect > 20 else 3
        
        result, score = _pack_for_aspect(w, h, num_starts=num_starts,
                                         time_limit=time_per_aspect, deadline=deadline)
        
        if result is not None and score > best_score:
            # Validate before accepting
            if _validate_result(result, w, h, N):
                best_score = score
                best_result = result
            else:
                fixed = _fix_result(result, w, h, N)
                if fixed is not None and _validate_result(fixed, w, h, N):
                    score = np.sum(fixed[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = fixed
    
    # Fallback: simple grid
    if best_result is None:
        w, h = 1.0, 1.0
        cols, rows = 5, 5
        positions = []
        radii = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pos = [(i + 0.5) * w / cols, (j + 0.5) * h / rows]
                r = 0.5 * min(w/cols, h/rows) * 0.9
                positions.append(pos)
                radii.append(r)
                count += 1
        best_result = np.column_stack([np.array(positions), np.array(radii)])
    
    return best_result
>>>>>>> REPLACE
