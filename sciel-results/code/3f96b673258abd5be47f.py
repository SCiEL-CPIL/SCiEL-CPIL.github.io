import random

N = 64

def construct():
    all_points = [(x, y) for y in range(N) for x in range(N)]
    best_points = []
    best_size = 0
    
    # We will try a few different strategies:
    # 1. Random restarts with different seeds
    # 2. Deterministic orderings to diversify search
    
    # Generate a list of (candidates, selected, pair_dists) for each strategy
    strategies = []
    
    # Random restarts
    for seed in range(500):
        strategies.append(('random', seed))
    
    # Deterministic orderings
    strategies.append(('sorted_x', None))
    strategies.append(('sorted_y', None))
    strategies.append(('sorted_sum', None))
    strategies.append(('sorted_dist_center', None))
    strategies.append(('sorted_dist_origin', None))
    
    for strat, seed in strategies:
        candidates = all_points[:]
        
        if strat == 'random':
            random.seed(seed)
            random.shuffle(candidates)
        elif strat == 'sorted_x':
            candidates.sort(key=lambda p: (p[0], p[1]))
        elif strat == 'sorted_y':
            candidates.sort(key=lambda p: (p[1], p[0]))
        elif strat == 'sorted_sum':
            candidates.sort(key=lambda p: (p[0] + p[1], p[0]))
        elif strat == 'sorted_dist_center':
            cx, cy = N / 2.0, N / 2.0
            candidates.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        elif strat == 'sorted_dist_origin':
            candidates.sort(key=lambda p: p[0] ** 2 + p[1] ** 2)
        
        selected = []
        pair_dists = set()
        
        for p in candidates:
            px, py = p
            
            # Compute distances to all selected points
            d_p_to_s = []
            for a in selected:
                ax, ay = a
                d_sq = (px - ax) ** 2 + (py - ay) ** 2
                d_p_to_s.append(d_sq)
            
            # Check if any two distances are equal (Isosceles with P as apex)
            if len(d_p_to_s) != len(set(d_p_to_s)):
                continue
            
            # Check if any distance from P is equal to any existing pair distance
            has_conflict = False
            for d_sq in d_p_to_s:
                if d_sq in pair_dists:
                    has_conflict = True
                    break
            
            if has_conflict:
                continue
            
            # Valid
            selected.append(p)
            pair_dists.update(d_p_to_s)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_points = selected[:]
    
    return best_points
