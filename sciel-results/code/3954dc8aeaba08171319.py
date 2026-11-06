<<<<<<< SEARCH
def _local_search(points, deadline):
    """
    Efficient local search with incremental scoring.
    Try single-point moves and two-point swaps.
    """
    current = [list(p) for p in points]
    current_score = _count_ordinary_fast(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Build candidate pool from lines through existing points
    coords = set()
    for i in range(12):
        for j in range(i+1, 12):
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                g = 1
            dx //= g
            dy //= g
            # Extend line in both directions
            for t in range(-3, 4):
                coords.add((x1 + t*dx, y1 + t*dy))
    
    # Also add nearby points
    for p in current:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                coords.add((p[0]+dx, p[1]+dy))
    
    candidate_list = list(coords)
    random.shuffle(candidate_list)
    
    no_improve = 0
    max_no_improve = 30
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        random.shuffle(candidate_list)
        
        for i in range(12):
            if time.time() >= deadline:
                break
            old_pt = tuple(current[i])
            for cand in candidate_list:
                if cand == old_pt:
                    continue
                if any(cand == tuple(current[j]) for j in range(12) if j != i):
                    continue
                
                new_pts = [list(p) for p in current]
                new_pts[i] = list(cand)
                
                if not _is_valid(new_pts):
                    continue
                
                score = _count_ordinary_fast(new_pts)
                if score < current_score:
                    current = new_pts
                    current_score = score
                    improved = True
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
                    break
            else:
                no_improve += 1
                continue
            break
        else:
            no_improve += 1
            continue
        
        if not improved:
            no_improve += 1
    
    return best, best_score
=======
def _local_search(points, deadline):
    """
    Efficient local search with single-point moves and two-point swaps.
    """
    current = [list(p) for p in points]
    current_score = _count_ordinary_fast(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Build a larger candidate pool
    coords = set()
    for i in range(12):
        for j in range(i+1, 12):
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                g = 1
            dx //= g
            dy //= g
            for t in range(-5, 6):
                coords.add((x1 + t*dx, y1 + t*dy))
    
    for p in current:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                coords.add((p[0]+dx, p[1]+dy))
    
    # Add some random points in a bounding box around the current set
    min_x = min(p[0] for p in current) - 5
    max_x = max(p[0] for p in current) + 5
    min_y = min(p[1] for p in current) - 5
    max_y = max(p[1] for p in current) + 5
    for _ in range(100):
        coords.add((random.randint(min_x, max_x), random.randint(min_y, max_y)))
        
    candidate_list = list(coords)
    random.shuffle(candidate_list)
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # Try single point moves
        for i in range(12):
            if time.time() >= deadline:
                break
            old_pt = tuple(current[i])
            for cand in candidate_list:
                if cand == old_pt:
                    continue
                if any(cand == tuple(current[j]) for j in range(12) if j != i):
                    continue
                
                new_pts = [list(p) for p in current]
                new_pts[i] = list(cand)
                
                if not _is_valid(new_pts):
                    continue
                
                score = _count_ordinary_fast(new_pts)
                if score < current_score:
                    current = new_pts
                    current_score = score
                    improved = True
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
                    break
            if improved:
                break
        
        # If no single move improved, try swaps
        if not improved:
            for i in range(12):
                if time.time() >= deadline:
                    break
                for j in range(i+1, 12):
                    if time.time() >= deadline:
                        break
                    old_i = tuple(current[i])
                    old_j = tuple(current[j])
                    for cand_i in candidate_list:
                        if cand_i == old_i or cand_i == old_j:
                            continue
                        if any(cand_i == tuple(current[k]) for k in range(12) if k not in [i, j]):
                            continue
                        for cand_j in candidate_list:
                            if cand_j == old_i or cand_j == old_j or cand_j == cand_i:
                                continue
                            if any(cand_j == tuple(current[k]) for k in range(12) if k not in [i, j]):
                                continue
                            
                            new_pts = [list(p) for p in current]
                            new_pts[i] = list(cand_i)
                            new_pts[j] = list(cand_j)
                            
                            if not _is_valid(new_pts):
                                continue
                            
                            score = _count_ordinary_fast(new_pts)
                            if score < current_score:
                                current = new_pts
                                current_score = score
                                improved = True
                                no_improve = 0
                                if current_score < best_score:
                                    best_score = current_score
                                    best = [list(p) for p in current]
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                no_improve += 1
        else:
            no_improve = 0
    
    return best, best_score
>>>>>>> REPLACE

