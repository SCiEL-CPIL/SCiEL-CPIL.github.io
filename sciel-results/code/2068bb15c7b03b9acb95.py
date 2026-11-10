import time
import random
import itertools

def _score(points):
    """Compute the maximum multiplicity of a positive doubled area."""
    counts = {}
    n = len(points)
    # Optimization: Pre-extract coordinates? 
    # 4060 triples is small.
    for i in range(n - 2):
        x1, y1 = points[i]
        for j in range(i + 1, n - 1):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                d = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if d > 0:
                    counts[d] = counts.get(d, 0) + 1
    if not counts:
        return 0
    return max(counts.values())

def _generate_grid(w, h, offset_x=0, offset_y=0):
    pts = []
    for x in range(w):
        for y in range(h):
            pts.append((x + offset_x, y + offset_y))
    return pts

def construct():
    start_time = time.time()
    deadline = 55.0  # 55 seconds to be safe
    
    # Initial candidates
    candidates = [
        _generate_grid(5, 6, -2, -2),  # Centered-ish
        _generate_grid(6, 5, -2, -2),
        _generate_grid(3, 10, -1, -4),
        _generate_grid(10, 3, -4, -1),
        _generate_grid(2, 15, -0, -7),
        _generate_grid(15, 2, -7, -0),
    ]
    
    # Filter valid points (within [-30, 30])
    def is_valid(pts):
        return all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts) and len(set(pts)) == 30
    
    valid_candidates = [c for c in candidates if is_valid(c)]
    if not valid_candidates:
        # Fallback to seed
        return [(x, y) for x in range(6) for y in range(5)]
        
    # Evaluate initial candidates
    best_score = -1
    best_points = None
    for c in valid_candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_points = c[:]
            
    # Local Search
    # We will perform random restarts or a single long search from the best start.
    # Given the low cost of evaluation, a single long search is good.
    # We can also try multiple starts if time permits.
    
    current_points = best_points[:]
    current_score = best_score
    
    # To speed up random point generation, we use a list of all possible points
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # Local search loop
    while time.time() - start_time < deadline:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_pt = current_points[idx]
        
        # Pick a random new point
        # Sampling from all_points is uniform.
        # We need to ensure it's not already in the set (except the one we are moving)
        # Simple rejection sampling
        new_pt = None
        for _ in range(10): # Try a few times to find a valid distinct point
            candidate_pt = random.choice(all_points)
            if candidate_pt == old_pt:
                continue
            if candidate_pt in current_points:
                continue
            new_pt = candidate_pt
            break
        
        if new_pt is None:
            continue
            
        # Apply move
        current_points[idx] = new_pt
        
        # Evaluate
        new_score = _score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
        elif new_score == current_score:
            # Accept with some probability to explore
            if random.random() < 0.5:
                pass # Keep the move
            else:
                # Revert
                current_points[idx] = old_pt
        else:
            # Revert
            current_points[idx] = old_pt
            
    return best_points
