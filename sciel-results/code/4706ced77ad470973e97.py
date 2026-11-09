import random
import time
import itertools

def _check_no_four_collinear(points):
    """Check that no four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Count how many points are collinear with i,j,k
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Direction vector
                dx = x2 - x1
                dy = y2 - y1
                count = 3  # i, j, k are collinear
                for m in range(n):
                    if m == i or m == j or m == k:
                        continue
                    xm, ym = points[m]
                    # Check if (xm,ym) is on the line through (x1,y1) and (x2,y2)
                    if dx * (ym - y1) == dy * (xm - x1):
                        count += 1
                if count >= 4:
                    return False
    return True

def _greedy_no_three_collinear(points):
    """Greedy lower bound on max subset with no three collinear."""
    n = len(points)
    # Precompute collinear triples
    collinear_triples = set()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                    collinear_triples.add((i,j,k))
    
    # Greedy: repeatedly pick point in fewest remaining triples
    remaining = set(range(n))
    chosen = []
    while remaining:
        # Count triples for each point
        counts = {}
        for p in remaining:
            c = 0
            for (a,b,c_) in collinear_triples:
                if a in remaining and b in remaining and c_ in remaining:
                    if p in (a,b,c_):
                        c += 1
            counts[p] = c
        # Pick point with minimum count
        min_p = min(remaining, key=lambda p: counts[p])
        chosen.append(min_p)
        remaining.remove(min_p)
    return len(chosen)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    best_score = 20
    best_points = None
    
    # Initial: try some structured configurations
    # 4x5 grid perturbed
    for attempt in range(50):
        if time.time() > deadline:
            break
        # Create a 4x5 grid
        base_x = list(range(-2, 3))  # -2,-1,0,1,2
        base_y = list(range(-2, 3))  # -2,-1,0,1,2
        # That's 25 points, need 20. Use 4x5 = 20
        xs = [-4, -2, 0, 2, 4]
        ys = [-4, -2, 0, 2, 4]
        # 5x5 = 25, remove 5
        points = []
        for x in xs:
            for y in ys:
                points.append([x, y])
        # Remove 5 random points to get 20
        random.shuffle(points)
        points = points[:20]
        
        # Perturb some points
        for i in range(len(points)):
            if random.random() < 0.3:
                points[i][0] += random.choice([-1, 1])
                points[i][1] += random.choice([-1, 1])
        
        # Check bounds
        valid = True
        seen = set()
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                valid = False
                break
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        
        if not valid:
            continue
        
        if not _check_no_four_collinear(points):
            continue
        
        score = _greedy_no_three_collinear(points)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in points]
    
    # Random search
    for attempt in range(5000):
        if time.time() > deadline:
            break
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        
        if not _check_no_four_collinear(points):
            continue
        
        score = _greedy_no_three_collinear(points)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in points]
    
    # Local search on best
    if best_points is not None and time.time() < deadline - 10:
        for iteration in range(200):
            if time.time() > deadline:
                break
            current = [list(p) for p in best_points]
            # Try moving one point
            i = random.randint(0, 19)
            old_x, old_y = current[i]
            current[i][0] = random.randint(-100, 100)
            current[i][1] = random.randint(-100, 100)
            
            # Check validity
            valid = True
            seen = set()
            for p in current:
                if abs(p[0]) > 100 or abs(p[1]) > 100:
                    valid = False
                    break
                if (p[0], p[1]) in seen:
                    valid = False
                    break
                seen.add((p[0], p[1]))
            
            if not valid:
                continue
            
            if not _check_no_four_collinear(current):
                continue
            
            score = _greedy_no_three_collinear(current)
            if score < best_score:
                best_score = score
                best_points = [list(p) for p in current]
            else:
                current[i] = [old_x, old_y]
    
    if best_points is None:
        # Fallback: parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_points
