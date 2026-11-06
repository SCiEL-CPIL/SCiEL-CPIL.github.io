import time
import itertools
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    def compute_score(points):
        counts = {}
        for i, j, k in itertools.combinations(range(len(points)), 3):
            x1, y1 = points[i]
            x2, y2 = points[j]
            x3, y3 = points[k]
            a = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if a > 0:
                counts[a] = counts.get(a, 0) + 1
        if not counts:
            return 0
        return max(counts.values())

    def local_search(points, max_iters=1000):
        current_points = points[:]
        current_score = compute_score(current_points)
        n = len(current_points)
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                old_p = current_points[i]
                # Try neighbors in a 3x3 or 5x5 box
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        new_p = (old_p[0] + dx, old_p[1] + dy)
                        if not (-30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30):
                            continue
                        # Check distinctness
                        dup = False
                        for j in range(n):
                            if j != i and current_points[j] == new_p:
                                dup = True
                                break
                        if dup:
                            continue
                        
                        # Test move
                        test_points = current_points[:]
                        test_points[i] = new_p
                        new_score = compute_score(test_points)
                        
                        if new_score > current_score:
                            current_points = test_points
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return current_points, current_score

    best_points = None
    best_score = 0
    
    # Initial Candidates
    # 1. 5x6 Grid
    pts = [(x, y) for x in range(5) for y in range(6)]
    # Center
    cx = (min(p[0] for p in pts) + max(p[0] for p in pts)) // 2
    cy = (min(p[1] for p in pts) + max(p[1] for p in pts)) // 2
    pts = [(x - cx, y - cy) for x, y in pts]
    
    p, s = local_search(pts)
    if s > best_score:
        best_score = s
        best_points = p

    # 2. 3x10 Grid
    pts = [(x, y) for x in range(3) for y in range(10)]
    cx = (min(p[0] for p in pts) + max(p[0] for p in pts)) // 2
    cy = (min(p[1] for p in pts) + max(p[1] for p in pts)) // 2
    pts = [(x - cx, y - cy) for x, y in pts]
    
    p, s = local_search(pts)
    if s > best_score:
        best_score = s
        best_points = p

    # 3. 2x15 Grid
    pts = [(x, y) for x in range(2) for y in range(15)]
    cx = (min(p[0] for p in pts) + max(p[0] for p in pts)) // 2
    cy = (min(p[1] for p in pts) + max(p[1] for p in pts)) // 2
    pts = [(x - cx, y - cy) for x, y in pts]
    
    p, s = local_search(pts)
    if s > best_score:
        best_score = s
        best_points = p
        
    # 4. 6x5 Grid
    pts = [(x, y) for x in range(6) for y in range(5)]
    cx = (min(p[0] for p in pts) + max(p[0] for p in pts)) // 2
    cy = (min(p[1] for p in pts) + max(p[1] for p in pts)) // 2
    pts = [(x - cx, y - cy) for x, y in pts]
    
    p, s = local_search(pts)
    if s > best_score:
        best_score = s
        best_points = p

    # 5. Sheared 5x6
    base = [(x, y) for x in range(5) for y in range(6)]
    for k in [1, -1]:
        pts = [(x, y + k*x) for x, y in base]
        cx = (min(p[0] for p in pts) + max(p[0] for p in pts)) // 2
        cy = (min(p[1] for p in pts) + max(p[1] for p in pts)) // 2
        pts = [(x - cx, y - cy) for x, y in pts]
        
        p, s = local_search(pts)
        if s > best_score:
            best_score = s
            best_points = p

    # 6. Random Start
    random.seed(42)
    pts = []
    while len(pts) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in pts:
            pts.append((x, y))
    p, s = local_search(pts)
    if s > best_score:
        best_score = s
        best_points = p

    return best_points
