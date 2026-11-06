import time
import random

def construct():
    deadline = time.time() + 140
    random.seed(42)
    
    def check_no_four_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                dx1 = points[j][0] - points[i][0]
                dy1 = points[j][1] - points[i][1]
                for k in range(j+1, n):
                    dx2 = points[k][0] - points[i][0]
                    dy2 = points[k][1] - points[i][1]
                    if dx1 * dy2 == dy1 * dx2:
                        for l in range(k+1, n):
                            dx3 = points[l][0] - points[i][0]
                            dy3 = points[l][1] - points[i][1]
                            if dx1 * dy3 == dy1 * dx3:
                                return False
        return True
    
    def compute_collinear_triples(points):
        n = len(points)
        collinear = set()
        for i in range(n):
            for j in range(i+1, n):
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                for k in range(j+1, n):
                    dx2 = points[k][0] - points[i][0]
                    dy2 = points[k][1] - points[i][1]
                    if dx * dy2 == dy * dx2:
                        collinear.add(frozenset([i, j, k]))
        return collinear
    
    def max_no_three_collinear(points, timeout=3.0):
        n = len(points)
        start_time = time.time()
        collinear = compute_collinear_triples(points)
        
        collinear_pairs = [set() for _ in range(n)]
        for triple in collinear:
            for p in triple:
                others = tuple(sorted(triple - {p}))
                collinear_pairs[p].add(others)
        
        best = [0]
        
        def branch(idx, selected, forbidden):
            if time.time() - start_time > timeout:
                return
            
            remaining = n - idx
            upper = len(selected) + remaining
            if upper <= best[0]:
                return
            
            if idx == n:
                if len(selected) > best[0]:
                    best[0] = len(selected)
                return
            
            # Skip
            branch(idx + 1, selected, forbidden)
            
            # Include
            if idx not in forbidden:
                can_add = True
                new_forbidden = set(forbidden)
                for pair in collinear_pairs[idx]:
                    q, r = pair
                    if q in selected and r in selected:
                        can_add = False
                        break
                    elif q in selected:
                        new_forbidden.add(r)
                    elif r in selected:
                        new_forbidden.add(q)
                
                if can_add:
                    branch(idx + 1, selected | {idx}, frozenset(new_forbidden))
        
        branch(0, frozenset(), frozenset())
        return best[0]
    
    def score(points):
        if not check_no_four_collinear(points):
            return 999
        return max_no_three_collinear(points, timeout=3.0)
    
    best_points = None
    best_score = 999
    
    # Generate candidates
    candidates = []
    
    # Random
    for _ in range(80):
        pts = []
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if [x, y] not in pts:
                pts.append([x, y])
        candidates.append(pts)
    
    # Perturbed grid
    for _ in range(40):
        pts = []
        for i in range(4):
            for j in range(5):
                x = j * 8 + random.randint(-2, 2)
                y = i * 8 + random.randint(-2, 2)
                if [x, y] not in pts and abs(x) <= 100 and abs(y) <= 100:
                    pts.append([x, y])
        if len(pts) == 20:
            candidates.append(pts)
    
    # Evaluate
    for pts in candidates:
        if time.time() > deadline - 30:
            break
        s = score(pts)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in pts]
    
    # Local search
    if best_points is not None:
        current = [p[:] for p in best_points]
        current_score = best_score
        no_improve = 0
        
        while time.time() < deadline - 20 and no_improve < 80:
            improved = False
            for i in range(20):
                if time.time() > deadline - 20:
                    break
                for attempt in range(20):
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                    new_pts = [p[:] for p in current]
                    new_pts[i] = [new_x, new_y]
                    
                    if len(set(tuple(p) for p in new_pts)) != 20:
                        continue
                    
                    s = score(new_pts)
                    if s < current_score:
                        current = new_pts
                        current_score = s
                        no_improve = 0
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current]
                        break
                if time.time() > deadline - 20:
                    break
            if not improved:
                no_improve += 1
    
    if best_points is None:
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_points
