import random
import itertools
import time

def is_collinear(p1, p2, p3):
    return (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]) == 0)

def has_four_collinear(points):
    # Check all combinations of 4
    for quad in itertools.combinations(points, 4):
        if is_collinear(quad[0], quad[1], quad[2]) and is_collinear(quad[0], quad[1], quad[3]):
            return True
    return False

def count_three_collinear(points):
    count = 0
    for tri in itertools.combinations(points, 3):
        if is_collinear(tri[0], tri[1], tri[2]):
            count += 1
    return count

def max_3_free_subset(points):
    # Branch and bound or simple greedy for small n
    # n=20 is small enough for a decent branch and bound
    n = len(points)
    # Precompute collinearity matrix
    coll = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if is_collinear(points[i], points[j], points[k]):
                    coll[i][j] = coll[i][k] = coll[j][k] = True
                    
    best = 0
    def backtrack(start, current, size):
        nonlocal best
        if size > best:
            best = size
        if start >= n:
            return
        # Pruning: if size + (n - start) <= best, return
        if size + (n - start) <= best:
            return
            
        # Option 1: Include points[start]
        # Check if it conflicts with any in current
        valid = True
        for i in current:
            if coll[i][start]:
                valid = False
                break
        if valid:
            backtrack(start + 1, current + [start], size + 1)
            
        # Option 2: Exclude points[start]
        backtrack(start + 1, current, size)
        
    backtrack(0, [], 0)
    return best

def construct():
    random.seed(42)
    # Start with a grid
    pts = [[x, y] for x in range(5) for y in range(4)]
    
    # Break 4-collinear
    for _ in range(1000):
        if not has_four_collinear(pts):
            break
        # Find a 4-collinear set
        quad_found = None
        for quad in itertools.combinations(pts, 4):
            if is_collinear(quad[0], quad[1], quad[2]) and is_collinear(quad[0], quad[1], quad[3]):
                quad_found = quad
                break
        if not quad_found:
            break
        # Move one point randomly
        idx = random.choice([0, 1, 2, 3])
        p = quad_found[idx]
        # Find index in pts
        p_idx = pts.index(p)
        # Move to random location
        new_x = random.randint(-10, 10)
        new_y = random.randint(-10, 10)
        # Ensure distinct
        while [new_x, new_y] in pts:
            new_x = random.randint(-10, 10)
            new_y = random.randint(-10, 10)
        pts[p_idx] = [new_x, new_y]
        
    # Now try to improve score by increasing 3-collinear
    # This is a heuristic: move a point to align with two others
    current_score = max_3_free_subset(pts)
    for _ in range(500):
        # Pick a random point
        i = random.randint(0, 19)
        p = pts[i]
        # Pick two other points
        j, k = random.sample([x for x in range(20) if x != i], 2)
        p2, p3 = pts[j], pts[k]
        # Find a point on the line p2-p3
        # Direction
        dx = p3[0] - p2[0]
        dy = p3[1] - p2[1]
        # Extend
        # Try p2 + t * (dx, dy) for t in -2..2 (excluding 0,1)
        candidates = []
        for t in [-3, -2, -1, 2, 3, 4]:
            nx = p2[0] + t * dx
            ny = p2[1] + t * dy
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                if [nx, ny] not in pts:
                    candidates.append([nx, ny])
        if candidates:
            new_p = random.choice(candidates)
            # Check if moving p to new_p creates 4-collinear
            old_p = pts[i]
            pts[i] = new_p
            if not has_four_collinear(pts):
                new_score = max_3_free_subset(pts)
                if new_score < current_score:
                    current_score = new_score
                else:
                    pts[i] = old_p
            else:
                pts[i] = old_p
                
    return pts
