import random
import itertools
import time
import math

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
    n = len(points)
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
        if size + (n - start) <= best:
            return
        valid = True
        for i in current:
            if coll[i][start]:
                valid = False
                break
        if valid:
            backtrack(start + 1, current + [start], size + 1)
        backtrack(start + 1, current, size)
        
    backtrack(0, [], 0)
    return best

def find_best_move(pts, i):
    """Find the best position for point i that maximizes 3-collinearities without creating 4-collinear"""
    best_pos = pts[i]
    best_triples = -1
    others = [pts[j] for j in range(20) if j != i]
    
    # Try all pairs of other points - place on the line through them
    for a_idx in range(19):
        for b_idx in range(a_idx+1, 19):
            p2 = others[a_idx]
            p3 = others[b_idx]
            dx = p3[0] - p2[0]
            dy = p3[1] - p2[1]
            # Try integer positions on this line
            for t in range(-5, 6):
                if t == 0 or t == 1:
                    continue
                nx = p2[0] + t * dx
                ny = p2[1] + t * dy
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    if [nx, ny] not in pts:
                        # Check no 4-collinear
                        test_pts = pts[:]
                        test_pts[i] = [nx, ny]
                        if not has_four_collinear(test_pts):
                            triples = count_three_collinear(test_pts)
                            if triples > best_triples:
                                best_triples = triples
                                best_pos = [nx, ny]
    return best_pos

def construct():
    random.seed(42)
    # Start with a 5x4 grid
    pts = [[x, y] for x in range(5) for y in range(4)]
    
    # Phase 1: Break all 4-collinearities
    for _ in range(5000):
        quad_found = None
        for quad in itertools.combinations(pts, 4):
            if is_collinear(quad[0], quad[1], quad[2]) and is_collinear(quad[0], quad[1], quad[3]):
                quad_found = quad
                break
        if not quad_found:
            break
        # Move one point from the quad to break it
        idx = random.choice([0, 1, 2, 3])
        p = quad_found[idx]
        p_idx = pts.index(p)
        # Try several random positions
        moved = False
        for _ in range(50):
            new_x = random.randint(-20, 20)
            new_y = random.randint(-20, 20)
            if [new_x, new_y] not in pts:
                old_p = pts[p_idx]
                pts[p_idx] = [new_x, new_y]
                if not has_four_collinear(pts):
                    moved = True
                    break
                pts[p_idx] = old_p
        if not moved:
            # Last resort: wider range
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            while [new_x, new_y] in pts:
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
            pts[p_idx] = [new_x, new_y]
    
    # Phase 2: Maximize 3-collinearities via local search
    current_score = max_3_free_subset(pts)
    current_triples = count_three_collinear(pts)
    
    improved = True
    while improved:
        improved = False
        for i in range(20):
            new_pos = find_best_move(pts, i)
            if new_pos != pts[i]:
                old_p = pts[i]
                pts[i] = new_pos
                new_triples = count_three_collinear(pts)
                new_score = max_3_free_subset(pts)
                if new_triples > current_triples or (new_triples == current_triples and new_score < current_score):
                    current_triples = new_triples
                    current_score = new_score
                    improved = True
                else:
                    pts[i] = old_p
    
    # Phase 3: Final local search with random restarts
    for _ in range(100):
        i = random.randint(0, 19)
        old_p = pts[i]
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        if [new_x, new_y] not in pts:
            pts[i] = [new_x, new_y]
            if not has_four_collinear(pts):
                new_triples = count_three_collinear(pts)
                new_score = max_3_free_subset(pts)
                if new_triples > current_triples or (new_triples == current_triples and new_score < current_score):
                    current_triples = new_triples
                    current_score = new_score
                else:
                    pts[i] = old_p
            else:
                pts[i] = old_p
    
    return pts
