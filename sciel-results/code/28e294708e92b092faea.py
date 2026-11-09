import random
import itertools
import time
import math
import sys

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

def construct():
    random.seed(42)
    # Initial configuration: Try to create many triples
    # Start with a grid-like structure but perturbed to avoid 4-collinear
    # Or start with random points and fix 4-collinearities
    
    # Strategy: 
    # 1. Generate 20 random points.
    # 2. Fix 4-collinearities by moving points.
    # 3. Use Simulated Annealing to maximize triples.
    
    def generate_initial_points():
        pts = []
        # Try to place points on a circle or specific curves to avoid 4-collinear initially?
        # Random is fine if we fix later.
        for _ in range(20):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if [x, y] not in pts:
                    pts.append([x, y])
                    break
        return pts

    def fix_four_collinear(pts):
        # Move points to break 4-collinearities
        for _ in range(1000):
            quad_found = None
            for quad in itertools.combinations(pts, 4):
                if is_collinear(quad[0], quad[1], quad[2]) and is_collinear(quad[0], quad[1], quad[3]):
                    quad_found = quad
                    break
            if not quad_found:
                return pts
            
            # Pick a random point from the quad to move
            idx = random.choice([0, 1, 2, 3])
            p = quad_found[idx]
            p_idx = pts.index(p)
            
            # Try to move it to a random valid spot
            moved = False
            for _ in range(100):
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                if [new_x, new_y] not in pts:
                    old_p = pts[p_idx]
                    pts[p_idx] = [new_x, new_y]
                    if not has_four_collinear(pts):
                        moved = True
                        break
                    pts[p_idx] = old_p
            if not moved:
                # Force a move to a very random spot, might create new 4-collinear but we'll fix in next iter
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                while [new_x, new_y] in pts:
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                pts[p_idx] = [new_x, new_y]
                
        return pts

    pts = generate_initial_points()
    pts = fix_four_collinear(pts)
    
    # Simulated Annealing
    current_triples = count_three_collinear(pts)
    best_triples = current_triples
    best_pts = pts[:]
    
    T = 10.0
    cooling = 0.99
    
    # Time limit 120s
    start_time = time.time()
    
    while time.time() - start_time < 100: # Leave 20s for final check
        # Pick a random point
        i = random.randint(0, 19)
        old_p = pts[i]
        
        # Generate candidate moves
        # 1. Random move
        # 2. Move to create a triple with two existing points
        candidates = []
        
        # Random move
        for _ in range(5):
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)
            if [nx, ny] not in pts:
                candidates.append([nx, ny])
        
        # Targeted moves: place on line through two other points
        others = [pts[j] for j in range(20) if j != i]
        for a_idx in range(19):
            for b_idx in range(a_idx + 1, 19):
                p2 = others[a_idx]
                p3 = others[b_idx]
                dx = p3[0] - p2[0]
                dy = p3[1] - p2[1]
                # Check if line is not degenerate
                if dx == 0 and dy == 0:
                    continue
                # Try positions on this line
                # We want to place p_i such that it forms a triple with p2, p3
                # So p_i must be on the line.
                # To avoid 4-collinear, the line should currently have < 3 points (excluding p_i if it was there, but p_i is being moved).
                # Actually, if the line already has 3 other points, placing p_i makes 4.
                # So we check the count of points on this line.
                line_pts = []
                for p in pts:
                    if is_collinear(p2, p3, p):
                        line_pts.append(p)
                
                # If len(line_pts) >= 3, we cannot place p_i on this line (would make 4 or 5)
                # Unless p_i is one of them? No, p_i is being moved away.
                # So if len(line_pts) >= 3, skip this line.
                if len(line_pts) >= 3:
                    continue
                
                # Find integer points on the line segment or extension
                # Parametric: p2 + t * (dx, dy)
                # We want small t to stay in bounds
                for t in range(-20, 21):
                    if t == 0 or t == 1: # p2 and p3
                        continue
                    nx = p2[0] + t * dx
                    ny = p2[1] + t * dy
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        if [nx, ny] not in pts:
                            candidates.append([nx, ny])
                            break # One candidate per line is enough
        
        if not candidates:
            T *= cooling
            continue
            
        new_p = random.choice(candidates)
        
        # Check validity: no 4 collinear
        pts[i] = new_p
        if has_four_collinear(pts):
            pts[i] = old_p
            continue
            
        new_triples = count_three_collinear(pts)
        delta = new_triples - current_triples
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_triples = new_triples
            if current_triples > best_triples:
                best_triples = current_triples
                best_pts = pts[:]
        else:
            pts[i] = old_p
            
        T *= cooling
        
    # Final check and return best
    # Re-evaluate best_pts to be sure
    return best_pts
