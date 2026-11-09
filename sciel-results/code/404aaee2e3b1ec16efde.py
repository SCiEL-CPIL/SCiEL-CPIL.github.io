import random
import itertools
import time
from math import gcd
from collections import defaultdict

def is_collinear(p1, p2, p3):
    return (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]) == 0)

def has_four_collinear(points):
    """Check if any 4 points are collinear"""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if is_collinear(points[i], points[j], points[k]):
                    # Check if any 4th point is also on this line
                    for l in range(k+1, n):
                        if is_collinear(points[i], points[j], points[l]):
                            return True
    return False

def max_3_free_subset(points):
    """Find the largest subset with no three collinear using branch and bound"""
    n = len(points)
    # Build collinearity graph: coll[i][j] = True if i,j,k are collinear for some k
    coll = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if is_collinear(points[i], points[j], points[k]):
                    coll[i][j] = coll[i][k] = coll[j][k] = True
    
    best = 0
    
    def backtrack(start, current_size, last_added):
        nonlocal best
        if current_size > best:
            best = current_size
        if start >= n:
            return
        if current_size + (n - start) <= best:
            return
        
        # Try adding point 'start'
        can_add = True
        if last_added >= 0:
            if coll[last_added][start]:
                can_add = False
        
        if can_add:
            backtrack(start + 1, current_size + 1, start)
        
        backtrack(start + 1, current_size, -1)
    
    backtrack(0, 0, -1)
    return best

def construct():
    random.seed(42)
    
    # Phase 1: Generate initial configuration with no 4-collinear
    # Use a perturbed grid approach
    base_points = []
    # Start with 20 points from a grid, then perturb to break 4-collinearities
    x_range = range(-8, 8)
    y_range = range(-8, 8)
    
    # Create a set of candidate points
    candidates = [[x, y] for x in x_range for y in y_range]
    random.shuffle(candidates)
    
    pts = []
    for p in candidates:
        if len(pts) >= 20:
            break
        test_pts = pts + [p]
        if not has_four_collinear(test_pts):
            pts.append(p)
    
    # If we couldn't get 20, try with wider range
    while len(pts) < 20:
        new_x = random.randint(-50, 50)
        new_y = random.randint(-50, 50)
        p = [new_x, new_y]
        if p not in pts:
            test_pts = pts + [p]
            if not has_four_collinear(test_pts):
                pts.append(p)
    
    # Phase 2: Local search to maximize 3-collinearities
    current_score = max_3_free_subset(pts)
    
    def count_three_collinear(points):
        count = 0
        for tri in itertools.combinations(points, 3):
            if is_collinear(tri[0], tri[1], tri[2]):
                count += 1
        return count
    
    current_triples = count_three_collinear(pts)
    
    improved = True
    max_iterations = 5000
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(20):
            old_pos = pts[i]
            
            # Try moving point i to various positions
            best_new_pos = old_pos
            best_new_triples = current_triples
            best_new_score = current_score
            
            # Try positions on lines through pairs of other points
            others = [pts[j] for j in range(20) if j != i]
            
            for a_idx in range(19):
                for b_idx in range(a_idx+1, 19):
                    p_a = others[a_idx]
                    p_b = others[b_idx]
                    dx = p_b[0] - p_a[0]
                    dy = p_b[1] - p_a[1]
                    
                    # Try integer positions along this line
                    g = gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    dx_n = dx // g
                    dy_n = dy // g
                    
                    for t in range(-15, 16):
                        if t == 0 or t == 1:
                            continue
                        nx = p_a[0] + t * dx_n
                        ny = p_a[1] + t * dy_n
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            new_p = [nx, ny]
                            if new_p not in pts:
                                test_pts = pts[:]
                                test_pts[i] = new_p
                                if not has_four_collinear(test_pts):
                                    triples = count_three_collinear(test_pts)
                                    if triples > best_new_triples or (triples == best_new_triples):
                                        best_new_triples = triples
                                        best_new_pos = new_p
            
            # Also try random positions
            for _ in range(20):
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                new_p = [nx, ny]
                if new_p not in pts:
                    test_pts = pts[:]
                    test_pts[i] = new_p
                    if not has_four_collinear(test_pts):
                        triples = count_three_collinear(test_pts)
                        if triples > best_new_triples:
                            best_new_triples = triples
                            best_new_pos = new_p
            
            if best_new_pos != old_pos and best_new_triples > current_triples:
                pts[i] = best_new_pos
                current_triples = best_new_triples
                current_score = max_3_free_subset(pts)
                improved = True
    
    # Phase 3: Simulated annealing style refinement
    best_pts = pts[:]
    best_score = current_score
    
    for restart in range(30):
        # Start from best configuration
        pts = best_pts[:]
        
        for step in range(200):
            i = random.randint(0, 19)
            old_pos = pts[i]
            
            # Random perturbation
            nx = random.randint(-50, 50)
            ny = random.randint(-50, 50)
            new_p = [nx, ny]
            
            if new_p not in pts:
                pts[i] = new_p
                if not has_four_collinear(pts):
                    new_triples = count_three_collinear(pts)
                    new_score = max_3_free_subset(pts)
                    
                    # Accept if better or with some probability
                    if new_triples > current_triples or (new_triples == current_triples and new_score < best_score):
                        current_triples = new_triples
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_pts = pts[:]
                    else:
                        pts[i] = old_pos
                else:
                    pts[i] = old_pos
            else:
                pts[i] = old_pos
    
    return best_pts
