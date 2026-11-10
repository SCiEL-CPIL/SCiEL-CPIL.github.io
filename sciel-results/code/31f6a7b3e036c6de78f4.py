import itertools
import random
import time

def get_collinear_triples(points):
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Check collinearity: (x2-x1)*(y3-y1) == (x3-x1)*(y2-y1)
                if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]):
                    triples.append((i, j, k))
    return triples

def has_four_collinear(points):
    n = len(points)
    # Check all 4-subsets
    # This is O(N^4) which is 20^4 = 160,000. Fast.
    for combo in itertools.combinations(range(n), 4):
        p1, p2, p3, p4 = [points[i] for i in combo]
        # Check if any 3 of the 4 are collinear? No, "4 collinear" means all 4 on one line.
        # If 3 are collinear, the 4th must also be on that line.
        # Check if p1, p2, p3 are collinear. If so, check if p4 is on the same line.
        # Actually, simpler: check if all 4 are collinear.
        # A set of 4 points is collinear if the area of any triangle formed by 3 of them is 0, 
        # AND the 4th is on the line.
        # If p1,p2,p3 are not collinear, they can't be part of a 4-collinear set.
        # So:
        if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]):
             # p1,p2,p3 collinear. Check p4.
             if (p4[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p4[1]-p1[1]):
                 return True
        # Also check other combinations? 
        # If p1,p2,p4 are collinear?
        # If p1,p2,p3 not collinear, p1,p2,p4 might be.
        # To be safe, check if any 3 are collinear, then check the 4th.
        # If no 3 are collinear, 4 cannot be.
        # So checking (1,2,3) and (1,2,4) is sufficient?
        # If 1,2,3 collinear and 4 on line -> 4 collinear.
        # If 1,2,4 collinear and 3 on line -> 4 collinear.
        # If 1,3,4 collinear... 
        # Actually, if 4 points are collinear, ANY 3 of them are collinear.
        # So if (1,2,3) are not collinear, they cannot be part of a 4-collinear set with 4?
        # Yes.
        # So we just need to check:
        # Are 1,2,3 collinear? If yes, is 4 on the line?
        # Are 1,2,4 collinear? If yes, is 3 on the line?
        # This covers all cases where 4 are collinear.
        if (p2[0]-p1[0])*(p4[1]-p1[1]) == (p4[0]-p1[0])*(p2[1]-p1[1]):
             if (p3[0]-p1[0])*(p4[1]-p1[1]) == (p4[0]-p1[0])*(p3[1]-p1[1]):
                 return True
    return False

def calc_score(points):
    n = len(points)
    triples = get_collinear_triples(points)
    # Precompute for each point, the list of pairs it forms a triple with
    # triples_by_point[i] = list of (j, k)
    triples_by_point = [[] for _ in range(n)]
    for i, j, k in triples:
        triples_by_point[i].append((j, k))
        triples_by_point[j].append((i, k))
        triples_by_point[k].append((i, j))
        
    best = 0
    # B&B
    def dfs(idx, selected_count, selected_set):
        nonlocal best
        if idx == n:
            if selected_count > best:
                best = selected_count
            return
        
        # Upper bound
        if selected_count + (n - idx) <= best:
            return
            
        # Option 1: Exclude point idx
        dfs(idx + 1, selected_count, selected_set)
        
        # Option 2: Include point idx
        # Check if adding idx creates a collinear triple with two points in selected_set
        valid = True
        for j, k in triples_by_point[idx]:
            if j in selected_set and k in selected_set:
                valid = False
                break
                
        if valid:
            selected_set.add(idx)
            dfs(idx + 1, selected_count + 1, selected_set)
            selected_set.remove(idx)
            
    dfs(0, 0, set())
    return best

def construct():
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    best_score = 20
    best_points = [[x, x * x] for x in range(-10, 10)]
    
    def try_improve(points, time_limit):
        nonlocal best_score, best_points
        current_points = [p[:] for p in points]
        current_score = calc_score(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        while time.time() - start_time < time_limit:
            # Pick a random point to move
            i = random.randint(0, 19)
            old_p = current_points[i]
            
            # Generate a few random new positions
            improved = False
            for _ in range(10):
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                new_p = [new_x, new_y]
                
                # Check distinctness
                if any(new_p == current_points[j] for j in range(20) if j != i):
                    continue
                    
                # Temporarily replace
                current_points[i] = new_p
                
                # Check validity (no 4 collinear)
                if has_four_collinear(current_points):
                    current_points[i] = old_p
                    continue
                    
                # Check distinctness again (just in case, though checked above)
                
                # Calculate new score
                new_score = calc_score(current_points)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    current_points[i] = old_p
                    
            if not improved:
                # Stuck? Maybe try a bigger jump or restart?
                # For now, just continue.
                pass
                
        return current_points

    # Initial strategies
    # 1. Random
    random_points = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
    # Ensure distinct
    while len(set(map(tuple, random_points))) < 20:
        random_points[random.randint(0,19)] = [random.randint(-100, 100), random.randint(-100, 100)]
    
    # Ensure no 4 collinear for random start?
    # It's likely valid. If not, fix it.
    if has_four_collinear(random_points):
         # Simple fix: move points until valid
         for _ in range(100):
             if not has_four_collinear(random_points): break
             i = random.randint(0, 19)
             random_points[i] = [random.randint(-100, 100), random.randint(-100, 100)]
             # Check distinct
             if any(random_points[i] == random_points[j] for j in range(20) if j != i):
                 continue

    if not has_four_collinear(random_points):
        try_improve(random_points, start_time + 30)

    # 2. Grid based
    grid_points = []
    for x in range(-5, 5):
        for y in range(-5, 5):
            grid_points.append([x, y])
    # Perturb
    for i in range(20):
        grid_points[i] = [grid_points[i][0] + random.randint(-2, 2), grid_points[i][1] + random.randint(-2, 2)]
    # Fix validity
    if has_four_collinear(grid_points):
         for _ in range(100):
             if not has_four_collinear(grid_points): break
             i = random.randint(0, 19)
             grid_points[i] = [grid_points[i][0] + random.randint(-2, 2), grid_points[i][1] + random.randint(-2, 2)]

    if not has_four_collinear(grid_points):
        try_improve(grid_points, start_time + 60)

    # 3. Parabola perturb
    para_points = [[x, x*x + random.randint(-2, 2)] for x in range(-10, 10)]
    if has_four_collinear(para_points):
         for _ in range(100):
             if not has_four_collinear(para_points): break
             i = random.randint(0, 19)
             para_points[i] = [para_points[i][0], para_points[i][1] + random.randint(-2, 2)]

    if not has_four_collinear(para_points):
        try_improve(para_points, start_time + 90)

    return best_points

# EVOLVE-BLOCK-END
