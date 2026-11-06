import math
import random
import time
from collections import defaultdict

def line_hash(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return ('v', x1)
    if y1 == y2:
        return ('h', y1)
    dy = y2 - y1
    dx = x2 - x1
    g = math.gcd(abs(dx), abs(dy))
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0:
        dx_n = -dx_n
        dy_n = -dy_n
    elif dx_n == 0 and dy_n < 0:
        dy_n = -dy_n
    a = dy_n
    b = -dx_n
    c = dx_n * y1 - dy_n * x1
    g = math.gcd(abs(a), abs(b), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def evaluate(points):
    n = len(points)
    if n != 16:
        return -1
    if len(set(map(tuple, points))) != 16:
        return -1
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = line_hash(points[i], points[j])
            line_counts[key] += 1
    
    score = 0
    for key, count in line_counts.items():
        disc = 1 + 8 * count
        sqrt_disc = int(math.isqrt(disc))
        if sqrt_disc * sqrt_disc != disc:
            return -1
        k = (1 + sqrt_disc) // 2
        if k >= 5:
            return -1
        if k == 4:
            score += 1
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial candidate: 4x4 grid
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = evaluate(initial_points)
    best_points = [p[:] for p in initial_points]
    
    random.seed(42)
    
    def local_search(points, max_iters=1000):
        """
        Perform local search with simulated annealing.
        Returns (best_points_found, best_score_found).
        """
        current_points = [p[:] for p in points]
        current_score = evaluate(current_points)
        if current_score < 0:
            return current_points, current_score
        
        best_local = [p[:] for p in current_points]
        best_local_score = current_score
        
        temp = 1.0
        cooling = 0.995
        no_improve = 0
        max_no_improve = 3000
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 15)
            old_pt = current_points[idx][:]
            
            # Random move strategy: mix of small and large steps
            if random.random() < 0.6:
                step = random.randint(1, 4)
            else:
                step = random.randint(4, 15)
                
            new_x = old_pt[0] + random.randint(-step, step)
            new_y = old_pt[1] + random.randint(-step, step)
            
            if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                continue
                
            new_pt = [new_x, new_y]
            
            # Check distinctness
            is_distinct = True
            for i, p in enumerate(current_points):
                if i != idx and p == new_pt:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
                
            current_points[idx] = new_pt
            new_score = evaluate(current_points)
            
            if new_score == -1:
                current_points[idx] = old_pt
                no_improve += 1
                continue
            
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                no_improve = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local = [p[:] for p in current_points]
                    temp = 1.0 # Reset temperature on improvement
            else:
                # Accept if random() < exp(delta / temp)
                if delta == 0:
                    current_score = new_score
                    no_improve += 1
                else:
                    prob = math.exp(delta / temp)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        current_points[idx] = old_pt
                    no_improve += 1
            
            temp *= cooling
            if temp < 0.005:
                temp = 0.005
                
            if no_improve > max_no_improve:
                break
                
        return best_local, best_local_score

    # Phase 1: Optimize the initial grid
    pts, score = local_search(initial_points, max_iters=2000)
    if score > best_score:
        best_score = score
        best_points = [p[:] for p in pts]
        
    if best_score >= 15:
        return best_points

    # Phase 2: Random restarts with diverse initial configurations
    # We try to create configurations that are "close" to grids but perturbed,
    # or completely random, to escape local optima.
    
    restart_attempts = 0
    max_restarts = 20
    
    while time.time() < deadline and restart_attempts < max_restarts:
        restart_attempts += 1
        
        # Generate a diverse starting point
        mode = random.randint(0, 2)
        
        if mode == 0:
            # Perturbed grid
            pts = [p[:] for p in initial_points]
            for i in range(16):
                # Randomly shift some points
                if random.random() < 0.5:
                    pts[i] = [pts[i][0] + random.randint(-5, 5), pts[i][1] + random.randint(-5, 5)]
        elif mode == 1:
            # Random points in a smaller box
            pts = []
            while len(pts) < 16:
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                if [x, y] not in pts:
                    pts.append([x, y])
        else:
            # Random points in a larger box
            pts = []
            while len(pts) < 16:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if [x, y] not in pts:
                    pts.append([x, y])
        
        # Validate initial point (bounds and distinctness are handled in generation, but check score)
        s = evaluate(pts)
        if s < 0:
            continue
            
        # Run local search on this restart
        # Allocate remaining time roughly evenly, but stop early if deadline is near
        remaining_time = deadline - time.time()
        # Heuristic: if lots of time left, do more iterations. Otherwise, fewer.
        iters = int(remaining_time * 50) # Rough estimate
        iters = max(200, min(iters, 3000))
        
        pts, score = local_search(pts, max_iters=iters)
        
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            
        if best_score >= 15:
            break

    return best_points
