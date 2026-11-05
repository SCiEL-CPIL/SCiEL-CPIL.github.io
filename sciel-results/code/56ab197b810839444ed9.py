import time
import random
import math
from itertools import combinations

DEADLINE = 115.0

def normalize_slope(dx, dy):
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 1)
    if dy == 0:
        return (1, 0)
    g = math.gcd(dx, dy)
    if g < 0:
        g = -g
    dx, dy = dx // g, dy // g
    if dx < 0:
        dx, dy = -dx, -dy
    return (dx, dy)

def has_four_collinear(points):
    n = len(points)
    for i in range(n):
        slopes = {}
        xi, yi = points[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            slope = normalize_slope(dx, dy)
            if slope is None:
                continue
            if slope in slopes:
                slopes[slope] += 1
                if slopes[slope] >= 3:
                    return True
            else:
                slopes[slope] = 1
    return False

def compute_conflict_graph(points):
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            slope = normalize_slope(dx, dy)
            if slope is None:
                continue
            for k in range(j + 1, n):
                xk, yk = points[k]
                dxk = xk - xi
                dyk = yk - yi
                slope_k = normalize_slope(dxk, dyk)
                if slope_k == slope:
                    triples.append((i, j, k))
    
    conflict = [[set() for _ in range(n)] for _ in range(n)]
    for i, j, k in triples:
        conflict[i][j].add(k)
        conflict[i][k].add(j)
        conflict[j][i].add(k)
        conflict[j][k].add(i)
        conflict[k][i].add(j)
        conflict[k][j].add(i)
        
    return conflict, len(triples)

def max_no_three_collinear(points):
    n = len(points)
    conflict, num_triples = compute_conflict_graph(points)
    
    if num_triples == 0:
        return n
        
    best = [0]
    
    def search(available, current, forbidden):
        current_size = bin(current).count('1')
        if available == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        if current_size + bin(available).count('1') <= best[0]:
            return
            
        v = (available & -available).bit_length() - 1
        bit_v = 1 << v
        
        search(available ^ bit_v, current, forbidden)
        
        new_current = current | bit_v
        new_forbidden = forbidden
        temp_current = current
        while temp_current:
            u_bit = temp_current & -temp_current
            u = u_bit.bit_length() - 1
            for k in conflict[v][u]:
                new_forbidden |= (1 << k)
            temp_current ^= u_bit
            
        newly_forbidden = new_forbidden & available
        new_available = (available ^ bit_v) ^ newly_forbidden
        
        search(new_available, new_current, new_forbidden)
        
    full_mask = (1 << n) - 1
    search(full_mask, 0, 0)
    return best[0]

def generate_initial_points(seed):
    random.seed(seed)
    points = []
    used = set()
    while len(points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in used:
            used.add((x, y))
            points.append((x, y))
    
    max_attempts = 1000
    attempts = 0
    while has_four_collinear(points) and attempts < max_attempts:
        idx = random.randint(0, 19)
        old_p = points[idx]
        found = False
        for _ in range(100):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            new_p = (x, y)
            if new_p == old_p or new_p in used:
                continue
            points[idx] = new_p
            if not has_four_collinear(points):
                used.discard(old_p)
                used.add(new_p)
                found = True
                break
            points[idx] = old_p
        if not found:
            points[idx] = old_p
        attempts += 1
    
    return points, used

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_score = 20
    best_points = None
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000,
             7000, 8000, 9000, 10000, 11111, 22222, 33333, 44444, 55555, 66666,
             77777, 88888, 99999, 12345, 67890, 13579, 24680, 35791, 46802, 57913]
    
    for seed in seeds:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        points, used = generate_initial_points(seed)
        if best_points is None:
            best_points = points[:]
        
        current_score = max_no_three_collinear(points)
        if current_score < best_score:
            best_score = current_score
            best_points = points[:]
            
        if best_score <= 10:
            break
        
        # Local search
        no_improve_count = 0
        max_no_improve = 30
        
        while time.time() - start_time < DEADLINE - 3 and no_improve_count < max_no_improve:
            move_type = rng.random()
            improved = False
            
            if move_type < 0.5:
                # Single point move
                idx = rng.randint(0, 19)
                old_p = points[idx]
                moved = False
                for _ in range(30):
                    x = rng.randint(-100, 100)
                    y = rng.randint(-100, 100)
                    new_p = (x, y)
                    if new_p == old_p or new_p in used:
                        continue
                    points[idx] = new_p
                    if not has_four_collinear(points):
                        moved = True
                        break
                    points[idx] = old_p
                if moved:
                    new_score = max_no_three_collinear(points)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        used.discard(old_p)
                        used.add(points[idx])
                        if current_score < best_score:
                            best_score = current_score
                            best_points = points[:]
                            no_improve_count = 0
                    else:
                        points[idx] = old_p
                else:
                    no_improve_count += 1
            else:
                # Pair swap
                i, j = rng.sample(range(20), 2)
                pi, pj = points[i], points[j]
                moved = False
                for _ in range(30):
                    xi = rng.randint(-100, 100)
                    yi = rng.randint(-100, 100)
                    xj = rng.randint(-100, 100)
                    yj = rng.randint(-100, 100)
                    new_pi = (xi, yi)
                    new_pj = (xj, yj)
                    if new_pi == pi or new_pj == pj or new_pi in used or new_pj in used or new_pi == new_pj:
                        continue
                    points[i] = new_pi
                    points[j] = new_pj
                    if not has_four_collinear(points):
                        moved = True
                        break
                    points[i] = pi
                    points[j] = pj
                if moved:
                    new_score = max_no_three_collinear(points)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        used.discard(pi)
                        used.discard(pj)
                        used.add(points[i])
                        used.add(points[j])
                        if current_score < best_score:
                            best_score = current_score
                            best_points = points[:]
                            no_improve_count = 0
                    else:
                        points[i] = pi
                        points[j] = pj
                else:
                    no_improve_count += 1
            
            if not improved:
                no_improve_count += 1
        
        if best_score <= 12:
            break
    
    if best_points is None:
        points, _ = generate_initial_points(42)
        best_points = points
    
    return [[p[0], p[1]] for p in best_points]
