import time
import random
import math
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin within 60s limit

    def make_grid(w, h, ox=0, oy=0):
        pts = []
        for x in range(ox, ox + w):
            for y in range(oy, oy + h):
                pts.append((x, y))
        return pts

    def evaluate(points):
        areas = []
        n = len(points)
        for i in range(n):
            xi, yi = points[i]
            for j in range(i+1, n):
                xj, yj = points[j]
                for k in range(j+1, n):
                    xk, yk = points[k]
                    area = abs(xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj))
                    if area > 0:
                        areas.append(area)
        if not areas:
            return 0
        counts = Counter(areas)
        return max(counts.values())

    # Generate diverse initial candidates
    candidates = []
    
    # Various grid shapes centered
    for w, h in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15), (30, 1), (1, 30)]:
        ox = -w // 2
        oy = -h // 2
        candidates.append(make_grid(w, h, ox, oy))
        
    # Modified grids (removing corners to maintain area frequency)
    full_7x5 = [(x, y) for x in range(7) for y in range(5)]
    to_remove = [(0, 0), (6, 0), (0, 4), (6, 4), (0, 2)]
    grid_7x5_mod = [p for p in full_7x5 if p not in to_remove]
    candidates.append(grid_7x5_mod)
    
    full_8x4 = [(x, y) for x in range(8) for y in range(4)]
    to_remove_8x4 = [(0, 0), (7, 3)]
    grid_8x4_mod = [p for p in full_8x4 if p not in to_remove_8x4]
    candidates.append(grid_8x4_mod)

    # Number-theoretic construction: Generalized Arithmetic Progression
    # Points are of the form P0 + i*v1 + j*v2 for i in [0, m), j in [0, n)
    # This creates a parallelogram lattice. The areas of triangles formed by 
    # points in such a lattice are multiples of the area of the fundamental parallelogram.
    # Specifically, if v1=(a,b) and v2=(c,d), the doubled area of the fundamental triangle is |ad-bc|.
    # By choosing v1 and v2 such that the determinant is 1 (primitive vectors), we maximize the 
    # frequency of small areas (area=1, 2, etc).
    
    def gen_lattice(m, n, v1, v2, ox=0, oy=0):
        pts = []
        for i in range(m):
            for j in range(n):
                x = ox + i*v1[0] + j*v2[0]
                y = oy + i*v1[1] + j*v2[1]
                pts.append((x, y))
        return pts

    # Standard grid is v1=(1,0), v2=(0,1). Det = 1.
    # Other primitive vectors with det 1:
    # v1=(1,0), v2=(1,1) -> shear
    # v1=(1,0), v2=(0,1) -> grid
    # v1=(1,1), v2=(1,0) -> shear
    # v1=(2,1), v2=(1,0) -> det -1
    # v1=(1,2), v2=(0,1) -> det 1
    
    lattices = [
        (6, 5, (1,0), (0,1), -3, -2),
        (6, 5, (1,0), (1,1), -3, -2),
        (6, 5, (1,1), (1,0), -3, -2),
        (6, 5, (2,1), (1,0), -3, -2),
        (6, 5, (1,2), (0,1), -3, -2),
        (5, 6, (1,0), (1,1), -2, -3),
        (10, 3, (1,0), (1,1), -5, -1),
        (10, 3, (1,1), (1,0), -5, -1),
    ]
    
    for m, n, v1, v2, ox, oy in lattices:
        pts = gen_lattice(m, n, v1, v2, ox, oy)
        # Check bounds and distinctness
        if len(set(pts)) == 30 and all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            candidates.append(pts)

    # Some other structured patterns
    for radius in [5, 6]:
        pts = [(x, y) for x in range(-radius, radius+1) for y in range(-radius, radius+1) 
               if abs(x) + abs(y) <= radius]
        if len(pts) >= 30:
            candidates.append(pts[:30])
            
    # Random clusters
    for _ in range(10):
        cx, cy = random.randint(-20, 20), random.randint(-20, 20)
        pts = set()
        while len(pts) < 30:
            x = cx + random.randint(-5, 5)
            y = cy + random.randint(-5, 5)
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
        candidates.append(list(pts))
        
    # Fully random
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        candidates.append(list(pts))

    best_points = None
    best_score = -1
    
    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        if len(pts) != 30:
            continue
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]

    if best_points is None:
        best_points = make_grid(6, 5)
        best_score = evaluate(best_points)

    current_points = best_points[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    alpha = 0.9995
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        new_points = current_points[:]
        idx = random.randint(0, 29)
        x, y = new_points[idx]
        
        r = random.random()
        if r < 0.33:
            dx = random.choice([-3, -2, -1, 1, 2, 3])
            dy = 0
        elif r < 0.66:
            dx = 0
            dy = random.choice([-3, -2, -1, 1, 2, 3])
        else:
            dx = random.choice([-3, -2, -1, 1, 2, 3])
            dy = random.choice([-3, -2, -1, 1, 2, 3])
            
        new_x = x + dx
        new_y = y + dy
        
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        new_pt = (new_x, new_y)
        if new_pt in current_points:
            continue
                
        new_points[idx] = new_pt
        
        new_score = evaluate(new_points)
        
        if new_score >= current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            delta = current_score - new_score
            if random.random() < math.exp(-delta / T):
                current_points = new_points
                current_score = new_score
                
        T *= alpha
        if T < T_min:
            T = T_min

    return best_points

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin within 60s limit

    def make_grid(w, h, ox=0, oy=0):
        pts = []
        for x in range(ox, ox + w):
            for y in range(oy, oy + h):
                pts.append((x, y))
        return pts

    def evaluate(points):
        areas = []
        n = len(points)
        for i in range(n):
            xi, yi = points[i]
            for j in range(i+1, n):
                xj, yj = points[j]
                for k in range(j+1, n):
                    xk, yk = points[k]
                    area = abs(xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj))
                    if area > 0:
                        areas.append(area)
        if not areas:
            return 0
        counts = Counter(areas)
        return max(counts.values())

    # Generate diverse initial candidates
    candidates = []
    
    # Various grid shapes centered
    for w, h in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15), (30, 1), (1, 30)]:
        ox = -w // 2
        oy = -h // 2
        candidates.append(make_grid(w, h, ox, oy))
        
    # Modified grids (removing corners to maintain area frequency)
    full_7x5 = [(x, y) for x in range(7) for y in range(5)]
    to_remove = [(0, 0), (6, 0), (0, 4), (6, 4), (0, 2)]
    grid_7x5_mod = [p for p in full_7x5 if p not in to_remove]
    candidates.append(grid_7x5_mod)
    
    full_8x4 = [(x, y) for x in range(8) for y in range(4)]
    to_remove_8x4 = [(0, 0), (7, 3)]
    grid_8x4_mod = [p for p in full_8x4 if p not in to_remove_8x4]
    candidates.append(grid_8x4_mod)

    # Number-theoretic construction: Generalized Arithmetic Progression
    # Points are of the form P0 + i*v1 + j*v2 for i in [0, m), j in [0, n)
    # This creates a parallelogram lattice. The areas of triangles formed by 
    # points in such a lattice are multiples of the area of the fundamental parallelogram.
    # Specifically, if v1=(a,b) and v2=(c,d), the doubled area of the fundamental triangle is |ad-bc|.
    # By choosing v1 and v2 such that the determinant is 1 (primitive vectors), we maximize the 
    # frequency of small areas (area=1, 2, etc).
    
    def gen_lattice(m, n, v1, v2, ox=0, oy=0):
        pts = []
        for i in range(m):
            for j in range(n):
                x = ox + i*v1[0] + j*v2[0]
                y = oy + i*v1[1] + j*v2[1]
                pts.append((x, y))
        return pts

    # Standard grid is v1=(1,0), v2=(0,1). Det = 1.
    # Other primitive vectors with det 1:
    # v1=(1,0), v2=(1,1) -> shear
    # v1=(1,0), v2=(0,1) -> grid
    # v1=(1,1), v2=(1,0) -> shear
    # v1=(2,1), v2=(1,0) -> det -1
    # v1=(1,2), v2=(0,1) -> det 1
    
    lattices = [
        (6, 5, (1,0), (0,1), -3, -2),
        (6, 5, (1,0), (1,1), -3, -2),
        (6, 5, (1,1), (1,0), -3, -2),
        (6, 5, (2,1), (1,0), -3, -2),
        (6, 5, (1,2), (0,1), -3, -2),
        (5, 6, (1,0), (1,1), -2, -3),
        (10, 3, (1,0), (1,1), -5, -1),
        (10, 3, (1,1), (1,0), -5, -1),
    ]
    
    for m, n, v1, v2, ox, oy in lattices:
        pts = gen_lattice(m, n, v1, v2, ox, oy)
        # Check bounds and distinctness
        if len(set(pts)) == 30 and all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            candidates.append(pts)

    # Some other structured patterns
    for radius in [5, 6]:
        pts = [(x, y) for x in range(-radius, radius+1) for y in range(-radius, radius+1) 
               if abs(x) + abs(y) <= radius]
        if len(pts) >= 30:
            candidates.append(pts[:30])
            
    # Random clusters
    for _ in range(10):
        cx, cy = random.randint(-20, 20), random.randint(-20, 20)
        pts = set()
        while len(pts) < 30:
            x = cx + random.randint(-5, 5)
            y = cy + random.randint(-5, 5)
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
        candidates.append(list(pts))
        
    # Fully random
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        candidates.append(list(pts))

    best_points = None
    best_score = -1
    
    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        if len(pts) != 30:
            continue
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]

    if best_points is None:
        best_points = make_grid(6, 5)
        best_score = evaluate(best_points)

    current_points = best_points[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    alpha = 0.9995
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        new_points = current_points[:]
        idx = random.randint(0, 29)
        x, y = new_points[idx]
        
        r = random.random()
        if r < 0.33:
            dx = random.choice([-3, -2, -1, 1, 2, 3])
            dy = 0
        elif r < 0.66:
            dx = 0
            dy = random.choice([-3, -2, -1, 1, 2, 3])
        else:
            dx = random.choice([-3, -2, -1, 1, 2, 3])
            dy = random.choice([-3, -2, -1, 1, 2, 3])
            
        new_x = x + dx
        new_y = y + dy
        
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        new_pt = (new_x, new_y)
        if new_pt in current_points:
            continue
                
        new_points[idx] = new_pt
        
        new_score = evaluate(new_points)
        
        if new_score >= current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            delta = current_score - new_score
            if random.random() < math.exp(-delta / T):
                current_points = new_points
                current_score = new_score
                
        T *= alpha
        if T < T_min:
            T = T_min

    return best_points
# EVOLVE-BLOCK-END
