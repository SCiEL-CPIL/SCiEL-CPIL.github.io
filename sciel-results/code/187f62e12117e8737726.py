import random
import math
import time

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Helper to check if a number is a perfect square
    def is_perfect_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    # Check if 3 points are collinear
    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        area2 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        return area2 == 0

    # Check if 4 points are concyclic
    def is_concyclic(p1, p2, p3, p4):
        rows = []
        for p in (p1, p2, p3, p4):
            x, y = p
            rows.append([x*x + y*y, x, y, 1])
        def det4(m):
            return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - 
                             m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1]) -
                    m[0][1]*(m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - 
                             m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0]) +
                    m[0][2]*(m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0]) -
                    m[0][3]*(m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0]))
        return det4(rows) == 0

    def is_valid_set(points, D):
        n = len(points)
        if len(set(points)) != n:
            return False
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + D*dy*dy
                if not is_perfect_square(d2):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def fitness(points, D):
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + D*dy*dy
                if is_perfect_square(d2):
                    count += 1
        # Penalize collinearities
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        count -= 2
        # Penalize concyclicities
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            count -= 2
        return count

    # Seed solution (5 points) with D=1
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    seed_D = 1
    
    found_solution = None
    found_D = None
    
    # Try multiple D values with structured search
    D_candidates = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 21, 22, 23, 30, 31, 33, 34, 35, 38, 39, 42, 46, 47, 51, 55, 57, 58, 62, 65, 69, 70, 71, 74, 77, 78, 82, 83, 86, 87, 91, 93, 94, 95]
    
    for D in D_candidates:
        if time.time() > deadline - 10.0:
            break
        
        if D == 1:
            # Use seed as starting point
            init_points = seed_points + [(random.randint(-100, 100), random.randint(-100, 100))]
        else:
            # Random initialization for other D values
            box = 50
            init_points = [(random.randint(-box, box), random.randint(-box, box)) for _ in range(6)]
            while len(set(init_points)) < 6:
                idx = random.randint(0, 5)
                init_points[idx] = (random.randint(-box, box), random.randint(-box, box))
        
        # Simulated annealing with larger steps
        points = list(init_points)
        current_fitness = fitness(points, D)
        best_points = list(points)
        best_fitness = current_fitness
        temp = 3.0
        
        for step in range(500):
            if time.time() > deadline - 5.0:
                break
            
            idx = random.randint(0, 5)
            old_x, old_y = points[idx]
            
            # Variable step size
            step_mag = random.choice([1, 2, 3, 5, 8, 13, 21])
            dx = random.randint(-step_mag, step_mag)
            dy = random.randint(-step_mag, step_mag)
            if dx == 0 and dy == 0:
                continue
            new_x = old_x + dx
            new_y = old_y + dy
            
            points[idx] = (new_x, new_y)
            new_fitness = fitness(points, D)
            
            delta = new_fitness - current_fitness
            if delta > 0 or random.random() < math.exp(delta / max(temp, 0.01)):
                current_fitness = new_fitness
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_points = list(points)
                    if best_fitness >= 15 and is_valid_set(points, D):
                        found_solution = points
                        found_D = D
                        break
            else:
                points[idx] = (old_x, old_y)
            
            temp = max(0.01, temp * 0.97)
        
        if found_solution:
            break
    
    if found_solution:
        pts = found_solution
        D_val = found_D
    else:
        # Fallback to seed
        pts = seed_points
        D_val = seed_D
    
    # Final validation
    if not is_valid_set(pts, D_val):
        pts = seed_points
        D_val = 1
    
    return {'D': D_val, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
# EVOLVE-BLOCK-END
