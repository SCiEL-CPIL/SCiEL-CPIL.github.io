import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 145  # Leave 5s margin for evaluator overhead
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def count_collinear_triples(points):
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                for k in range(j+1, n):
                    p3 = points[k]
                    if dx * (p3[1] - p1[1]) - dy * (p3[0] - p1[0]) == 0:
                        count += 1
        return count

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    p3 = points[k]
                    if dx * (p3[1] - p1[1]) - dy * (p3[0] - p1[0]) == 0:
                        count += 1
                        if count >= 4:
                            return True
        return False

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def generate_structured_initial():
        """Generate initial points with structured patterns that have many collinear triples."""
        strategy = random.randint(0, 3)
        
        if strategy == 0:
            # 4x5 grid with perturbation
            pts = []
            for x in range(5):
                for y in range(4):
                    px = x * 25 - 50 + random.randint(-3, 3)
                    py = y * 25 - 37 + random.randint(-3, 3)
                    px = max(-100, min(100, px))
                    py = max(-100, min(100, py))
                    pts.append([px, py])
        elif strategy == 1:
            # Two overlapping grids
            pts = []
            for x in range(5):
                for y in range(4):
                    px = x * 20 - 40 + random.randint(-2, 2)
                    py = y * 20 - 30 + random.randint(-2, 2)
                    px = max(-100, min(100, px))
                    py = max(-100, min(100, py))
                    pts.append([px, py])
        elif strategy == 2:
            # Clustered: 4 clusters of 5 points
            centers = [random_point() for _ in range(4)]
            pts = []
            for c in centers:
                for _ in range(5):
                    px = c[0] + random.randint(-8, 8)
                    py = c[1] + random.randint(-8, 8)
                    px = max(-100, min(100, px))
                    py = max(-100, min(100, py))
                    pts.append([px, py])
        else:
            # Random
            pts = [random_point() for _ in range(20)]
        
        # Ensure distinct
        seen = set()
        unique_pts = []
        for p in pts:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                unique_pts.append(p)
        
        # Fill if necessary
        while len(unique_pts) < 20:
            p = random_point()
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                unique_pts.append(p)
                
        pts = unique_pts[:20]
        
        # If invalid, try again
        if has_4_collinear(pts):
            return generate_structured_initial()
            
        return pts

    best_points = None
    best_triples = -1
    
    # Simulated Annealing Parameters
    T = 50.0
    cooling_rate = 0.9995
    min_T = 0.5
    
    for restart in range(80):
        if time.time() - start_time > timeout:
            break
            
        points = generate_structured_initial()
        current_triples = count_collinear_triples(points)
        
        if current_triples > best_triples:
            best_triples = current_triples
            best_points = [p[:] for p in points]
            
        T = 50.0
        
        # Local Search Loop
        steps = 0
        max_steps = 3000
        while steps < max_steps and time.time() - start_time < timeout:
            steps += 1
            T *= cooling_rate
            
            idx = random.randint(0, 19)
            old_p = points[idx][:]
            
            # Generate new point - multiple strategies
            move_type = random.random()
            
            if move_type < 0.4:
                # Small random perturbation
                new_p = [old_p[0] + random.randint(-8, 8), old_p[1] + random.randint(-8, 8)]
            elif move_type < 0.7:
                # Line snap: pick two other points and align
                j, k = random.sample([x for x in range(20) if x != idx], 2)
                p1 = points[j]
                p2 = points[k]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g > 0:
                    step_x = dx // g
                    step_y = dy // g
                else:
                    step_x = 1
                    step_y = 0
                
                t = random.randint(-6, 6)
                new_p = [p1[0] + t * step_x, p1[1] + t * step_y]
            else:
                # Larger jump
                new_p = [old_p[0] + random.randint(-20, 20), old_p[1] + random.randint(-20, 20)]
            
            new_p[0] = max(-100, min(100, new_p[0]))
            new_p[1] = max(-100, min(100, new_p[1]))
            
            if new_p == old_p:
                continue
                
            # Check duplicates
            is_dup = False
            for i in range(20):
                if i != idx and points[i] == new_p:
                    is_dup = True
                    break
            if is_dup:
                continue
                
            points[idx] = new_p
            
            # Check 4-collinear
            if has_4_collinear(points):
                points[idx] = old_p
                continue
                
            new_triples = count_collinear_triples(points)
            
            # Acceptance criteria
            delta = new_triples - current_triples
            if delta > 0 or (T > min_T and random.random() < math.exp(delta / T)):
                current_triples = new_triples
                if current_triples > best_triples:
                    best_triples = current_triples
                    best_points = [p[:] for p in points]
            else:
                points[idx] = old_p

    if best_points is None:
        # Fallback
        return [[x, x*x] for x in range(-10, 10)]
        
    # Final validation
    if has_4_collinear(best_points) or len(set(map(tuple, best_points))) < 20:
        return [[x, x*x] for x in range(-10, 10)]
        
    return best_points
