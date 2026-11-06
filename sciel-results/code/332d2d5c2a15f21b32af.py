import random
import time
import math

def construct():
    start_time = time.time()
    timeout = 140  # Leave 10s margin for evaluator overhead
    
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
        # For each pair, check if there are at least 2 other points on the same line
        for i in range(n):
            for j in range(i+1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                count = 2 # p1 and p2
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

    def generate_initial_points():
        # Try a few different initialization strategies
        strategy = random.randint(0, 2)
        if strategy == 0:
            # Random
            pts = [random_point() for _ in range(20)]
        elif strategy == 1:
            # Perturbed Grid (4x5)
            pts = []
            for x in range(5):
                for y in range(4):
                    px = x * 20 - 40 + random.randint(-5, 5)
                    py = y * 20 - 30 + random.randint(-5, 5)
                    px = max(-100, min(100, px))
                    py = max(-100, min(100, py))
                    pts.append([px, py])
        else:
            # Clustered Random
            # Pick 4 centers, assign 5 points to each
            centers = [random_point() for _ in range(4)]
            pts = []
            for c in centers:
                for _ in range(5):
                    px = c[0] + random.randint(-10, 10)
                    py = c[1] + random.randint(-10, 10)
                    px = max(-100, min(100, px))
                    py = max(-100, min(100, py))
                    pts.append([px, py])
        
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
        
        # If invalid, try again or just return and let search fix it?
        # Better to ensure validity at start if possible, or accept and penalize?
        # The constraint is hard. If we start invalid, we can't evaluate score properly?
        # Actually, score is defined for valid configs.
        # So we must ensure validity.
        if has_4_collinear(pts):
             # Simple fix: move points involved in 4-collinear lines
             # Or just random restart
             return generate_initial_points()
             
        return pts

    best_points = None
    best_triples = -1
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    
    for restart in range(50):
        if time.time() - start_time > timeout:
            break
            
        points = generate_initial_points()
        current_triples = count_collinear_triples(points)
        
        if current_triples > best_triples:
            best_triples = current_triples
            best_points = [p[:] for p in points]
            
        T = 10.0
        
        # Local Search Loop
        steps = 0
        max_steps = 2000
        while steps < max_steps and time.time() - start_time < timeout:
            steps += 1
            T *= cooling_rate
            
            idx = random.randint(0, 19)
            old_p = points[idx][:]
            
            # Generate new point
            # 50% random jump, 50% line-snap
            if random.random() < 0.5:
                new_p = [old_p[0] + random.randint(-10, 10), old_p[1] + random.randint(-10, 10)]
            else:
                # Pick two other points and try to align
                j, k = random.sample([x for x in range(20) if x != idx], 2)
                p1 = points[j]
                p2 = points[k]
                # Line through p1, p2: y = mx + c
                # We want an integer point on this line.
                # Just pick a random parameter t and calculate?
                # Or just pick a random point on the line segment or extended?
                # Simpler: pick a random point, then project? No.
                # Let's just pick a random point on the grid and see if it's close to the line?
                # Actually, just pick a random point on the line defined by p1, p2.
                # Direction vector d = (dx, dy).
                # Point = p1 + t * d.
                # We need integer coordinates.
                # If dx, dy are not coprime, we can step by (dx/g, dy/g).
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g > 0:
                    step_x = dx // g
                    step_y = dy // g
                else:
                    step_x = 1
                    step_y = 0
                
                # Pick a random t
                t = random.randint(-5, 5)
                new_p = [p1[0] + t * step_x, p1[1] + t * step_y]
            
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
            if delta > 0 or random.random() < math.exp(delta / T):
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
