import random
import time

def construct():
    random.seed(42)
    
    # Bounds
    MIN_COORD = -100
    MAX_COORD = 100
    
    # Initial random set of 20 points
    def random_point():
        return [random.randint(MIN_COORD, MAX_COORD), random.randint(MIN_COORD, MAX_COORD)]
    
    # Ensure distinct points
    def generate_initial():
        pts = []
        seen = set()
        while len(pts) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                pts.append(p)
        return pts
    
    def is_valid(pts):
        # Check if any 4 points are collinear
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                # Define line by i, j
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                A = y1 - y2
                B = x2 - x1
                C = x1*y2 - x2*y1
                count = 2 # i and j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = pts[k]
                    if A*x3 + B*y3 + C == 0:
                        count += 1
                        if count >= 4:
                            return False
        return True

    def count_triples(pts):
        # Count lines with exactly 3 points
        n = len(pts)
        triples = 0
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                A = y1 - y2
                B = x2 - x1
                C = x1*y2 - x2*y1
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = pts[k]
                    if A*x3 + B*y3 + C == 0:
                        count += 1
                if count == 3:
                    triples += 1
        return triples

    def score(pts):
        # We want to maximize triples, but validity is paramount.
        # Return a tuple (valid, triples) or a single float cost.
        # Cost = -triples if valid, else +10000
        if not is_valid(pts):
            return 10000
        return -count_triples(pts)

    # Simulated Annealing / Hill Climbing
    current_pts = generate_initial()
    current_cost = score(current_pts)
    
    best_pts = current_pts[:]
    best_cost = current_cost
    
    start_time = time.time()
    max_time = 100 # Leave 20s margin
    
    # Start with high temperature, cool down
    T = 50.0
    cooling = 0.9999
    
    while time.time() - start_time < max_time:
        # Generate neighbor
        new_pts = current_pts[:]
        idx = random.randint(0, 19)
        
        # Move point idx to a new random location
        # Sometimes just jitter, sometimes jump
        if random.random() < 0.5:
            # Jitter
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_x = current_pts[idx][0] + dx
            new_y = current_pts[idx][1] + dy
            if MIN_COORD <= new_x <= MAX_COORD and MIN_COORD <= new_y <= MAX_COORD:
                new_pts[idx] = [new_x, new_y]
            else:
                continue
        else:
            # Jump
            new_pts[idx] = random_point()
            
        # Check if point is already in set (distinctness)
        if new_pts[idx] in new_pts[:idx] or new_pts[idx] in new_pts[idx+1:]:
            continue
            
        new_cost = score(new_pts)
        
        delta = new_cost - current_cost
        
        # Acceptance criterion
        if delta < 0:
            accept = True
        else:
            accept = random.random() < pow(2.71828, -delta / T)
            
        if accept:
            current_pts = new_pts
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_pts = current_pts[:]
                
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    return best_pts
