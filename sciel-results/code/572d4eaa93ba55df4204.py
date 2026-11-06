import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    def is_valid(points):
        n = len(points)
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:
                    continue
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_norm = dx // g
                dy_norm = dy // g
                if dx_norm < 0 or (dx_norm == 0 and dy_norm < 0):
                    dx_norm = -dx_norm
                    dy_norm = -dy_norm
                
                A = dy_norm
                B = -dx_norm
                C = A * p1[0] + B * p1[1]
                
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
                
        for key, pts in lines.items():
            if len(pts) >= 4:
                return False
        return True

    def count_triples(points):
        n = len(points)
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:
                    continue
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_norm = dx // g
                dy_norm = dy // g
                if dx_norm < 0 or (dx_norm == 0 and dy_norm < 0):
                    dx_norm = -dx_norm
                    dy_norm = -dy_norm
                
                A = dy_norm
                B = -dx_norm
                C = A * p1[0] + B * p1[1]
                
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
                
        total_triples = 0
        for key, pts in lines.items():
            k = len(pts)
            if k >= 3:
                total_triples += k * (k-1) * (k-2) // 6
        return total_triples

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def greedy_init():
        """Greedy initialization: build points one by one, choosing the point that creates the most new triples."""
        pts = []
        # Start with a few random points
        for _ in range(5):
            p = random_point()
            while any(p == q for q in pts):
                p = random_point()
            pts.append(p)
        
        while len(pts) < 20:
            best_p = None
            best_gain = -1
            # Sample candidate points
            candidates = set()
            for _ in range(50):
                candidates.add(tuple(random_point()))
            
            for c in candidates:
                c = list(c)
                if any(c == q for q in pts):
                    continue
                candidate = pts + [c]
                if not is_valid(candidate):
                    continue
                t = count_triples(candidate)
                # Gain is triples in new config minus triples in old
                gain = t - count_triples(pts)
                if gain > best_gain:
                    best_gain = gain
                    best_p = c
            
            if best_p is not None:
                pts.append(best_p)
            else:
                # Fallback: random
                while True:
                    p = random_point()
                    if not any(p == q for q in pts):
                        candidate = pts + [p]
                        if is_valid(candidate):
                            pts.append(p)
                            break
        
        return pts

    def multi_point_move(points, num_swaps):
        """Perform a multi-point swap: randomly select num_swaps indices and replace them."""
        candidate = points.copy()
        indices = random.sample(range(20), num_swaps)
        for idx in indices:
            new_p = random_point()
            # Check distinctness
            if any(new_p == candidate[j] for j in range(20) if j != idx):
                continue
            candidate[idx] = new_p
        
        if not is_valid(candidate):
            return None, None
        
        t = count_triples(candidate)
        return candidate, t

    # Initialize with multiple strategies
    best_points = None
    best_triples = -1
    
    # Strategy 1: Greedy init
    for _ in range(3):
        pts = greedy_init()
        if len(pts) == 20:
            t = count_triples(pts)
            if t > best_triples:
                best_triples = t
                best_points = pts
    
    # Strategy 2: Random init
    for _ in range(5):
        pts = []
        while len(pts) < 20:
            p = random_point()
            if any(p == q for q in pts):
                continue
            candidate = pts + [p]
            if is_valid(candidate):
                pts = candidate
            else:
                continue
        
        if len(pts) == 20:
            t = count_triples(pts)
            if t > best_triples:
                best_triples = t
                best_points = pts
    
    if best_points is None:
        # Fallback: Parabola (Score 20)
        return [[x, x * x] for x in range(-10, 10)]

    current_points = best_points
    current_triples = best_triples
    
    # Simulated annealing
    temperature = 10.0
    cooling_rate = 0.999
    no_improve_count = 0
    
    while time.time() < deadline and temperature > 0.1:
        # Choose move size: 1, 2, or 3 point swaps
        if random.random() < 0.6:
            num_swaps = 1
        elif random.random() < 0.8:
            num_swaps = 2
        else:
            num_swaps = 3
        
        candidate, new_triples = multi_point_move(current_points, num_swaps)
        
        if candidate is None:
            continue
        
        delta = new_triples - current_triples
        
        if delta > 0 or random.random() < math.exp(delta / temperature):
            current_points = candidate
            current_triples = new_triples
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        temperature *= cooling_rate
        
        # If stuck for a while, try a restart
        if no_improve_count > 50:
            pts = greedy_init()
            if len(pts) == 20:
                t = count_triples(pts)
                if t > current_triples:
                    current_points = pts
                    current_triples = t
            no_improve_count = 0
            temperature = max(temperature, 1.0)

    return current_points
