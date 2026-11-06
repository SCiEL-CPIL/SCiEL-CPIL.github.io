import time
import random
import sys
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 20
    COORD_MIN = -100
    COORD_MAX = 100

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return (0, 0, 0)
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def build_line_map(points):
        n = len(points)
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map

    def is_valid(points):
        line_map = build_line_map(points)
        for line_pts in line_map.values():
            if len(line_pts) >= 4:
                return False
        return True

    def get_collinear_triples(points):
        line_map = build_line_map(points)
        triples = []
        for line_pts in line_map.values():
            if len(line_pts) >= 3:
                pts = sorted(line_pts)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        for k in range(j + 1, len(pts)):
                            triples.append((pts[i], pts[j], pts[k]))
        return triples

    def compute_exact_score(points):
        triples = get_collinear_triples(points)
        if not triples:
            return N
        
        num_vars = N
        # Maximize sum(x_i)
        # Equivalent to Minimize -sum(x_i)
        c = -np.ones(num_vars)
        
        # Constraints: x_i + x_j + x_k <= 2 for each triple
        A = []
        for i, j, k in triples:
            row = np.zeros(num_vars)
            row[i] = 1
            row[j] = 1
            row[k] = 1
            A.append(row)
        
        if not A:
            return N
            
        A = np.array(A)
        lb = np.zeros(len(A))
        ub = np.full(len(A), 2)
        
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(num_vars)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                # res.fun is the minimum value of -sum(x_i)
                # So max sum(x_i) is -res.fun
                score = int(round(-res.fun))
                return score
            else:
                # Fallback to simple heuristic if solver fails
                return N
        except Exception:
            return N

    def generate_random_points():
        pts = []
        while len(pts) < N:
            x = random.randint(COORD_MIN, COORD_MAX)
            y = random.randint(COORD_MIN, COORD_MAX)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        return pts

    best_config = None
    best_score = N + 1
    
    # Initial search for a valid configuration
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        pts = generate_random_points()
        if is_valid(pts):
            best_config = pts
            break

    if best_config is None:
        # Fallback: Parabola
        return [[x, x * x] for x in range(-10, 10)]

    current_pts = [p[:] for p in best_config]
    current_score = compute_exact_score(current_pts)
    best_score = current_score

    # Local Search with Simulated Annealing
    # We want to minimize score.
    temp = 5.0
    cooling = 0.995
    step_count = 0
    no_improve_steps = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 5:
        step_count += 1
        idx = random.randint(0, N - 1)
        old_x, old_y = current_pts[idx]
        
        new_x = random.randint(COORD_MIN, COORD_MAX)
        new_y = random.randint(COORD_MIN, COORD_MAX)
        
        if new_x == old_x and new_y == old_y:
            continue
            
        if any(current_pts[i] == [new_x, new_y] for i in range(N) if i != idx):
            continue
            
        current_pts[idx] = [new_x, new_y]
        
        if not is_valid(current_pts):
            current_pts[idx] = [old_x, old_y]
            continue
            
        new_score = compute_exact_score(current_pts)
        
        delta = new_score - current_score
        
        if new_score < best_score:
            best_score = new_score
            best_config = [p[:] for p in current_pts]
            no_improve_steps = 0
            # Restart annealing slightly to explore around new best?
            # Or just keep going.
        else:
            no_improve_steps += 1
            if no_improve_steps > max_no_improve:
                # Restart with new random config to escape global local minima
                for _ in range(10):
                    if time.time() > deadline - 10: break
                    new_pts = generate_random_points()
                    if is_valid(new_pts):
                        current_pts = [p[:] for p in new_pts]
                        current_score = compute_exact_score(current_pts)
                        break
                no_improve_steps = 0
                temp = 5.0
                continue

        if delta < 0:
            # Improvement
            current_score = new_score
        else:
            # Deterioration or neutral
            if random.random() < pow(2.718, -delta / temp):
                current_score = new_score
            else:
                current_pts[idx] = [old_x, old_y]
                continue
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    return best_config

if __name__ == "__main__":
    res = construct()
    print(res)
