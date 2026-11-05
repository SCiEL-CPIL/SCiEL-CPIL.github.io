import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer

    N = 20
    LIMIT = 100

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def is_valid(points):
        """Check if no 4 points are collinear."""
        n = len(points)
        # Check all 4-combinations
        for i in range(n - 3):
            for j in range(i + 1, n - 2):
                for k in range(j + 1, n - 1):
                    p1, p2, p3 = points[i], points[j], points[k]
                    # If p1,p2,p3 are not collinear, no need to check p4
                    if not are_collinear(p1, p2, p3):
                        continue
                    for l in range(k + 1, n):
                        p4 = points[l]
                        if are_collinear(p1, p2, p4):
                            return False
        return True

    def get_triples(points):
        """Return list of collinear triples (i, j, k) with i<j<k"""
        n = len(points)
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples

    def calc_exact_score(points):
        """
        Calculate the size of the largest subset with no 3 collinear points
        using Integer Linear Programming.
        """
        n = len(points)
        triples = get_triples(points)
        
        if not triples:
            return n
            
        # Variables x_i in {0, 1}
        # Maximize sum(x_i)
        # Subject to: for each triple (i,j,k), x_i + x_j + x_k <= 2
        #             x_i >= 0
        
        # Objective coefficients
        c = -np.ones(n)  # Minimize -sum(x) is equivalent to Maximize sum(x)
        
        # Constraints
        A_rows = []
        b_lower = []
        b_upper = []
        
        for i, j, k in triples:
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            row[k] = 1
            A_rows.append(row)
            b_lower.append(0)
            b_upper.append(2)
            
        if not A_rows:
            return n
            
        A = np.array(A_rows)
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: all variables are integers
        integrality = np.ones(n)
        
        # Linear Constraint: A @ x <= b_upper (and >= b_lower, but b_lower is 0 which is handled by bounds)
        # scipy milp uses LinearConstraint(A, lb, ub)
        # Here A is the constraint matrix, lb and ub are vectors.
        constraint = LinearConstraint(A, lb=b_lower, ub=b_upper)
        
        try:
            res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            if res.status == 0:
                # res.fun is the minimum of the objective (which is -sum(x))
                # So max sum is -res.fun
                return int(round(-res.fun))
            else:
                # Fallback to heuristic if solver fails
                return calc_heuristic_score(points)
        except Exception:
            return calc_heuristic_score(points)

    def calc_heuristic_score(points):
        """
        Approximate score: 20 - max_disjoint_triples
        """
        n = len(points)
        triples = get_triples(points)
        if not triples:
            return n
        
        max_disjoint = 0
        for _ in range(3):
            used = [False] * n
            count = 0
            shuffled_triples = triples[:]
            random.shuffle(shuffled_triples)
            for t in shuffled_triples:
                i, j, k = t
                if not used[i] and not used[j] and not used[k]:
                    used[i] = True
                    used[j] = True
                    used[k] = True
                    count += 1
            max_disjoint = max(max_disjoint, count)
            
        return n - max_disjoint

    def generate_line_pack():
        """
        Generate a point set by packing points on random lines.
        """
        points = set()
        
        # Generate some lines
        lines = []
        for _ in range(10):
            # Small integer coefficients
            a = random.randint(-5, 5)
            b = random.randint(-5, 5)
            if a == 0 and b == 0:
                continue
            # Choose c such that line passes through the box
            # ax + by = c
            # x, y in [-100, 100]
            # c range approx [-500, 500]
            c = random.randint(-200, 200)
            lines.append((a, b, c))
            
        # Pick 3 points on each of several lines
        # We want to ensure we get 20 points
        # Let's try to pick points from 7 lines, 3 each = 21, then remove 1 or merge
        selected_lines = random.sample(lines, min(7, len(lines))) if len(lines) >= 7 else lines
        
        for a, b, c in selected_lines:
            # Find integer points on ax + by = c in [-100, 100]
            # If a=0, by=c => y=c/b. If b=0, ax=c => x=c/a.
            # General: x = (c - by)/a.
            # We can iterate y or x.
            # To keep it simple, just pick random x, solve for y, check integer.
            # Or iterate y from -100 to 100.
            pts_on_line = []
            if a != 0:
                # x = (c - by) / a
                # Iterate y
                for y in range(-100, 101):
                    num = c - b * y
                    if num % a == 0:
                        x = num // a
                        if -100 <= x <= 100:
                            pts_on_line.append((x, y))
            elif b != 0:
                # y = (c - ax) / b
                # Iterate x
                for x in range(-100, 101):
                    num = c - a * x
                    if num % b == 0:
                        y = num // b
                        if -100 <= y <= 100:
                            pts_on_line.append((x, y))
            
            if len(pts_on_line) >= 3:
                # Pick 3 random points from this line
                chosen = random.sample(pts_on_line, 3)
                points.update(chosen)
            elif len(pts_on_line) > 0:
                points.update(pts_on_line)

        # If we don't have enough points, add random ones
        while len(points) < N:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            points.add(p)
            
        # If we have too many, remove random ones
        if len(points) > N:
            pts_list = list(points)
            random.shuffle(pts_list)
            points = set(pts_list[:N])
            
        return list(points)

    def fix_validity(points):
        """
        Try to fix 4-collinearities by moving points.
        """
        points = points[:]
        for _ in range(100):
            if is_valid(points):
                return points
            
            # Find a 4-collinear set
            found = False
            for i in range(N - 3):
                for j in range(i + 1, N - 2):
                    for k in range(j + 1, N - 1):
                        p1, p2, p3 = points[i], points[j], points[k]
                        if not are_collinear(p1, p2, p3):
                            continue
                        for l in range(k + 1, N):
                            p4 = points[l]
                            if are_collinear(p1, p2, p4):
                                # Move p4
                                old_p = p4
                                new_p = None
                                for _ in range(50):
                                    cand = (random.randint(-100, 100), random.randint(-100, 100))
                                    if cand not in points:
                                        new_p = cand
                                        break
                                if new_p:
                                    points[l] = new_p
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
                
        return points

    # --- Main Search ---
    
    best_points = None
    best_score = N + 1
    
    # Phase 1: Generate initial candidates
    candidates = []
    for _ in range(30):
        pts = generate_line_pack()
        pts = fix_validity(pts)
        if is_valid(pts):
            # Use heuristic first to filter
            h_score = calc_heuristic_score(pts)
            candidates.append((h_score, pts))
            
    if not candidates:
        # Fallback: simple grid perturbation
        pts = [(i * 10, (i * i) % 100 - 50) for i in range(N)]
        pts = fix_validity(pts)
        if not is_valid(pts):
             # Last resort: random valid
             while True:
                 pts = [tuple(random.randint(-100, 100) for _ in range(2)) for _ in range(N)]
                 if len(set(pts)) == N and is_valid(pts):
                     break
        return [[p[0], p[1]] for p in pts]

    # Sort by heuristic score
    candidates.sort(key=lambda x: x[0])
    
    # Evaluate exact score for top candidates
    top_k = min(5, len(candidates))
    for i in range(top_k):
        if time.time() > deadline - 10:
            break
        h, pts = candidates[i]
        e_score = calc_exact_score(pts)
        if e_score < best_score:
            best_score = e_score
            best_points = pts[:]
            
    if best_points is None:
        best_points = candidates[0][1][:]
        best_score = calc_exact_score(best_points)

    # Phase 2: Local Search
    current_pts = best_points[:]
    current_score = best_score
    
    # Precompute lines for move suggestions?
    # For now, random moves with bias towards existing lines.
    
    while time.time() < deadline - 5:
        # Pick a point to move
        idx = random.randint(0, N - 1)
        old_p = current_pts[idx]
        
        # Suggest new position
        # 50% chance: random
        # 50% chance: on a line with 2 other points
        new_p = None
        if random.random() < 0.5:
            # Find lines with 2 points
            # This is expensive to do every step.
            # Just pick a random line from a precomputed set?
            # Or just try to align with existing points.
            # Pick another point j
            j = random.randint(0, N - 1)
            if j != idx:
                p_j = current_pts[j]
                # Line through p_j and old_p? No, we are moving old_p.
                # Let's pick a random point p_k and try to place new_p on line(p_j, p_k)
                k = random.randint(0, N - 1)
                if k != idx and k != j:
                    p_k = current_pts[k]
                    # Line through p_j and p_k
                    # Try to find an integer point on this line
                    # dx = p_k[0] - p_j[0], dy = p_k[1] - p_j[1]
                    dx = p_k[0] - p_j[0]
                    dy = p_k[1] - p_j[1]
                    if dx == 0 and dy == 0:
                        continue
                    # Parametric: p_j + t * (dx, dy)
                    # We want t such that x,y are integers.
                    # If gcd(dx, dy) = g, then t must be integer / g?
                    # Actually, if we step by (dx/g, dy/g), we get integer points.
                    g = np.gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    step_x = dx // g
                    step_y = dy // g
                    # Pick a random t
                    t = random.randint(-5, 5)
                    if t == 0:
                        continue
                    cand_x = p_j[0] + t * step_x
                    cand_y = p_j[1] + t * step_y
                    if -100 <= cand_x <= 100 and -100 <= cand_y <= 100:
                        new_p = (cand_x, cand_y)
                        
        if new_p is None or new_p in current_pts:
            # Fallback to random
            while True:
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
                if new_p not in current_pts:
                    break
                    
        trial_pts = current_pts[:]
        trial_pts[idx] = new_p
        
        if not is_valid(trial_pts):
            continue
            
        # Calculate score
        # To save time, we can check heuristic first
        h_score = calc_heuristic_score(trial_pts)
        
        # If heuristic is much worse, skip exact
        if h_score > current_score + 2:
            continue
            
        e_score = calc_exact_score(trial_pts)
        
        if e_score < current_score:
            current_score = e_score
            current_pts = trial_pts[:]
            if current_score < best_score:
                best_score = current_score
                best_points = current_pts[:]
        elif e_score == current_score and random.random() < 0.1:
            # Random walk
            current_pts = trial_pts[:]

    return [[p[0], p[1]] for p in best_points]

if __name__ == "__main__":
    res = construct()
    print(res)
