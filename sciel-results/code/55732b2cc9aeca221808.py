import random
import time
import sys
import math
from itertools import combinations
from math import gcd

def cross(o, a, b):
    """Cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - b[0])

def are_collinear(p1, p2, p3):
    return cross(p1, p2, p3) == 0

def _build_lines(points):
    """Build all lines with >= 3 points, return list of (line_key, [indices]) where len >= 3."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Normalize line equation ax + by + c = 0
            if dx == 0:
                a, b, c = 1, 0, -x1
            elif dy == 0:
                a, b, c = 0, 1, -y1
            else:
                g = gcd(abs(dx), abs(dy))
                dxn, dyn = dx // g, dy // g
                if dxn < 0:
                    dxn, dyn = -dxn, -dyn
                a, b = dyn, -dxn
                c = -(a * x1 + b * y1)
                g2 = gcd(gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2; b //= g2; c //= g2
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    lines = []
    for key, idxs in line_map.items():
        if len(idxs) >= 3:
            lines.append((key, sorted(idxs)))
    return lines

def has_four_collinear(points):
    """Check if any 4 points are collinear."""
    n = len(points)
    lines = _build_lines(points)
    for _, idxs in lines:
        if len(idxs) >= 4:
            return True
    return False

def count_collinear_triples(points):
    """Count collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def max_independent_set_3_uniform(points):
    """
    Compute the size of the largest subset of points with no three collinear.
    Uses ILP via scipy.optimize.milp.
    """
    n = len(points)
    if n <= 2:
        return n
    
    # Build collinear triples
    collinear_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    collinear_triples.append((i, j, k))
    
    if not collinear_triples:
        return n
    
    # ILP formulation:
    # Maximize sum(x_i)
    # Subject to: x_i + x_j + x_k <= 2 for each collinear triple (i,j,k)
    # x_i in {0, 1}
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        m = len(collinear_triples)
        
        # Objective: maximize sum(x_i) -> minimize -sum(x_i)
        c = np.full(n, -1.0)
        
        # Constraints: for each triple, sum of 3 vars <= 2
        A = np.zeros((m, n))
        for idx, (i, j, k) in enumerate(collinear_triples):
            A[idx, i] = 1
            A[idx, j] = 1
            A[idx, k] = 1
        
        # x_i in [0, 1]
        integrality = np.ones(n)
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        constraints = LinearConstraint(A, lb=-np.inf, ub=2 * np.ones(m))
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success and res.x is not None:
            # Round to nearest integer
            x = np.round(res.x).astype(int)
            return int(np.sum(x))
        else:
            # Fallback to branch and bound
            return _bbscore(points, collinear_triples)
            
    except Exception:
        return _bbscore(points, collinear_triples)

def _bbscore(points, collinear_triples):
    """Fallback branch and bound for scoring."""
    n = len(points)
    # Build pair_to_third
    pair_to_third = {}
    for i in range(n):
        pair_to_third[i] = {}
        for j in range(n):
            if i == j:
                continue
            pair_to_third[i][j] = set()
    for i, j, k in collinear_triples:
        pair_to_third[i][j].add(k)
        pair_to_third[i][k].add(j)
        pair_to_third[j][i].add(k)
        pair_to_third[j][k].add(i)
        pair_to_third[k][i].add(j)
        pair_to_third[k][j].add(i)
    
    best_size = 0
    full_mask = (1 << n) - 1
    
    def solve_recur(current_size, current_mask, remaining_mask):
        nonlocal best_size
        if not remaining_mask:
            if current_size > best_size:
                best_size = current_size
            return
        if current_size + bin(remaining_mask).count('1') <= best_size:
            return
        
        p = -1
        for i in range(n):
            if remaining_mask & (1 << i):
                p = i
                break
        if p == -1:
            return
        
        # Exclude p
        solve_recur(current_size, current_mask, remaining_mask & ~(1 << p))
        
        # Include p
        compatible = True
        cm = current_mask
        while cm:
            lsb = cm & -cm
            r = (lsb.bit_length() - 1)
            if pair_to_third[p][r] & current_mask:
                compatible = False
                break
            cm ^= lsb
        
        if compatible:
            new_current_mask = current_mask | (1 << p)
            new_remaining = remaining_mask & ~(1 << p)
            forbidden_mask = 0
            ncm = new_current_mask
            while ncm:
                lsb = ncm & -ncm
                r = (lsb.bit_length() - 1)
                for q in pair_to_third[p][r]:
                    if new_remaining & (1 << q):
                        forbidden_mask |= (1 << q)
                ncm ^= lsb
            new_remaining &= ~forbidden_mask
            solve_recur(current_size + 1, new_current_mask, new_remaining)
    
    solve_recur(0, 0, full_mask)
    return best_size

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_score = 20
    best_points = None
    best_triple_count = -1

    def random_point():
        return (random.randint(-100, 100), random.randint(-100, 100))

    def grid_init():
        """
        Generate a perturbed grid initialization.
        Start with a dense grid, then perturb points to break 4-collinearities.
        """
        # Try different grid dimensions
        dims = [(4, 5), (5, 4), (6, 4), (4, 6), (7, 3), (3, 7)]
        random.shuffle(dims)
        
        for rows, cols in dims:
            ox = random.randint(-50, 50)
            oy = random.randint(-50, 50)
            sx = random.randint(3, 15)
            sy = random.randint(3, 15)
            
            candidates = []
            for i in range(rows):
                for j in range(cols):
                    x = ox + i * sx
                    y = oy + j * sy
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        candidates.append((x, y))
            
            if len(candidates) < 20:
                continue
                
            random.shuffle(candidates)
            points = candidates[:20]
            
            # Perturb points to break 4-collinearities
            # We will rely on repair to fix any remaining 4-collinearities
            return points
        return None

    def repair_four_collinear(points):
        """Repair points to remove any 4-collinear violations."""
        points = list(points)
        for _ in range(500):
            if not has_four_collinear(points):
                return points
            # Find a violating line
            lines = _build_lines(points)
            violated = None
            for _, idxs in lines:
                if len(idxs) >= 4:
                    violated = idxs
                    break
            if violated is None:
                return points
            # Move a random point from the violated line
            idx = random.choice(violated)
            # Try to find a new position that doesn't create new 4-collinearities
            for _ in range(50):
                new_p = random_point()
                if new_p not in points:
                    points[idx] = new_p
                    if not has_four_collinear(points):
                        return points
            # If we couldn't fix it easily, keep trying
        return points

    def simulated_annealing(points, max_steps=2000):
        """
        Simulated Annealing to maximize collinear triples while maintaining validity.
        """
        points = list(points)
        current_tc = count_collinear_triples(points)
        T = 1.0
        cooling_rate = 0.995
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Pick a random point to move
            idx = random.randint(0, 19)
            old_p = points[idx]
            new_p = random_point()
            
            if new_p == old_p:
                continue
            if new_p in points:
                continue
                
            candidate = list(points)
            candidate[idx] = new_p
            
            # Check validity
            if has_four_collinear(candidate):
                T *= cooling_rate
                continue
                
            new_tc = count_collinear_triples(candidate)
            delta_tc = new_tc - current_tc
            
            # Acceptance probability
            if delta_tc > 0:
                points = candidate
                current_tc = new_tc
            else:
                # Accept worse move with probability exp(delta / T)
                if delta_tc == 0:
                    # Accept equal moves to explore
                    if random.random() < 0.5:
                        points = candidate
                        current_tc = new_tc
                else:
                    prob = math.exp(delta_tc / T)
                    if random.random() < prob:
                        points = candidate
                        current_tc = new_tc
                        
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
                
        return points, current_tc

    # Phase 1: Generate diverse initial configurations and run SA
    for restart in range(50):
        if time.time() > deadline:
            break
        
        points = grid_init()
        if points is None:
            continue
            
        points = repair_four_collinear(points)
        
        if has_four_collinear(points):
            continue
        
        # Run Simulated Annealing
        points, tc = simulated_annealing(points, max_steps=3000)
        
        if has_four_collinear(points):
            continue
        
        # Compute exact score using ILP
        score = max_independent_set_3_uniform(points)
        
        if score < best_score:
            best_score = score
            best_points = list(points)
            best_triple_count = tc
        elif score == best_score and tc > best_triple_count:
            best_triple_count = tc
            best_points = list(points)
            
        # If we find a very good score, we can do a more intensive search on it
        if best_score <= 14:
             # Do a quick local polish
             for _ in range(1000):
                 if time.time() > deadline: break
                 idx = random.randint(0, 19)
                 new_p = random_point()
                 if new_p in points: continue
                 cand = list(points)
                 cand[idx] = new_p
                 if has_four_collinear(cand): continue
                 new_tc = count_collinear_triples(cand)
                 if new_tc > tc:
                     points = cand
                     tc = new_tc
                     score = max_independent_set_3_uniform(points)
                     if score < best_score:
                         best_score = score
                         best_points = list(points)
                         best_triple_count = tc

    if best_points is None:
        # Fallback: random valid set
        points = []
        seen = set()
        while len(points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                points.append(p)
        for _ in range(200):
            if not has_four_collinear(points):
                break
            idx = random.randint(0, 19)
            new_p = random_point()
            if new_p not in points:
                points[idx] = new_p
        best_points = points
    
    return [list(p) for p in best_points]
