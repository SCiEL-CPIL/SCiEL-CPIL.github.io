"""Heilbronn triangle problem: maximize minimum triangle area for 11 points.

Structural change: Formulate as a sequence of linear programs (LPs) that maximize
the minimum area t subject to linear constraints that all 165 triangle areas are
at least t. This avoids the non-smoothness of the min-function.

We use scipy.optimize.linprog with the highspy or highs_ipm solver.
"""
import numpy as np
import time
import random
from scipy.optimize import linprog

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # safety margin before 150s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _smart_init(rng):
    """Generate a smarter initial configuration using a grid-like approach with jitter."""
    pts = []
    for i in range(3):
        for j in range(4):
            if len(pts) >= N:
                break
            x = (j + 0.5) / 4.0
            y_max = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
            y = (i + 0.5) / 3.0 * y_max
            x += rng.gauss(0, 0.05)
            y += rng.gauss(0, 0.05)
            pts.append([x, y])
    while len(pts) < N:
        pts.append(_random_point_in_triangle(rng))
    return np.array(pts)

def _solve_lp_for_point(pts, idx, t_min):
    """
    Solve LP to optimize point idx.
    Variables: x_idx, y_idx, t.
    Maximize t.
    Constraints:
      t <= t_min (lower bound)
      t <= 0.5 * |cross| for all pairs (j,k) != idx
      (x_idx, y_idx) in triangle
    
    The area of triangle (i,j,k) where i=idx is:
      0.5 * |x_j*(y_k - y_i) + x_k*(y_i - y_j) + x_i*(y_j - y_k)|
    
    This is piecewise linear in x_i, y_i. We handle both signs.
    For each pair (j,k), we have:
      t <= 0.5 * (x_j*(y_k - y_i) + x_k*(y_i - y_j) + x_i*(y_j - y_k))
      t <= -0.5 * (x_j*(y_k - y_i) + x_k*(y_i - y_j) + x_i*(y_j - y_k))
    
    So for each (j,k), we get 2 linear constraints in (x_i, y_i, t).
    """
    x_j = pts[:, 0]
    y_j = pts[:, 1]
    
    # Build constraints A_ub @ [x_i, y_i, t] <= b_ub
    # We maximize t, so we minimize -t
    # c = [0, 0, -1]
    c = np.array([0.0, 0.0, -1.0])
    
    A_ub_rows = []
    b_ub_rows = []
    
    other_indices = [i for i in range(N) if i != idx]
    for j in other_indices:
        for k in other_indices:
            if k <= j:
                continue
            # Area = 0.5 * |x_j*(y_k - y_i) + x_k*(y_i - y_j) + x_i*(y_j - y_k)|
            # Let A = x_j*y_k - x_j*y_i + x_k*y_i - x_k*y_j + x_i*y_j - x_i*y_k
            # A = x_i*(y_j - y_k) + y_i*(x_k - x_j) + (x_j*y_k - x_k*y_j)
            # Area = 0.5 * |A|
            # So t <= 0.5*A and t <= -0.5*A
            # 0.5*A - t >= 0  =>  t - 0.5*A <= 0
            # -0.5*A - t >= 0 => t + 0.5*A <= 0
            # A = x_i*(y_j - y_k) + y_i*(x_k - x_j) + (x_j*y_k - x_k*y_j)
            coeff_x = y_j[k] - y_j[j]  # coefficient of x_i
            coeff_y = x_j[k] - x_j[j]  # coefficient of y_i
            const = x_j[j]*y_j[k] - x_j[k]*y_j[j]
            
            # t - 0.5*(coeff_x*x_i + coeff_y*y_i + const) <= 0
            # -0.5*coeff_x*x_i - 0.5*coeff_y*y_i + t <= 0.5*const
            A_ub_rows.append([-0.5*coeff_x, -0.5*coeff_y, 1.0])
            b_ub_rows.append(0.5*const)
            
            # t + 0.5*(coeff_x*x_i + coeff_y*y_i + const) <= 0
            # 0.5*coeff_x*x_i + 0.5*coeff_y*y_i + t <= -0.5*const
            A_ub_rows.append([0.5*coeff_x, 0.5*coeff_y, 1.0])
            b_ub_rows.append(-0.5*const)
    
    # Triangle constraints for (x_i, y_i):
    # y_i >= 0  =>  -y_i <= 0
    A_ub_rows.append([0.0, -1.0, 0.0])
    b_ub_rows.append(0.0)
    
    # y_i <= sqrt(3)*x_i  =>  -sqrt(3)*x_i + y_i <= 0
    A_ub_rows.append([-np.sqrt(3), 1.0, 0.0])
    b_ub_rows.append(0.0)
    
    # y_i <= sqrt(3)*(1-x_i)  =>  sqrt(3)*x_i + y_i <= sqrt(3)
    A_ub_rows.append([np.sqrt(3), 1.0, 0.0])
    b_ub_rows.append(np.sqrt(3))
    
    # x_i >= 0  =>  -x_i <= 0
    A_ub_rows.append([-1.0, 0.0, 0.0])
    b_ub_rows.append(0.0)
    
    # x_i <= 1  =>  x_i <= 1
    A_ub_rows.append([1.0, 0.0, 0.0])
    b_ub_rows.append(1.0)
    
    # t >= t_min  =>  -t <= -t_min
    A_ub_rows.append([0.0, 0.0, -1.0])
    b_ub_rows.append(-t_min)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    # Bounds for variables
    bounds = [(0, 1), (0, S3H), (t_min, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            new_x = res.x[0]
            new_y = res.x[1]
            new_t = res.x[2]
            pts_new = pts.copy()
            pts_new[idx] = [new_x, new_y]
            return pts_new, new_t
        else:
            return None, None
    except Exception:
        return None, None

def _local_refine(pts, max_iters=200, rng=None):
    """Quick local refinement using coordinate descent style approach."""
    if rng is None:
        rng = random.Random(12345)
    
    best_area = _min_triangle_area(pts)
    current_pts = pts.copy()
    
    step_size = 0.02
    for it in range(max_iters):
        improved = False
        for i in range(N):
            for dx, dy in [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size)]:
                trial = current_pts.copy()
                trial[i, 0] += dx
                trial[i, 1] += dy
                trial = _project_to_triangle(trial)
                area = _min_triangle_area(trial)
                if area > best_area:
                    best_area = area
                    current_pts = trial
                    improved = True
                    break
        if not improved:
            step_size *= 0.7
            if step_size < 0.001:
                break
    
    return current_pts, best_area

def construct():
    """Construct 11 points in the unit equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_area = 0.0
    
    # Phase 1: Smart initialization
    initial_pts = _smart_init(rng)
    current_pts = initial_pts.copy()
    current_area = _min_triangle_area(current_pts)
    
    if current_area > best_area:
        best_area = current_area
        best_pts = current_pts.copy()
    
    # Phase 2: Iterative LP-based local optimization
    # Cyclically optimize each point using LP
    num_cyclic_passes = 50
    for p in range(num_cyclic_passes):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        improved_in_pass = False
        for i in range(N):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Solve LP for point i
            new_pts, new_t = _solve_lp_for_point(current_pts, i, best_area)
            if new_pts is not None:
                new_area = _min_triangle_area(new_pts)
                if new_area > best_area:
                    best_area = new_area
                    best_pts = new_pts.copy()
                    current_pts = new_pts
                    improved_in_pass = True
        
        if not improved_in_pass:
            # No improvement in a full pass; try a random restart with perturbation
            if time.time() - start_time > DEADLINE - 15:
                break
            current_pts = best_pts + rng.gauss(0, 0.01, (N, 2))
            current_pts = _project_to_triangle(current_pts)
    
    # Phase 3: Local refinement from best found
    if time.time() - start_time < DEADLINE - 5:
        refined_pts, refined_area = _local_refine(best_pts, max_iters=100, rng=rng)
        if refined_area > best_area:
            best_area = refined_area
            best_pts = refined_pts
    
    # Phase 4: Additional LP-based optimization with different initializations
    num_restarts = 3
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # New initial configuration
        restart_pts = _smart_init(rng)
        current_pts = restart_pts.copy()
        current_area = _min_triangle_area(current_pts)
        
        if current_area > best_area:
            best_area = current_area
            best_pts = current_pts.copy()
        
        # LP-based optimization
        for p in range(30):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            improved_in_pass = False
            for i in range(N):
                if time.time() - start_time > DEADLINE - 10:
                    break
                
                new_pts, new_t = _solve_lp_for_point(current_pts, i, best_area)
                if new_pts is not None:
                    new_area = _min_triangle_area(new_pts)
                    if new_area > best_area:
                        best_area = new_area
                        best_pts = new_pts.copy()
                        current_pts = new_pts
                        improved_in_pass = True
            
            if not improved_in_pass:
                break
        
        # Local refine
        if time.time() - start_time < DEADLINE - 5:
            refined_pts, refined_area = _local_refine(best_pts, max_iters=50, rng=rng)
            if refined_area > best_area:
                best_area = refined_area
                best_pts = refined_pts
    
    if best_pts is None:
        best_pts = _smart_init(rng)
    
    return best_pts
