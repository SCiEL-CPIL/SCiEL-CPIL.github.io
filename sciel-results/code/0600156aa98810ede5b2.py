import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0  # Area of the unit-side equilateral triangle
DEADLINE = 145.0      # Safety margin from the 150s limit

def bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    Vertices: V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    u is weight for V1, v is weight for V2, w=1-u-v is weight for V0.
    x = u*1 + v*0.5
    y = v*sqrt(3)/2
    """
    x = u + 0.5 * v
    y = S3H * v
    return np.column_stack((x, y))

def cart_to_bary(x, y):
    """
    Convert Cartesian coordinates to barycentric (u, v).
    w = y / S3H
    v = x - 0.5 * w
    u = 1 - v - w
    """
    w = y / S3H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return u, v

def compute_min_area(points):
    """
    Compute the minimum area of any triangle formed by three points.
    points: (N, 2) array
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Vectorized computation
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    
    # Cross product z-component for all pairs
    # For triangle (i, j, k), area = 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
    # We can compute the signed double area for all triples.
    
    # Efficient way: 
    # Area(i,j,k) = 0.5 * | x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j) |
    
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs(x[i]*(y[j] - y[k]) + x[j]*(y[k] - y[i]) + x[k]*(y[i] - y[j]))
                if area < min_a:
                    min_a = area
    return min_a

def is_valid_barycentric(u, v):
    """
    Check if barycentric coordinates are strictly inside the triangle.
    """
    eps = 1e-6
    return np.all(u > eps) and np.all(v > eps) and np.all(u + v < 1.0 - eps)

def clamp_to_triangle(u, v):
    """
    Clamp u and v to be strictly inside the triangle.
    """
    eps = 1e-5
    u = np.clip(u, eps, 1.0 - 2*eps)
    v = np.clip(v, eps, 1.0 - 2*eps)
    sum_uv = u + v
    mask = sum_uv > 1.0 - 2*eps
    if np.any(mask):
        scale = (1.0 - 2*eps) / sum_uv[mask]
        u = u.copy()
        v = v.copy()
        u[mask] *= scale
        v[mask] *= scale
    return u, v

def random_point_in_triangle():
    """
    Generate a random point inside the triangle using barycentric coordinates.
    """
    r1 = np.random.rand()
    r2 = np.random.rand()
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    return r1, r2

def hexagonal_initialization(rng):
    """
    Generate points on a hexagonal grid inside the triangle.
    This provides a structured, well-distributed starting point.
    """
    points = []
    # Try different grid densities
    best_score = -1
    best_pts = None
    
    # The triangle has side length 1.
    # Hexagonal grid spacing d.
    for d in [0.2, 0.25, 0.3, 0.4]:
        pts = []
        # Generate points in a hexagonal lattice
        # Rows are offset by d/2
        y = 0.0
        row = 0
        while y <= S3H + d:
            x_start = 0.0 if row % 2 == 0 else d/2
            x = x_start
            while x <= 1.0 + d:
                # Check if point is inside triangle
                # x in [0, 1]
                # y in [0, S3H]
                # y <= S3H * x  (for x in [0, 0.5]) -> actually y <= S3H * (1-x) for x in [0.5, 1]
                # Let's use barycentric check
                u, v = cart_to_bary(x, y)
                if u >= -1e-9 and v >= -1e-9 and u+v <= 1.0 + 1e-9:
                    # Ensure strictly inside for optimization start?
                    # Clamping later handles it.
                    pts.append([x, y])
                x += d
            y += d * S3H
            row += 1
        
        if len(pts) < N:
            continue
            
        pts = np.array(pts)
        
        # Select N points from the grid
        # Simplest: random selection
        # Better: select points that maximize min distance? Or just random and let SA fix it.
        # Let's try a few random selections and keep the best
        for _ in range(10):
            if len(pts) == N:
                sel = pts
            else:
                indices = rng.choice(len(pts), N, replace=False)
                sel = pts[indices]
            
            score = compute_min_area(sel)
            if score > best_score:
                best_score = score
                best_pts = sel.copy()
    
    if best_pts is None:
        # Fallback to random
        u = np.random.rand(N)
        v = np.random.rand(N)
        for i in range(N):
            if u[i] + v[i] > 1.0:
                u[i] = 1.0 - u[i]
                v[i] = 1.0 - v[i]
        u, v = clamp_to_triangle(u, v)
        best_pts = bary_to_cart(u, v)
        
    return best_pts

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # 1. Initialization: Hexagonal Grid
    current_pts = hexagonal_initialization(rng)
    u, v = cart_to_bary(current_pts[:, 0], current_pts[:, 1])
    u, v = clamp_to_triangle(u, v)
    current_pts = bary_to_cart(u, v)
    
    current_score = compute_min_area(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # 2. Simulated Annealing Phase
    # Parameters
    T_start = 0.05
    T_end = 0.001
    n_sa_iter = 10000 # Max iterations, but limited by time
    
    for i in range(n_sa_iter):
        if time.time() - start_time > DEADLINE - 30: # Leave time for local opt
            break
            
        T = T_start * (T_end / T_start) ** (i / n_sa_iter)
        
        # Propose move: perturb 1 or 2 points
        n_perturb = rng.randint(1, 3)
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        new_u = u.copy()
        new_v = v.copy()
        
        step_size = 0.05 * (1.0 + T/T_start)
        
        for idx in indices:
            new_u[idx] += rng.normal(0, step_size)
            new_v[idx] += rng.normal(0, step_size)
            
        new_u, new_v = clamp_to_triangle(new_u, new_v)
        new_pts = bary_to_cart(new_u, new_v)
        new_score = compute_min_area(new_pts)
        
        if new_score >= current_score:
            u, v = new_u, new_v
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            delta = current_score - new_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    u, v = new_u, new_v
                    current_pts = new_pts
                    current_score = new_score
    
    # 3. Local Optimization Phase
    # Use Nelder-Mead to polish the best solution found by SA
    if time.time() - start_time < DEADLINE - 5:
        
        # Define objective for minimization (negative min area)
        def objective(params):
            uv = params.reshape(N, 2)
            u_opt = uv[:, 0]
            v_opt = uv[:, 1]
            
            # Penalty for out-of-bounds
            # We rely on Nelder-Mead staying close, but add penalty just in case
            # If points go outside, area calculation might be weird or points might be invalid
            # Let's clamp inside the objective to ensure validity, but this makes it non-differentiable
            # Better: just check validity. If invalid, return large value.
            
            # Check validity
            if not (np.all(u_opt > -1e-5) and np.all(v_opt > -1e-5) and np.all(u_opt + v_opt < 1.0 + 1e-5)):
                # Soft penalty
                penalty = 0.0
                penalty += np.sum(np.maximum(0, -u_opt)**2)
                penalty += np.sum(np.maximum(0, -v_opt)**2)
                penalty += np.sum(np.maximum(0, u_opt + v_opt - 1.0)**2)
                if penalty > 1e-4:
                    return 1.0 + penalty # Large value to discourage
            
            pts = bary_to_cart(u_opt, v_opt)
            min_a = compute_min_area(pts)
            return -min_a
        
        # Initial guess from best SA result
        u_best = best_pts[:, 0]
        v_best = best_pts[:, 1]
        # Wait, best_pts is Cartesian. Convert to barycentric for optimization
        u_b, v_b = cart_to_bary(best_pts[:, 0], best_pts[:, 1])
        u_b, v_b = clamp_to_triangle(u_b, v_b)
        
        x0 = np.column_stack((u_b, v_b)).flatten()
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success or res.fun < objective(x0):
                # Check if result is valid
                u_res = res.x[0::2]
                v_res = res.x[1::2]
                
                # Clamp
                u_res, v_res = clamp_to_triangle(u_res, v_res)
                pts_res = bary_to_cart(u_res, v_res)
                score_res = compute_min_area(pts_res)
                
                if score_res > best_score:
                    best_score = score_res
                    best_pts = pts_res
        except Exception as e:
            pass # If optimization fails, keep SA result
            
    # Final Output
    # Ensure points are strictly inside
    u_out, v_out = cart_to_bary(best_pts[:, 0], best_pts[:, 1])
    u_out, v_out = clamp_to_triangle(u_out, v_out)
    points = bary_to_cart(u_out, v_out)
    
    return points

# Keep the original function name and signature
# The task says "Keep the function names and signatures of the seed program."
# Seed program has `construct()` returning array of shape (n, 2).

if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
    print(pts)
