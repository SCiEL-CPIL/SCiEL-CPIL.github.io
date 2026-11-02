import numpy as np
import time

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0  # Area of the unit-side equilateral triangle
DEADLINE = 115.0      # Safety margin from the 120s limit

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

def compute_min_area(points):
    """
    Compute the minimum area of any triangle formed by three points.
    points: (N, 2) array
    """
    n = len(points)
    min_a = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i]))
                if area < min_a:
                    min_a = area
    return min_a

def is_valid_barycentric(u, v):
    """
    Check if barycentric coordinates are strictly inside the triangle.
    We use a small epsilon to ensure points are not on the boundary.
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
    # Ensure u+v <= 1 - 2*eps
    sum_uv = u + v
    mask = sum_uv > 1.0 - 2*eps
    if np.any(mask):
        # Scale down to fit
        scale = (1.0 - 2*eps) / sum_uv[mask]
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
    # Add small jitter to avoid exact boundaries if needed, though random is usually fine
    return r1, r2

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Initial random population
    best_params = None
    best_score = -np.inf
    
    # Generate initial candidate
    u = np.random.rand(N)
    v = np.random.rand(N)
    # Map to valid region
    for i in range(N):
        if u[i] + v[i] > 1.0:
            u[i] = 1.0 - u[i]
            v[i] = 1.0 - v[i]
    u, v = clamp_to_triangle(u, v)
    current_params = np.zeros(2 * N)
    current_params[0::2] = u
    current_params[1::2] = v
    
    current_pts = bary_to_cart(u, v)
    current_score = compute_min_area(current_pts)
    
    best_params = current_params.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    
    steps = 0
    max_steps = 50000 # Safety limit on steps, but time is the real constraint
    
    while time.time() - start_time < DEADLINE and T > T_min:
        steps += 1
        
        # Propose a move: perturb one or two random points
        n_perturbed = np.random.randint(1, 4) # Perturb 1 to 3 points
        indices = np.random.choice(N, size=n_perturbed, replace=False)
        
        new_u = u.copy()
        new_v = v.copy()
        
        for idx in indices:
            # Gaussian perturbation in barycentric space
            delta_u = np.random.normal(0, 0.05)
            delta_v = np.random.normal(0, 0.05)
            new_u[idx] += delta_u
            new_v[idx] += delta_v
            
        # Clamp to valid region
        new_u, new_v = clamp_to_triangle(new_u, new_v)
        
        new_params = np.zeros(2 * N)
        new_params[0::2] = new_u
        new_params[1::2] = new_v
        
        new_pts = bary_to_cart(new_u, new_v)
        new_score = compute_min_area(new_pts)
        
        # Acceptance criterion
        if new_score > current_score:
            # Always accept improvement
            u, v = new_u, new_v
            current_params = new_params
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()
        else:
            # Accept worse solution with probability exp((new - current) / T)
            # Note: We are maximizing, so delta = new - current is negative
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    u, v = new_u, new_v
                    current_params = new_params
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        
        # Periodic restart if stuck? 
        # If no improvement for a long time, we might restart from best
        # For now, simple SA is robust enough for this small problem
        
    # Final Refinement: Local Search (Nelder-Mead) on the best solution
    # We use a simple local search to polish the result
    if time.time() - start_time < DEADLINE - 5:
        # Convert best_params to u, v
        u_best = best_params[0::2].copy()
        v_best = best_params[1::2].copy()
        
        # Simple coordinate descent or Nelder-Mead
        # We'll do a few iterations of random local search with smaller steps
        local_T = 0.001
        local_steps = 5000
        for _ in range(local_steps):
            if time.time() - start_time > DEADLINE - 2:
                break
            idx = np.random.randint(0, N)
            new_u = u_best.copy()
            new_v = v_best.copy()
            
            delta_u = np.random.normal(0, 0.01)
            delta_v = np.random.normal(0, 0.01)
            new_u[idx] += delta_u
            new_v[idx] += delta_v
            
            new_u, new_v = clamp_to_triangle(new_u, new_v)
            
            new_pts = bary_to_cart(new_u, new_v)
            new_score = compute_min_area(new_pts)
            
            if new_score > best_score:
                u_best = new_u
                v_best = new_v
                best_score = new_score
                best_params = np.zeros(2 * N)
                best_params[0::2] = u_best
                best_params[1::2] = v_best
        
    # Final output
    u_out = best_params[0::2]
    v_out = best_params[1::2]
    u_out, v_out = clamp_to_triangle(u_out, v_out)
    points = bary_to_cart(u_out, v_out)
    
    return points
