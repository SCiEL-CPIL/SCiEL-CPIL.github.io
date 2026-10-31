"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _triangle_area(p1, p2, p3):
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    return 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))

def _min_area(points):
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _clamp_to_triangle(p):
    x, y = p
    h = np.sqrt(3) / 2
    eps = 1e-9
    # Constraints: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    # Note: slope is sqrt(3), not h. h = sqrt(3)/2.
    # y <= 2*h*x and y <= 2*h*(1-x)
    
    y = max(eps, min(2*h*(1-eps), y)) # Max possible y is h at x=0.5
    
    # Calculate valid x range for given y
    # y <= 2*h*x  => x >= y/(2*h)
    # y <= 2*h*(1-x) => 1-x >= y/(2*h) => x <= 1 - y/(2*h)
    
    x_min = y / (2 * h)
    x_max = 1 - y / (2 * h)
    
    if x_min > x_max:
        # If y is too high, clamp y down
        y = h * 0.99
        x_min = y / (2 * h)
        x_max = 1 - y / (2 * h)
        
    x = max(x_min + eps, min(x_max - eps, x))
    return np.array([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    h = np.sqrt(3) / 2
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, h])
    
    def bary_to_cart(u, v, w):
        return u * A + v * B + w * C

    def cart_to_bary(p):
        x, y = p
        # Solve for u, v, w
        # x = v + 0.5w
        # y = h w  => w = y/h
        w = y / h
        v = x - 0.5 * w
        u = 1 - v - w
        return np.array([u, v, w])

    def objective(params):
        # params is flat array of 2N barycentric coords (u, v) for N points
        # w = 1 - u - v
        uv = params.reshape(N, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1 - u - v
        
        # Ensure non-negative and normalized if needed (though bounds should handle it)
        # For optimization, we just compute points. If invalid, return large penalty.
        if np.any(u < -1e-8) or np.any(v < -1e-8) or np.any(w < -1e-8):
            return 1e10
            
        # Normalize to be safe
        s = u + v + w
        u = u / s
        v = v / s
        w = w / s
        
        points = np.array([bary_to_cart(u[i], v[i], w[i]) for i in range(N)])
        min_a = _min_area(points)
        return -min_a

    # Generate a good initial guess
    # A common good configuration for 11 points is a triangular grid
    # Let's try a 3x3 grid plus corners/center adjustments, or just a random symmetric start
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = -float('inf')
    
    # Start with a few different initial configurations
    initial_configs = []
    
    # Config 1: Triangular lattice
    # 11 points. 
    # Rows: 1, 2, 3, 3, 2? No, total 11.
    # Let's just use a dense random start with symmetry enforcement
    # Or a simple grid in barycentric coordinates
    
    # Grid in barycentric space (u, v)
    # u, v >= 0, u+v <= 1
    # Let's try a 4x4 grid in the (u,v) triangle
    # We need 11 points.
    # We can take a subset of a grid.
    
    # Let's try 3 restarts with local optimization from different starts
    for restart in range(3):
        if time.time() > deadline - 20:
            break
            
        # Generate random points in barycentric coords
        u = rng.random(N)
        v = rng.random(N)
        # Scale to ensure u+v < 1
        # Simple way: generate in unit square and project
        # Or generate Dirichlet(1,1,1)
        # Dirichlet alpha=1 is uniform on simplex
        bary = rng.dirichlet([1, 1, 1], size=N)
        u = bary[:, 0]
        v = bary[:, 1]
        
        x0 = np.concatenate([u, v])
        
        try:
            # Use L-BFGS-B for efficiency
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N), # u, v in [0,1] is safe, w=1-u-v
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            score = -res.fun
            if score > best_score:
                best_score = score
                # Convert back to points
                uv = res.x.reshape(N, 2)
                u_opt = uv[:, 0]
                v_opt = uv[:, 1]
                w_opt = 1 - u_opt - v_opt
                
                # Normalize again just in case
                s = u_opt + v_opt + w_opt
                u_opt = u_opt / s
                v_opt = v_opt / s
                w_opt = w_opt / s
                
                pts = np.array([bary_to_cart(u_opt[i], v_opt[i], w_opt[i]) for i in range(N)])
                best_points = pts
                
        except Exception:
            pass

    if best_points is None:
        # Fallback to a valid configuration if optimization failed
        # Use a simple grid
        h = np.sqrt(3)/2
        pts = []
        # 3x3 grid in Cartesian, clamped
        for i in range(3):
            for j in range(3):
                x = 0.25 + i * 0.25
                y = 0.25 * h + j * 0.25 * h
                p = _clamp_to_triangle([x, y])
                pts.append(p)
        # We need 11 points, we have 9. Add 2 more near center
        pts.append(_clamp_to_triangle([0.5, h/2]))
        pts.append(_clamp_to_triangle([0.5, h/4]))
        best_points = np.array(pts)

    # Final safety check and clamp
    final_points = np.array([_clamp_to_triangle(p) for p in best_points])
    return final_points
# EVOLVE-BLOCK-END
