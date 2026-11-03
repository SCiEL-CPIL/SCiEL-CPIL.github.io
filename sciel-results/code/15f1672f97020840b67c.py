import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

def bary_to_cart(u, v):
    """Convert barycentric coordinates (u, v) to Cartesian coordinates.
    Point P = (1-u-v)*A + u*B + v*C
    Since A is (0,0), P = u*B + v*C
    """
    x = u * B[0] + v * C[0]
    y = u * B[1] + v * C[1]
    return np.array([x, y])

def cart_to_bary(x, y):
    """Convert Cartesian coordinates to barycentric coordinates (u, v).
    x = u*1 + v*0.5
    y = u*0 + v*(sqrt(3)/2)
    => v = 2*y/sqrt(3)
    => u = x - 0.5*v
    """
    v = 2.0 * y / np.sqrt(3)
    u = x - 0.5 * v
    return np.array([u, v])

def compute_min_area(points):
    """
    Compute the minimum area of any triangle formed by the points.
    points: array of shape (N, 2)
    """
    min_area = np.inf
    n = len(points)
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            for k in range(j+1, n):
                pk = points[k]
                vec1 = pj - pi
                vec2 = pk - pi
                area = 0.5 * abs(vec1[0]*vec2[1] - vec1[1]*vec2[0])
                if area < min_area:
                    min_area = area
    return min_area

def clamp_bary(u, v):
    """Clamp barycentric coordinates to valid range [0,1] with u+v<=1."""
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
    return u, v

def objective(params):
    """
    params: array of shape (2*N,) representing u_i, v_i for each point.
    Returns negative of the minimum area.
    """
    u = params[0:N]
    v = params[N:2*N]
    
    u_clipped, v_clipped = clamp_bary(u, v)
    points = np.array([bary_to_cart(ui, vi) for ui, vi in zip(u_clipped, v_clipped)])
    
    min_area = compute_min_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    best_params = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initializations
    initializations = []
    
    # Random initializations
    for _ in range(8):
        u = rng.random(N)
        v = rng.random(N)
        u, v = clamp_bary(u, v)
        initializations.append(np.concatenate([u, v]))
    
    # Structured initialization: grid-like pattern
    # Place points in a rough grid within the triangle
    for _ in range(3):
        u_vals = []
        v_vals = []
        # Try different grid patterns
        for i in range(N):
            # Distribute points roughly evenly
            row = i // 4
            col = i % 4
            # Adjust for triangular shape
            u_base = col / 4.0
            v_base = row / 4.0 * (1 - col / 4.0)
            # Add small random perturbation
            u_val = u_base + rng.uniform(-0.05, 0.05)
            v_val = v_base + rng.uniform(-0.05, 0.05)
            u_vals.append(u_val)
            v_vals.append(v_val)
        u = np.array(u_vals)
        v = np.array(v_vals)
        u, v = clamp_bary(u, v)
        initializations.append(np.concatenate([u, v]))
    
    # Use SLSQP first (better for non-smooth min-area objective), then L-BFGS-B for refinement
    for init in initializations:
        if time.time() > deadline:
            break
            
        try:
            # Phase 1: SLSQP optimization
            res1 = minimize(objective, init, method='SLSQP', 
                           options={'maxiter': 200, 'disp': False})
            
            if res1.fun < best_score or best_params is None:
                # Verify and update best
                u = res1.x[0:N]
                v = res1.x[N:2*N]
                u_clipped, v_clipped = clamp_bary(u, v)
                points = np.array([bary_to_cart(ui, vi) for ui, vi in zip(u_clipped, v_clipped)])
                min_area = compute_min_area(points)
                if min_area > best_score:
                    best_score = min_area
                    best_params = res1.x.copy()
            
            # Phase 2: L-BFGS-B refinement from SLSQP result
            if time.time() > deadline:
                continue
                
            res2 = minimize(objective, res1.x, method='L-BFGS-B',
                           options={'maxiter': 500, 'disp': False})
            
            if res2.fun < best_score or best_params is None:
                u = res2.x[0:N]
                v = res2.x[N:2*N]
                u_clipped, v_clipped = clamp_bary(u, v)
                points = np.array([bary_to_cart(ui, vi) for ui, vi in zip(u_clipped, v_clipped)])
                min_area = compute_min_area(points)
                if min_area > best_score:
                    best_score = min_area
                    best_params = res2.x.copy()
                    
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to random if optimization failed completely
        u = rng.random(N)
        v = rng.random(N)
        u, v = clamp_bary(u, v)
        points = np.array([bary_to_cart(ui, vi) for ui, vi in zip(u, v)])
        return points

    u = best_params[0:N]
    v = best_params[N:2*N]
    
    # Final clipping
    u, v = clamp_bary(u, v)
    
    points = np.array([bary_to_cart(ui, vi) for ui, vi in zip(u, v)])
    return points
