"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Barycentric Coordinate Parameterization:
   - Represents points using barycentric coordinates (u, v, w) with u+v+w=1, u,v,w >= 0.
   - This naturally handles the triangle boundary constraints without projection.
2. Soft-Min Annealing:
   - Uses a differentiable approximation of the minimum area (soft-min) to allow 
     gradient-based optimization (L-BFGS-B).
   - Anneals the temperature parameter to sharpen the minimum.
3. Multi-Start:
   - Uses multiple random initializations and greedy bases.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

# Barycentric to Cartesian conversion
# Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
# P = u*A + v*B + w*C = v*(1,0) + w*(0.5, sqrt(3)/2)
# x = v + 0.5*w
# y = sqrt(3)/2 * w
def _bary_to_cart(bary_pts):
    """
    bary_pts: shape (N, 3) with columns u, v, w. u+v+w=1.
    Returns: shape (N, 2) Cartesian coordinates.
    """
    v = bary_pts[:, 1]
    w = bary_pts[:, 2]
    x = v + 0.5 * w
    y = S3H * w
    return np.column_stack((x, y))

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points in Cartesian coords."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _soft_min_area_cart(pts, temperature):
    """
    Compute the soft minimum of triangle areas.
    Soft min approximates the true min as temperature -> 0.
    """
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -T * log(sum(exp(-area/T)))
    # To avoid overflow, use logsumexp
    # min(a) = -T * log(sum(exp(-a/T)))
    # Let z = -areas / T
    # logsumexp(z) = log(sum(exp(z)))
    # soft_min = -T * logsumexp(z)
    
    if temperature < 1e-6:
        return np.min(areas)
        
    z = -areas / temperature
    # Shift for numerical stability
    z_max = np.max(z)
    log_sum = z_max + np.log(np.sum(np.exp(z - z_max)))
    soft_min = -temperature * log_sum
    
    return soft_min

def _objective_bary(x_flat, temperature):
    """
    Objective function for minimization in barycentric coordinates.
    x_flat: shape (N*3,)
    We enforce u+v+w=1 by parameterizing with 2 coordinates? 
    No, L-BFGS-B works better with explicit bounds. 
    We can use 2 coordinates per point (v, w) and set u = 1-v-w.
    Bounds: v>=0, w>=0, v+w<=1.
    This reduces dimensionality from 33 to 22.
    """
    # x_flat contains 2 coordinates per point: v, w
    # u is implicit: 1 - v - w
    v = x_flat[0::2]
    w = x_flat[1::2]
    
    # Construct barycentric points
    bary_pts = np.column_stack((1.0 - v - w, v, w))
    
    # Convert to Cartesian
    pts = _bary_to_cart(bary_pts)
    
    # Calculate soft min area
    area = _soft_min_area_cart(pts, temperature)
    
    return -area

def _random_bary(rng):
    """Generate random barycentric coordinates (v, w) for N points."""
    # Generate random v, w such that v>=0, w>=0, v+w<=1
    # Use Dirichlet distribution or simple rejection
    # Simple rejection is fine for N=11
    coords = []
    for _ in range(N):
        while True:
            v = rng.random()
            w = rng.random()
            if v + w <= 1.0:
                coords.append([v, w])
                break
    return np.array(coords).flatten()

def _greedy_init_bary(rng):
    """Greedy initialization in Cartesian space, converted to Barycentric."""
    pts_cart = []
    
    def _random_point_in_triangle(rng):
        u = rng.random()
        v = rng.random()
        while u + v > 1:
            u = rng.random()
            v = rng.random()
        x = u + 0.5 * v
        y = S3H * v
        return np.array([x, y])

    p0 = _random_point_in_triangle(rng)
    pts_cart.append(p0)
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        
        for _ in range(200): # Reduced candidates for speed in init
            cand = _random_point_in_triangle(rng)
            dists = np.linalg.norm(np.array(pts_cart) - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        
        if best_pos is not None:
            pts_cart.append(best_pos)
            
    pts_cart = np.array(pts_cart)
    
    # Convert Cartesian to Barycentric (v, w)
    # x = v + 0.5*w
    # y = S3H*w  => w = y / S3H
    # v = x - 0.5*w
    w = pts_cart[:, 1] / S3H
    v = pts_cart[:, 0] - 0.5 * w
    
    # Clip to valid range due to floating point errors
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    # Ensure v+w <= 1
    mask = v + w > 1
    if np.any(mask):
        # Scale down
        scale = 1.0 / (v[mask] + w[mask])
        v[mask] *= scale
        w[mask] *= scale
        
    coords = np.column_stack((v, w)).flatten()
    return coords

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Multi-start L-BFGS-B with Barycentric coordinates and Soft-Min Annealing.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts_cart = None
    best_score = -1.0
    
    # Generate initializations
    initializations = []
    
    # 1. Greedy bases
    for _ in range(5):
        try:
            init = _greedy_init_bary(rng)
            initializations.append(init)
        except Exception:
            pass
            
    # 2. Random bases
    for _ in range(10):
        try:
            init = _random_bary(rng)
            initializations.append(init)
        except Exception:
            pass
            
    # Shuffle
    rng.shuffle(initializations)
    
    # Optimization parameters
    # Bounds for v, w: [0, 1]
    # Note: v+w<=1 is a linear constraint, but L-BFGS-B handles box constraints.
    # We rely on the objective to penalize invalid points? 
    # No, _bary_to_cart assumes valid barycentric coords.
    # If v+w > 1, u becomes negative. The point is outside the triangle.
    # We should add a penalty for v+w > 1 or use a method that handles linear constraints.
    # L-BFGS-B only supports box constraints.
    # To strictly enforce v+w<=1, we can parameterize differently or add a penalty.
    # Given the smoothness, a penalty term is robust.
    
    def _objective_with_penalty(x_flat, temperature):
        v = x_flat[0::2]
        w = x_flat[1::2]
        
        # Penalty for v+w > 1
        excess = np.maximum(0, v + w - 1.0)
        penalty = 100.0 * np.sum(excess**2)
        
        # Also penalty for v<0 or w<0 (though bounds handle this, just in case)
        # Bounds will prevent v<0, w<0.
        
        base_obj = _objective_bary(x_flat, temperature)
        return base_obj + penalty

    for i, init in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Annealing schedule
        # Start with high temperature, lower it
        # We will do a few steps with decreasing T
        
        # To save time, we'll do a single optimization run with a moderate T 
        # or a short annealing chain.
        # Let's try a short chain: T = 0.1, 0.01, 0.001
        
        x_current = init
        current_score = -1.0
        
        # Evaluate initial
        bary_pts = np.column_stack((1.0 - x_current[0::2] - x_current[1::2], x_current[0::2], x_current[1::2]))
        pts_cart = _bary_to_cart(bary_pts)
        if np.all(pts_cart[:, 0] >= -1e-9) and np.all(pts_cart[:, 0] <= 1+1e-9) and np.all(pts_cart[:, 1] >= -1e-9) and np.all(pts_cart[:, 1] <= S3H+1e-9):
             current_score = _min_triangle_area_cart(pts_cart)
             if current_score > best_score:
                 best_score = current_score
                 best_pts_cart = pts_cart.copy()

        temperatures = [0.05, 0.01, 0.001]
        
        for T in temperatures:
            if time.time() - start_time > DEADLINE - 10:
                break
                
            try:
                res = minimize(
                    _objective_with_penalty, 
                    x_current, 
                    args=(T,),
                    method='L-BFGS-B', 
                    bounds=[(0, 1)] * (2 * N),
                    options={
                        'maxiter': 200, # Limit iterations per T
                        'ftol': 1e-8,
                        'gtol': 1e-6
                    }
                )
                
                x_current = res.x
                
                # Evaluate result
                bary_pts = np.column_stack((1.0 - x_current[0::2] - x_current[1::2], x_current[0::2], x_current[1::2]))
                pts_cart = _bary_to_cart(bary_pts)
                
                # Check validity
                valid = True
                if np.any(pts_cart[:, 0] < -1e-6) or np.any(pts_cart[:, 0] > 1+1e-6):
                    valid = False
                if np.any(pts_cart[:, 1] < -1e-6) or np.any(pts_cart[:, 1] > S3H+1e-6):
                    valid = False
                
                if valid:
                    score = _min_triangle_area_cart(pts_cart)
                    if score > best_score:
                        best_score = score
                        best_pts_cart = pts_cart.copy()
                        
            except Exception:
                # If optimization fails, continue to next T or break
                pass

    # Fallback
    if best_pts_cart is None:
        # Use a simple grid or random
        best_pts_cart = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
            [0.5, 0.0], [0.25, S3H/2], [0.75, S3H/2],
            [0.5, S3H/3],
            [0.25, 0.0], [0.75, 0.0],
            [0.125, S3H/4], [0.875, S3H/4]
        ])
        # Project to be safe
        # This fallback is just to ensure we return something
        # Ideally, the loop above finds something.
        
    return best_pts_cart

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area_cart(pts)}")
