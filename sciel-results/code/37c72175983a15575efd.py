import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_REF = H * 1.0 / 2.0 # Area of reference triangle is sqrt(3)/4

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian (x, y).
    Input bary shape: (N, 2) where col 0 is u, col 1 is v.
    w = 1 - u - v
    x = v * 1 + w * 0.5
    y = w * H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """
    Convert Cartesian (x, y) to barycentric (u, v).
    x = v + 0.5(1-u-v) = 0.5 + 0.5v - 0.5u
    y = H(1-u-v) = H - H(u+v)
    From y: u+v = 1 - y/H
    Let S = 1 - y/H
    x = 0.5 + 0.5v - 0.5u
    2x - 1 = v - u
    We have:
    v + u = S
    v - u = 2x - 1
    2v = S + 2x - 1 => v = (S + 2x - 1) / 2
    2u = S - (2x - 1) => u = (S - 2x + 1) / 2
    """
    x = cart[:, 0]
    y = cart[:, 1]
    S = 1.0 - y / H
    v = (S + 2.0 * x - 1.0) / 2.0
    u = (S - 2.0 * x + 1.0) / 2.0
    return np.column_stack([u, v])

def clamp_bary(bary):
    """
    Clamp barycentric coordinates to ensure points are inside the triangle.
    u >= 0, v >= 0, u+v <= 1
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Simple projection: if any coord is negative, set to 0 and renormalize?
    # Or just clip? Clipping might push points out of the sum=1 plane if not careful.
    # Better: use a barrier or penalty in objective.
    # For minimization with L-BFGS-B, we can use bounds.
    # Bounds for u: [0, 1], v: [0, 1] is not enough, need u+v <= 1.
    # L-BFGS-B supports box constraints only.
    # So we must use a penalty method or Nelder-Mead with manual projection.
    # Let's use Nelder-Mead and project back to the feasible region after each step?
    # Or just use a penalty function.
    # Penalty: large value if u<0 or v<0 or u+v>1.
    pass

def get_areas_cartesian(points):
    """
    Calculate areas of all triangles formed by the points.
    points: (N, 2) array
    Returns: array of 165 areas
    """
    # points[tri_idx] gives shape (165, 3, 2)
    p_i = points[tri_idx[:, 0]]
    p_j = points[tri_idx[:, 1]]
    p_k = points[tri_idx[:, 2]]
    
    # Vectorized area calculation
    # 0.5 * | (x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i) |
    dx_j = p_j[:, 0] - p_i[:, 0]
    dy_j = p_j[:, 1] - p_i[:, 1]
    dx_k = p_k[:, 0] - p_i[:, 0]
    dy_k = p_k[:, 1] - p_i[:, 1]
    
    cross = dx_j * dy_k - dx_k * dy_j
    areas = 0.5 * np.abs(cross)
    return areas

def soft_min_neg_area(params, tau=0.01):
    """
    Objective function: minimize -soft_min(areas)
    params: (2N,) array of barycentric coords
    """
    bary = params.reshape(N, 2)
    
    # Penalty for being outside the triangle
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # If any coordinate is significantly negative, penalize
    penalty = 0.0
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        # Quadratic penalty
        penalty = 100.0 * (np.sum(np.minimum(u, 0)**2) + np.sum(np.minimum(v, 0)**2) + np.sum(np.minimum(w, 0)**2))
    
    # Clip to valid range for area calculation to avoid numerical issues
    # For area calc, we just use the coords as is, but if they are way outside, area is large,
    # but we want to maximize min area, so large area is good?
    # No, if points are outside, the configuration is invalid.
    # The penalty handles this.
    
    points = bary_to_cart(bary)
    areas = get_areas_cartesian(points)
    
    # Soft min of areas: log(sum(exp(-area/tau)))
    # We want to MINIMIZE -soft_min
    # -soft_min = -log(sum(exp(-area/tau)))
    # To avoid underflow, shift by max area?
    # exp(-area/tau) can be very small if area is large.
    # We want the smallest area to dominate.
    # Let m = min(areas).
    # sum(exp(-(area-m)/tau)) * exp(m/tau)
    # log(sum) = log(sum(exp(-(area-m)/tau))) + m/tau
    # We want to minimize - (log(sum(exp(-(area-m)/tau))) + m/tau)
    # This is equivalent to minimizing -log(sum(exp(-area/tau)))
    
    # For numerical stability:
    min_area = np.min(areas)
    if min_area < 1e-12:
        # If points are collinear or coincident, area is 0.
        # We want to avoid this.
        # Return large penalty
        return 1000.0 + penalty
        
    # Shift areas to prevent underflow in exp
    # We are computing -log(sum(exp(-A/tau)))
    # = -log( sum( exp(-(A - min_A)/tau) * exp(min_A/tau) ) )
    # = -log( sum( exp(-(A - min_A)/tau) ) ) - min_A/tau
    
    shifted_areas = areas - min_area
    exp_terms = np.exp(-shifted_areas / tau)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = log_sum - min_area / tau
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val + penalty

def generate_random_init():
    # Random points inside triangle using barycentric coords
    # u, v > 0, u+v < 1
    # Generate in unit square, reject if u+v > 1
    bary = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = np.random.random()
            v = np.random.random()
            if u + v < 1.0:
                bary[i, 0] = u
                bary[i, 1] = v
                break
    return bary.flatten()

def generate_grid_init():
    # Simple grid in barycentric space
    # Map a 4x4 grid to the triangle
    bary = []
    for i in range(4):
        for j in range(4):
            u = (i + 0.5) / 4.0
            v = (j + 0.5) / 4.0
            if u + v < 1.0:
                bary.append([u, v])
    if len(bary) >= N:
        # Pick N random ones
        indices = np.random.choice(len(bary), N, replace=False)
        return np.array([bary[i] for i in indices]).flatten()
    else:
        return generate_random_init()

def generate_hex_init():
    # Hexagonal lattice in Cartesian, then convert to Barycentric
    # This often provides a good starting point for packing problems
    points_cart = []
    s = 0.25 # spacing
    h = s * np.sqrt(3) / 2
    for i in range(-1, 10):
        for j in range(-1, 10):
            x = i * s + (j % 2) * s / 2
            y = j * h
            p = np.array([x, y])
            # Check if inside triangle
            # Use barycentric coords to check
            b = cart_to_bary(p.reshape(1, 2))[0]
            if b[0] >= -1e-9 and b[1] >= -1e-9 and b[0]+b[1] <= 1+1e-9:
                points_cart.append(p)
    
    if len(points_cart) < N:
        return generate_random_init()
    
    # Farthest point sampling
    points_cart = np.array(points_cart)
    start_idx = np.random.randint(len(points_cart))
    selected = [points_cart[start_idx]]
    min_dists = np.linalg.norm(points_cart - selected[0], axis=1)
    
    for _ in range(N - 1):
        idx = np.argmax(min_dists)
        selected.append(points_cart[idx])
        d = np.linalg.norm(points_cart - selected[-1], axis=1)
        min_dists = np.minimum(min_dists, d)
        
    bary = cart_to_bary(np.array(selected))
    return bary.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_score = -float('inf')
    best_params = None
    
    # Initializations
    inits = []
    inits.append(generate_hex_init())
    inits.append(generate_grid_init())
    for _ in range(5):
        inits.append(generate_random_init())
        
    np.random.shuffle(inits)
    
    # Optimization loop
    for init in inits:
        if time.time() > deadline:
            break
            
        x0 = init.copy()
        
        # Try a few different tau values (annealing)
        # Start with larger tau (smoother), decrease to sharpen
        taus = [0.1, 0.05, 0.01, 0.005]
        
        current_params = x0
        for tau in taus:
            if time.time() > deadline:
                break
            
            # Define objective for this tau
            def obj(params, tau=tau):
                return soft_min_neg_area(params, tau)
                
            try:
                # Use Nelder-Mead for robustness
                # Or L-BFGS-B
                # Nelder-Mead is derivative-free, good for non-smooth
                res = minimize(obj, current_params, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_score:
                    # Validate
                    bary = res.x.reshape(N, 2)
                    u = bary[:, 0]
                    v = bary[:, 1]
                    w = 1.0 - u - v
                    if np.all(u >= -1e-6) and np.all(v >= -1e-6) and np.all(w >= -1e-6):
                        best_score = res.fun
                        best_params = res.x.copy()
                        current_params = res.x.copy()
            except Exception:
                pass

    if best_params is None:
        best_params = generate_random_init()
        
    # Convert best barycentric params to Cartesian
    bary = best_params.reshape(N, 2)
    # Clamp small negatives to 0
    bary[bary < 0] = 0
    # Normalize if sum > 1? 
    # The optimizer should have kept them inside.
    # Just in case, project to the closest point in the triangle
    # Simple projection: if u+v > 1, scale u,v by 1/(u+v)
    for i in range(N):
        if bary[i, 0] + bary[i, 1] > 1.0:
            s = 1.0 / (bary[i, 0] + bary[i, 1])
            bary[i, 0] *= s
            bary[i, 1] *= s
            
    points = bary_to_cart(bary)
    return points

# EVOLVE-BLOCK-END
