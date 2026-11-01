import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

# Precompute unit hexagon vertices (side 1)
# Vertices at angles 0, 60, ..., 300
UNIT_VERTS = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))] 
    for a in range(0, 360, 60)
])

# Outer hexagon normals (angles 30, 90, 150, 210, 270, 330)
# These are the normals to the sides of the outer hexagon
OUTER_NORMALS = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))] 
    for a in range(30, 360, 60)
])

def get_hex_vertices(center, angle_rad):
    """Return vertices of a unit hexagon at given center and rotation."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return center + UNIT_VERTS @ R.T

def smooth_max(values, temperature=0.01):
    """Softmax approximation of max for smooth optimization."""
    max_val = np.max(values)
    if temperature == 0:
        return max_val
    exp_vals = np.exp((values - max_val) / temperature)
    return max_val + temperature * np.log(np.sum(exp_vals))

def objective(params, temperature=0.01):
    """
    params: [x1, y1, theta1, x2, y2, theta2, ..., xN, yN, thetaN, S]
    Returns smooth estimate of outer side length S.
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return 1e6
        
    # Extract S (last parameter)
    S = params[-1]
    
    # Extract positions and angles
    # We need to check if all vertices of all inner hexagons are inside the outer hexagon.
    # Outer hexagon is centered at (0,0) with side S and rotation 0.
    # Constraint: For all vertices v of all inner hexagons, and all normals n of outer hex:
    # dot(v, n) <= S * APOTHEM
    
    apothem_outer = S * APOTHEM
    violations = []
    
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i + 1]
        theta = params[3*i + 2]
        
        verts = get_hex_vertices(np.array([cx, cy]), theta)
        
        # Project all vertices onto all 6 outer normals
        # verts is (6, 2), OUTER_NORMALS is (6, 2)
        # We want max over all vertices and all normals of dot(v, n)
        # dot(v, n) for all v, n is (6, 6) matrix
        projections = verts @ OUTER_NORMALS.T
        
        # The max projection must be <= apothem_outer
        # Violation = max_projection - apothem_outer
        # We want to minimize max(0, violation)
        # Using smooth max on the projections directly, then subtracting apothem?
        # Or smooth max of (projections - apothem_outer)?
        # Let's use smooth max of the projections, then compare to apothem.
        # Actually, to make it smooth and differentiable everywhere, we can just take the smooth max of the projections.
        # The constraint is smooth_max(projections) <= apothem_outer.
        # If we minimize smooth_max(projections), we are minimizing the required apothem.
        # So the objective can just be the smooth_max of the projections.
        # Then S = smooth_max / APOTHEM.
        # But we have S as a variable? No, if we minimize the projections, S is derived.
        # Let's just return the smooth max of the projections. The optimal S will be this value / APOTHEM.
        
        max_proj = smooth_max(projections.ravel(), temperature)
        violations.append(max_proj)
        
    # We want the maximum over all hexagons of their max projection
    # This determines the required apothem.
    global_max_proj = smooth_max(np.array(violations), temperature)
    
    # The side length S is global_max_proj / APOTHEM
    # We want to minimize S, which is equivalent to minimizing global_max_proj.
    return global_max_proj / APOTHEM

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial guess: The rigid lattice solution from the previous iteration
    # Axial coordinates for N=11 (Center + Ring 1 + Ring 2 subset)
    # We use the best candidate from the previous logic or just a standard dense packing.
    # Let's use the standard hexagonal packing for 11.
    # Center (0,0)
    # Ring 1: 6 hexes
    # Ring 2: 4 hexes? No, 11 = 1 + 6 + 4.
    # Let's initialize with a known valid configuration.
    
    # Axial to Cartesian: x = sqrt(3)(q + r/2), y = 1.5r
    axial_pts = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (1,1), (-1,2), (-2,1), (2,-1) # Same as ring2_opt1 in original
    ]
    
    initial_params = []
    for q, r in axial_pts:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        initial_params.extend([x, y, 0.0])
        
    # Initial S estimate
    # Calculate max projection for this config
    def calc_max_proj(params):
        max_p = 0
        for i in range(N):
            cx, cy, th = params[3*i:3*i+3]
            verts = get_hex_vertices(np.array([cx, cy]), th)
            projs = verts @ OUTER_NORMALS.T
            max_p = max(max_p, np.max(projs))
        return max_p
        
    init_max_proj = calc_max_proj(initial_params)
    initial_S = init_max_proj / APOTHEM
    initial_params.append(initial_S) # Actually, we don't need S in params if we minimize projections.
    
    # Let's refine the objective to just minimize the max projection.
    # So params are 3N variables.
    # Objective returns max_proj / APOTHEM.
    
    # Remove S from params logic in objective above? 
    # My objective function above takes S as last param but ignores it?
    # No, I defined objective to take params with S at end, but it calculates S from projections.
    # Let's simplify: params are just 3N. Objective returns required S.
    
    def objective_no_S(params, temperature=0.01):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e6
        violations = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i + 1]
            theta = params[3*i + 2]
            verts = get_hex_vertices(np.array([cx, cy]), theta)
            projections = verts @ OUTER_NORMALS.T
            max_proj = smooth_max(projections.ravel(), temperature)
            violations.append(max_proj)
        global_max_proj = smooth_max(np.array(violations), temperature)
        return global_max_proj / APOTHEM

    # Multi-start
    best_S = float('inf')
    best_params = None
    
    starts = [initial_params]
    
    # Add a few random starts
    np.random.seed(42)
    for _ in range(3):
        rand_params = []
        for q, r in axial_pts:
            x = math.sqrt(3) * (q + r / 2.0) + np.random.uniform(-0.2, 0.2)
            y = 1.5 * r + np.random.uniform(-0.2, 0.2)
            th = np.random.uniform(-0.1, 0.1)
            rand_params.extend([x, y, th])
        starts.append(rand_params)

    for start in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        try:
            # Use L-BFGS-B
            res = minimize(
                objective_no_S, 
                start, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception as e:
            pass

    if best_params is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Extract results
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i + 1])
        angle_deg = math.degrees(float(best_params[3*i + 2]))
        inner.append([x, y, angle_deg])
        
    center = [0.0, 0.0]
    side = float(best_S)
    rotation = 0.0 # Outer hexagon is fixed at 0 rotation
    
    return inner, center, side, rotation
