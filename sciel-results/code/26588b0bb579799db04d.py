"""Model-and-optimize approach for the Heilbronn-11 triangle problem using Barycentric coordinates and Soft-Min Gradient Ascent."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)  # 165 for N=11

DEADLINE = 145  # Safety margin from 150s limit

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (N, 3) to Cartesian coordinates (N, 2).
    bary[i] = (alpha, beta, gamma) where alpha+beta+gamma=1.
    Point = alpha*V0 + beta*V1 + gamma*V2
    """
    # V0 is (0,0), so it contributes 0
    # x = beta*1 + gamma*0.5
    # y = beta*0 + gamma*(sqrt(3)/2)
    beta = bary[:, 1]
    gamma = bary[:, 2]
    x = beta + 0.5 * gamma
    y = S3H * gamma
    return np.column_stack([x, y])

def _cart_to_bary(cart):
    """
    Convert Cartesian coordinates (N, 2) to barycentric coordinates (N, 3).
    This is the inverse of _bary_to_cart.
    y = gamma * S3H  => gamma = y / S3H
    x = beta + 0.5 * gamma => beta = x - 0.5 * gamma
    alpha = 1 - beta - gamma
    """
    x = cart[:, 0]
    y = cart[:, 1]
    gamma = y / S3H
    beta = x - 0.5 * gamma
    alpha = 1.0 - beta - gamma
    return np.column_stack([alpha, beta, gamma])

def _compute_areas_cart(pts):
    """Compute all triangle areas from Cartesian points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5

def _soft_min_area_bary(bary_flat, beta):
    """
    Calculate the negative soft-minimum of triangle areas.
    bary_flat is a 1D array of length N*3.
    We maximize the min area, so we minimize -soft_min_area.
    """
    bary = bary_flat.reshape(N, 3)
    # Normalize to ensure they sum to 1 (L-BFGS-B might drift slightly)
    # Actually, L-BFGS-B with bounds [0, 1] doesn't enforce sum=1.
    # It's better to parameterize with 2 coordinates per point to avoid the sum constraint.
    # Let's switch representation: use (beta, gamma) for each point. Alpha = 1 - beta - gamma.
    # Bounds: beta >= 0, gamma >= 0, beta + gamma <= 1.
    
    # This function is currently expected to take the 2D representation.
    # I will redefine the optimization variable to be (N, 2) representing (beta, gamma).
    pass

def _objective_soft_min(params, beta):
    """
    params: (N, 2) array of (beta, gamma) barycentric coordinates.
    Objective: -soft_min(triangle_areas)
    """
    b = params[:, 0]
    g = params[:, 1]
    
    # Calculate Cartesian coordinates
    x = b + 0.5 * g
    y = S3H * g
    pts = np.column_stack([x, y])
    
    # Calculate areas
    areas = _compute_areas_cart(pts)
    
    # Soft min
    # To avoid overflow in exp, subtract max
    max_a = np.max(areas)
    if max_a == 0:
        return 0.0
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    
    return -soft_min

def _grad_soft_min(params, beta):
    """
    Gradient of the objective function.
    This is complex to derive analytically for the soft min of cross products.
    However, L-BFGS-B can approximate gradients if not provided, but it's slower.
    Providing a numerical gradient or using 'approx' might be safer for correctness.
    Let's try to provide a numerical gradient or rely on L-BFGS-B's default.
    Actually, let's just use the objective and let scipy handle the gradient via finite differences 
    if analytic is too prone to error. But for speed, analytic is better.
    
    Let's stick to the objective only and let L-BFGS-B use numerical gradients.
    For N=11, dimension is 22. Finite difference gradient is 23 evaluations.
    If each evaluation is ~0.1ms, gradient is ~2.3ms.
    L-BFGS-B might take 100-500 iterations. 500 * 2.3ms = 1.15s per run.
    We have ~145s. So we can do ~100-200 restarts.
    """
    return None # Use numerical

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_score = 0.0
    best_params = None
    
    # Bounds for (beta, gamma):
    # beta >= 0, gamma >= 0, beta + gamma <= 1
    # L-BFGS-B supports box constraints. We can't directly do beta+gamma<=1 in box constraints.
    # However, we can use a change of variables or just clip inside the objective?
    # No, clipping breaks differentiability.
    # Alternative: Use barycentric coordinates (alpha, beta) with gamma = 1-alpha-beta.
    # Constraints: alpha>=0, beta>=0, alpha+beta<=1.
    # Still has the sum constraint.
    
    # Better approach: Use 2D Cartesian coordinates and project to triangle inside the objective?
    # If we project, the objective becomes non-smooth at the boundaries.
    
    # Let's use the (beta, gamma) representation but handle the constraint beta+gamma<=1 
    # by mapping from an unconstrained space? No, that's complex.
    
    # Actually, L-BFGS-B can handle simple bounds. 
    # If we use (beta, gamma), the constraint is a polytope.
    # We can use a barrier function or just project in the objective?
    # Let's try a different parameterization:
    # Let u, v be unconstrained. 
    # beta = sigmoid(u), gamma = sigmoid(v) * (1-beta)? No.
    
    # Simplest robust approach: 
    # Use Cartesian coordinates (x, y) with bounds:
    # 0 <= y <= S3H
    # y/S3 <= x <= 1 - y/S3
    # This is a non-box constraint.
    
    # Let's go back to Barycentric (alpha, beta, gamma) but optimize only 2 of them, 
    # and enforce the third via calculation.
    # Variables: alpha, beta for each point. gamma = 1 - alpha - beta.
    # Constraints: alpha >= 0, beta >= 0, alpha + beta <= 1.
    # This is still a non-box constraint for L-BFGS-B.
    
    # Workaround: Use a transformation that maps a box to the triangle.
    # Map (u, v) in [0,1]^2 to the triangle.
    # If u+v > 1, reflect? No, that's not one-to-one.
    
    # Let's use the standard trick: 
    # x = (1-sqrt(1-u^2-v^2))/2 ? No.
    
    # Okay, let's use `scipy.optimize.minimize` with method='L-BFGS-B' 
    # and just use the (beta, gamma) coordinates, and inside the objective, 
    # if beta+gamma > 1, we return a large penalty. 
    # This is a penalty method. It works well if the penalty is steep enough.
    
    def objective_with_penalty(params, beta):
        b = params[:, 0]
        g = params[:, 1]
        
        # Check constraint: b >= 0, g >= 0, b+g <= 1
        # L-BFGS-B enforces b>=0, g>=0 via bounds.
        # We need to handle b+g <= 1.
        violation = np.maximum(0, b + g - 1.0)
        penalty = 100.0 * np.sum(violation ** 2)
        
        if np.any(violation > 1e-4):
            return 1e6 + penalty
        
        return _objective_soft_min(params, beta) + penalty

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Initial guesses
    inits = []
    
    # 1. Symmetric-ish start
    init1_b = np.zeros((N, 2))
    init1_g = np.zeros((N, 2))
    # Place points roughly in a grid
    # We need to generate valid (b, g) pairs
    # Let's just use random valid points
    def random_valid_point(rng):
        while True:
            b = rng.random()
            g = rng.random()
            if b + g <= 1.0:
                return b, g
    
    for i in range(5):
        init = np.array([random_valid_point(rng) for _ in range(N)])
        inits.append(init)
        
    # Add a specific structured start: vertices and midpoints
    # Vertices in (b,g):
    # V0(0,0) -> (0,0)
    # V1(1,0) -> (1,0)
    # V2(0.5, S3H) -> (0,1)
    # Midpoints:
    # (0.5, 0) -> (0.5, 0)
    # (0.75, S3H/2) -> (0.75, 0.25)? 
    # x = b + 0.5g = 0.75, y = S3H g = S3H/2 => g=0.5, b=0.5. Sum=1. OK.
    # (0.25, S3H/2) -> g=0.5, b=0.0. Sum=0.5. OK.
    # Center (1/3, 1/3) in bary -> b=1/3, g=1/3?
    # x = 1/3 + 1/6 = 1/2. y = S3H/3. Correct.
    
    struct_init = np.array([
        [0.0, 0.0], # V0
        [1.0, 0.0], # V1
        [0.0, 1.0], # V2
        [0.5, 0.0], # Mid 0-1
        [0.5, 0.5], # Mid 1-2
        [0.0, 0.5], # Mid 0-2
        [1/3, 1/3], # Center
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.0],
        [0.0, 0.25],
    ])
    # Check validity
    for row in struct_init:
        if row[0] + row[1] > 1.0 + 1e-9:
             # Clip
             s = row[0] + row[1]
             row[0] /= s
             row[1] /= s
    inits.append(struct_init)
    
    # Optimization Loop
    # We will run multiple restarts.
    # Start with low beta (smooth) and increase beta (sharp) for refinement.
    
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    current_best_params = inits[0]
    
    for beta in betas:
        # For each beta, run a few restarts from the current best and some randoms
        restarts = 3
        for r in range(restarts):
            if time.time() > DEADLINE:
                break
                
            if r == 0:
                start_params = current_best_params.copy()
            else:
                # Perturb current best
                start_params = current_best_params + rng.normal(0, 0.05, size=(N, 2))
                # Clip to box [0,1]
                start_params = np.clip(start_params, 0.0, 1.0)
                # Handle sum constraint roughly by scaling if needed, or let penalty handle it
                # Actually, if sum > 1, penalty will kick in.
                
            # Run L-BFGS-B
            res = minimize(
                objective_with_penalty,
                start_params,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < 1e5:
                # Evaluate true score (hard min)
                final_params = res.x
                # Ensure validity
                b = final_params[:, 0]
                g = final_params[:, 1]
                # If any point violates b+g<=1 significantly, skip or project
                if np.any(b + g > 1.0 + 1e-5):
                    # Project to triangle
                    # Simple projection: scale b,g by (1/(b+g)) if b+g>1
                    for i in range(N):
                        s = b[i] + g[i]
                        if s > 1.0:
                            b[i] /= s
                            g[i] /= s
                    final_params[:, 0] = b
                    final_params[:, 1] = g
                    
                cart_pts = _bary_to_cart(np.column_stack([1-b-g, b, g]))
                areas = _compute_areas_cart(cart_pts)
                score = np.min(areas)
                
                if score > best_score:
                    best_score = score
                    current_best_params = final_params.copy()
                    
    # Final refinement with high beta on the best found
    if time.time() < DEADLINE - 5:
        beta_final = 200.0
        res = minimize(
            objective_with_penalty,
            current_best_params,
            args=(beta_final,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < 1e5:
            final_params = res.x
            b = final_params[:, 0]
            g = final_params[:, 1]
            if np.any(b + g > 1.0 + 1e-5):
                for i in range(N):
                    s = b[i] + g[i]
                    if s > 1.0:
                        b[i] /= s
                        g[i] /= s
                final_params[:, 0] = b
                final_params[:, 1] = g
                
            cart_pts = _bary_to_cart(np.column_stack([1-b-g, b, g]))
            areas = _compute_areas_cart(cart_pts)
            score = np.min(areas)
            
            if score > best_score:
                best_score = score
                current_best_params = final_params.copy()
                
    # Convert best params to Cartesian
    b = current_best_params[:, 0]
    g = current_best_params[:, 1]
    # Final safety projection
    for i in range(N):
        s = b[i] + g[i]
        if s > 1.0:
            b[i] /= s
            g[i] /= s
    alpha = 1 - b - g
    bary = np.column_stack([alpha, b, g])
    cart_pts = _bary_to_cart(bary)
    
    return cart_pts

# EVOLVE-BLOCK-END
