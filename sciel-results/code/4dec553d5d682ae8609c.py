import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all unique triples (i, j, k) with i < j < k
_triples = []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_TRIPLES = np.array(_triples, dtype=np.int32)
NUM_TRIPLES = len(_TRIPLES)

def _bary_to_cart(bary):
    """Convert barycentric (u,v,w) to Cartesian (x,y)."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates.
    Returns the minimum normalized triangle area.
    """
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    
    # dets = det([r1, r2, r3])
    cross_r2_r3 = np.cross(r2, r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    
    areas = 0.5 * np.abs(dets)
    return np.min(areas)

def _project_bary(bary):
    """
    Project barycentric coordinates to the simplex.
    """
    b = bary.copy()
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    b = np.clip(b, 0, 1)
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    return b

def _softmin_area_bary(p, beta):
    """
    Returns the softmin of the areas and the gradient with respect to p.
    p: (N, 3)
    beta: temperature parameter. Higher beta -> closer to hard min.
    """
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    
    # Compute signed determinants
    cross_r2_r3 = np.cross(r2, r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    
    # Areas are 0.5 * |det|
    # Use smooth absolute value to handle non-differentiability at 0
    eps = 1e-12
    smooth_abs = np.sqrt(dets**2 + eps)
    areas = 0.5 * smooth_abs
    
    # Compute logsumexp(-beta * areas)
    x = -beta * areas
    x_max = np.max(x)
    lse = x_max + np.log(np.sum(np.exp(x - x_max)))
    
    soft_min_val = lse / beta
    
    # Gradient calculation
    weights = np.exp(x - x_max)
    weights /= np.sum(weights)
    
    # dx_i/d(det_i) = -beta * 0.5 * det_i / sqrt(det_i^2 + eps)
    grad_det = -beta * 0.5 * dets / smooth_abs
    
    coeff = weights * grad_det
    
    # Scatter gradients back to p
    grad_p = np.zeros_like(p)
    
    g_r1 = coeff[:, np.newaxis] * np.cross(r2, r3)
    g_r2 = coeff[:, np.newaxis] * np.cross(r3, r1)
    g_r3 = coeff[:, np.newaxis] * np.cross(r1, r2)
    
    np.add.at(grad_p, _TRIPLES[:, 0], g_r1)
    np.add.at(grad_p, _TRIPLES[:, 1], g_r2)
    np.add.at(grad_p, _TRIPLES[:, 2], g_r3)
    
    # The objective is soft_min_val. We want to maximize it, so minimize -soft_min_val.
    # The gradient of -soft_min_val is -grad_p / beta.
    # Wait, in SLSQP we provide the gradient of the objective function we are minimizing.
    # If we define the objective as -soft_min_val, the gradient is -grad_p / beta.
    
    grad_obj = -grad_p / beta
    
    return -soft_min_val, grad_obj.flatten()

def _objective_and_grad(x_flat, beta):
    bary = x_flat.reshape(N, 3)
    # Note: SLSQP with constraints will keep us in the simplex if constraints are tight.
    # However, numerical drift can happen. We project for safety in evaluation, 
    # but for gradient-based methods, it's often better to let the constraints handle it.
    # But _softmin_area_bary assumes valid barycentric coords.
    # Let's project to ensure validity during the evaluation step inside the optimizer.
    # Actually, if we use constraints, the points should stay in the simplex.
    # Let's just evaluate on the raw x_flat interpreted as barycentric coords.
    # The constraints will prevent leaving the simplex.
    
    val, grad = _softmin_area_bary(bary, beta)
    return val, grad

def _generate_inits():
    inits = []
    
    # 1. Random Dirichlet inits
    rng = np.random.default_rng(123)
    for _ in range(5):
        inits.append(rng.dirichlet([1, 1, 1], size=N))
        
    # 2. Structured Grid (Hexagonal packing approximation)
    # Divide the triangle into smaller triangles.
    # For 11 points, a 3x3 grid + center + corners?
    # Let's try a few specific patterns.
    
    # Pattern A: 3 points near vertices, 6 around center, 1 center
    center = np.array([1/3, 1/3, 1/3])
    inits.append(np.tile(center, (N, 1))) # Degenerate, will be pushed apart
    
    # Pattern B: Vertices and midpoints
    vertices = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    midpoints = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    # 3 vertices + 3 midpoints + 5 random
    rest = rng.dirichlet([1, 1, 1], size=5)
    inits.append(np.vstack([vertices, midpoints, rest]))
    
    # Pattern C: Regular grid in barycentric coords
    # k=3 grid (4 points per side) -> 10 points? 
    # Let's just use random perturbations of the center
    for _ in range(3):
        p = np.tile(center, (N, 1)) + rng.normal(0, 0.1, (N, 3))
        p = _project_bary(p)
        inits.append(p)
        
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_area = 0.0
    best_bary = None
    
    inits = _generate_inits()
    
    # SLSQP requires constraints
    # Constraints: sum(bary[i]) = 1 for all i, and bary[i] >= 0 for all i, j
    # Since we are using x_flat of size N*3, we need to map constraints correctly.
    
    # Equality constraints: sum of each row is 1
    # Inequality constraints: each element is >= 0
    
    # However, SLSQP handles linear constraints efficiently.
    # Let's define them.
    
    # Equality: For each i in 0..N-1, x[3*i] + x[3*i+1] + x[3*i+2] = 1
    # This can be represented as a linear constraint: A_eq @ x = b_eq
    # A_eq is N x 3N.
    
    A_eq = np.zeros((N, 3 * N))
    b_eq = np.ones(N)
    for i in range(N):
        A_eq[i, 3*i] = 1
        A_eq[i, 3*i+1] = 1
        A_eq[i, 3*i+2] = 1
        
    # Inequality: x >= 0
    # A_ub @ x <= b_ub
    # -x <= 0
    A_ub = -np.eye(3 * N)
    b_ub = np.zeros(3 * N)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: A_eq @ x - b_eq, 'jac': lambda x: A_eq},
        {'type': 'ineq', 'fun': lambda x: -A_ub @ x - b_ub, 'jac': lambda x: -A_ub}
    ]
    
    # We will use a moderate beta. SLSQP is good at handling the soft-min if beta isn't too high.
    # If beta is too high, the gradient becomes very sparse and noisy.
    # Let's try beta=50 and beta=200.
    
    betas = [20, 100]
    
    for init in inits:
        if time.time() > deadline - 5.0:
            break
            
        x0 = init.flatten()
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            try:
                res = minimize(
                    _objective_and_grad,
                    x0,
                    args=(beta,),
                    method='SLSQP',
                    jac=True,
                    bounds=[(0, 1)] * (3 * N),
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
                )
                
                if res.success or res.fun < 0: # res.fun is -soft_min_area
                    refined_bary = _project_bary(res.x.reshape(N, 3))
                    true_area = _min_area_bary(refined_bary)
                    
                    if true_area > best_area:
                        best_area = true_area
                        best_bary = refined_bary.copy()
                        
            except Exception:
                continue
                
    if best_bary is None:
        # Fallback
        rng = np.random.default_rng(0)
        best_bary = rng.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(best_bary)
        
    return _bary_to_cart(best_bary)
