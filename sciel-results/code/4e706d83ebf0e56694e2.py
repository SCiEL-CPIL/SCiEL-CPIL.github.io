import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_EQUI = 0.5 * H  # Area of the unit equilateral triangle

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric coords (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    # Inverse of bary_to_cart
    # x = v + 0.5(1-u-v) = 0.5 - 0.5u + 0.5v
    # y = (1-u-v)H = H - Hu - Hv
    # From y: u+v = (H-y)/H = 1 - y/H
    # From x: 0.5(u-v) = 0.5 - x => u-v = 1 - 2x
    # u = 0.5 * ( (u+v) + (u-v) ) = 0.5 * (1 - y/H + 1 - 2x) = 1 - x - 0.5*y/H
    # v = 0.5 * ( (u+v) - (u-v) ) = 0.5 * (1 - y/H - 1 + 2x) = x - 0.5*y/H
    u = 1.0 - x - 0.5 * y / H
    v = x - 0.5 * y / H
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(x):
    """
    Objective function for minimization.
    We want to maximize the minimum area, so we minimize the negative of the minimum area.
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_hexagonal_init():
    """
    Generate a hexagonal grid initialization.
    This provides a structured, spread-out starting point.
    """
    pts = []
    # 11 points in a hexagonal-like arrangement
    # Center: (0.5, 0.5) in barycentric? No, centroid is (1/3, 1/3).
    # Let's use a perturbed grid that respects the triangular symmetry.
    
    # A good heuristic for 11 points is a 4x4 grid minus corners, or similar.
    # Let's try a specific known good configuration approximation.
    # 3 points on each edge (including vertices) = 9 points.
    # Plus 2 interior points.
    
    # Edge points (barycentric):
    # Bottom edge (w=0): (1/3, 0), (2/3, 0) -> wait, vertices are (1,0), (0,1), (0,0) in barycentric?
    # My barycentric system:
    # V0=(1,0,0) -> (1,0) in (u,v)
    # V1=(0,1,0) -> (0,1) in (u,v)
    # V2=(0,0,1) -> (0,0) in (u,v)
    
    # Vertices:
    pts.append([1.0, 0.0])
    pts.append([0.0, 1.0])
    pts.append([0.0, 0.0])
    
    # Edge midpoints and 1/3, 2/3 points
    # Edge V0-V1 (u+v=1, w=0):
    pts.append([0.5, 0.5])
    pts.append([1.0/3.0, 2.0/3.0])
    pts.append([2.0/3.0, 1.0/3.0])
    
    # Edge V1-V2 (u=0, v+w=1):
    pts.append([0.0, 0.5])
    pts.append([0.0, 1.0/3.0])
    pts.append([0.0, 2.0/3.0])
    
    # Edge V2-V0 (v=0, u+w=1):
    pts.append([0.5, 0.0])
    pts.append([1.0/3.0, 0.0])
    pts.append([2.0/3.0, 0.0])
    
    # That's 9 points. We need 2 more.
    # Interior points: Centroid (1/3, 1/3) and maybe another?
    # Actually, the 9 points above include vertices.
    # Let's just use a random jittered version of a dense grid for DE bounds.
    # For DE, we just need valid bounds.
    
    bary = np.array(pts)
    # Jitter slightly to break symmetry
    rng = np.random.default_rng(42)
    bary += rng.normal(0, 0.01, bary.shape)
    return clamp_bary(bary)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Define bounds for Differential Evolution
    # Barycentric coordinates u, v must be >= 0 and u+v <= 1.
    # Simple box bounds: u in [0, 1], v in [0, 1].
    # The clamp function will handle the triangular constraint, but DE works better
    # if the search space is tighter.
    # We can use a transformation or just rely on penalty/clamping.
    # For simplicity and robustness, we use box bounds [0, 1] for both u and v.
    # The clamp function projects invalid points back into the triangle.
    
    bounds = [(0.0, 1.0), (0.0, 1.0)] * N
    
    # Strategy:
    # 1. Run Differential Evolution for the majority of the time to find a good global basin.
    # 2. Use the best result from DE as the start for a BFGS local refinement.
    
    best_val = -np.inf
    best_bary = None
    
    # Initial guess for DE
    x0 = generate_hexagonal_init().flatten()
    
    # Run DE
    # maxiter and popsize are tuned to fit within ~120s.
    # DE is stochastic, so we set a fixed seed for reproducibility in testing,
    # but in production, random seed is fine.
    
    try:
        # We'll run DE for a specific number of iterations.
        # To strictly respect the time limit, we could use a callback,
        # but DE's internal loop is hard to interrupt cleanly without modifying scipy.
        # Instead, we estimate: 140 points in pop, 100 iterations = 14000 evaluations.
        # Each eval is ~10-50us. Total ~0.5s. This is very fast.
        # We can afford many more iterations.
        
        # Let's use a larger population and more iterations.
        # popsize=200, maxiter=200 -> 40,000 evals. Still fast.
        
        res_de = differential_evolution(
            hard_min_obj,
            bounds,
            strategy='best1bin',
            maxiter=1000,
            popsize=200,
            tol=1e-12,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42,
            polish=False,  # We will do our own polish
            init='sobol'
        )
        
        # Check time
        if time.time() > deadline - 10.0:
            # If DE took too long (unlikely), just use its result
            candidate_bary = clamp_bary(res_de.x.reshape(N, 2))
            val = -hard_min_obj(candidate_bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = candidate_bary
            cart = bary_to_cart(best_bary)
            return cart
            
        candidate_bary = clamp_bary(res_de.x.reshape(N, 2))
        val = -hard_min_obj(candidate_bary.flatten())
        
        if val > best_val:
            best_val = val
            best_bary = candidate_bary
            
    except Exception as e:
        # Fallback to initial guess if DE fails
        best_bary = generate_hexagonal_init()
        best_val = -hard_min_obj(best_bary.flatten())
        
    # Phase 2: Local Refinement with BFGS
    # BFGS is faster and more precise than Nelder-Mead for smooth-ish functions.
    # The function is piecewise linear (min of linear functions), so gradients are discontinuous.
    # However, BFGS often works well on such "max-min" problems if started close enough.
    # Alternatively, we can use a smooth approximation (soft-min) for the gradient.
    
    # Let's try BFGS on the hard min first. If it struggles, we might need soft min.
    # Given the time budget, let's do a few restarts of BFGS from the DE result
    # and slightly perturbed versions.
    
    rng = np.random.default_rng(123)
    
    for i in range(10):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb the best solution found so far
        if i == 0:
            start_bary = best_bary
        else:
            start_bary = best_bary + rng.normal(0, 0.001, (N, 2))
            start_bary = clamp_bary(start_bary)
            
        try:
            res_bfgs = minimize(
                hard_min_obj,
                start_bary.flatten(),
                method='BFGS',
                options={'gtol': 1e-8, 'maxiter': 100}
            )
            
            candidate_bary = clamp_bary(res_bfgs.x.reshape(N, 2))
            polished_val = -hard_min_obj(candidate_bary.flatten())
            
            if polished_val > best_val:
                best_val = polished_val
                best_bary = candidate_bary
                
        except Exception:
            continue
            
    # Final check
    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
