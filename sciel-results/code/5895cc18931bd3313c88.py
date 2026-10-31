import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    bary: (N, 2) array.
    Returns array of shape (165,) containing the area ratios.
    """
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
    """
    Clamp barycentric coordinates to the valid triangle region.
    u >= 0, v >= 0, u+v <= 1
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0)
    v = np.clip(v, eps, 1.0)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    return np.column_stack([u, v])

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """
    Convert Cartesian (x, y) to barycentric coords (u, v).
    Point = u*A + v*B + w*C where A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H => w = y/H
    v = x - 0.5*w
    u = 1 - v - w
    """
    x = cart[:, 0]
    y = cart[:, 1]
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            areas = calc_areas_bary(bary)
            val = np.min(areas)
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, scale=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    perturbed = base_bary + rng.normal(0, scale, base_bary.shape)
    return clamp_bary(perturbed)

def _soft_min_obj(bary_flat, beta=500.0):
    """Soft-min objective for maximizing the minimum triangle area."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def _delaunay_smooth_obj(bary_flat, gamma=50.0):
    """
    Smooth objective using a power-sum approximation of the min area.
    This is differentiable and encourages all areas to be balanced.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Power sum: (sum(areas^(-gamma)))^(-1/gamma) approximates min
    # Use negative power to avoid division by zero
    eps = 1e-12
    a = np.maximum(areas, eps)
    neg_powers = np.power(a, -gamma)
    log_sum = np.log(np.sum(neg_powers))
    smooth_min = np.exp(-log_sum / gamma)
    return -smooth_min

def _local_polish_bary(bary_init, beta=500.0, maxiter=300):
    """
    Local polish using L-BFGS-B on the soft-min objective.
    """
    def obj(x):
        return _soft_min_obj(x, beta=beta)
    
    bounds = [(0, 1)] * (N * 2)
    
    try:
        res = minimize(
            obj,
            bary_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return bary_init

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        val = np.min(areas)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Phase 1: Differential Evolution global search
    # Use a smooth objective for DE (it needs gradients implicitly via finite diff)
    # The delaunay_smooth_obj is smoother than hard min
    
    def de_objective(x_flat):
        if time.time() > deadline - 10:
            return 1e10  # Penalty if we're running out of time
        bary = x_flat.reshape(N, 2)
        bary = clamp_bary(bary)
        # Use a moderately smooth objective
        return _delaunay_smooth_obj(bary.flatten(), gamma=30.0)
    
    # Bounds for DE: u_i in [0, 1], v_i in [0, 1]
    de_bounds = [(0, 1)] * (N * 2)
    
    # Generate multiple diverse starting points for DE
    hex_init = generate_hexagonal_init()
    update_best(hex_init)
    
    # Run DE with moderate population
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                de_objective,
                bounds=de_bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if de_result.success:
                de_bary = de_result.x.reshape(N, 2)
                update_best(de_bary)
    except Exception:
        pass
    
    # Phase 2: Local polish from multiple diverse starts
    # Use the best found and several random/perturbed starts
    polish_starts = []
    
    if best_bary is not None:
        polish_starts.append(best_bary.copy())
    
    # Add hex init
    polish_starts.append(hex_init.copy())
    
    # Add some random starts
    for i in range(8):
        if time.time() > deadline - 20:
            break
        rand = generate_random_init(seed=500 + i)
        polish_starts.append(rand)
    
    # Add perturbed versions of best
    if best_bary is not None:
        for i in range(4):
            if time.time() > deadline - 15:
                break
            pert = generate_perturbed_init(best_bary, scale=0.03, seed=600 + i)
            polish_starts.append(pert)
    
    # Polish each start
    for start_bary in polish_starts:
        if time.time() > deadline - 10:
            break
        
        # First pass: moderate beta
        polished = _local_polish_bary(start_bary, beta=300.0, maxiter=200)
        update_best(polished)
        
        # Second pass: higher beta for tighter min
        if time.time() > deadline - 8:
            break
        polished2 = _local_polish_bary(polished, beta=800.0, maxiter=300)
        update_best(polished2)
    
    # Phase 3: Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        final = _local_polish_bary(best_bary, beta=1500.0, maxiter=500)
        update_best(final)
        
        # One more with even higher beta
        if time.time() < deadline - 3:
            final2 = _local_polish_bary(final, beta=3000.0, maxiter=300)
            update_best(final2)
    
    # Convert best barycentric to Cartesian and return
    if best_bary is not None:
        return bary_to_cart(best_bary)
    else:
        # Fallback: return random valid points
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
        return bary_to_cart(bary)
