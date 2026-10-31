import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
TOTAL_AREA = 0.5 * 1.0 * H # sqrt(3)/4

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*V0 + v*V1 + w*V2
    V0=(0,0), V1=(1,0), V2=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_min_area_bary(bary):
    """
    Calculate the minimum area of all triangles formed by N points.
    bary: (N, 2) array of (u, v).
    Returns the minimum area (absolute value).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for all triangles
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    # Determinant calculation
    # det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    # The area of the triangle in Cartesian is |det| * TOTAL_AREA
    # Since we want to maximize the min area, we maximize min(|det|)
    # The evaluator likely calculates the actual area.
    # Let's return the actual area to be safe, or the determinant?
    # The task says "maximize the smallest area".
    # Area = |det| * TOTAL_AREA.
    return np.min(np.abs(det)) * TOTAL_AREA

def objective(bary_flat):
    """
    Objective function for optimization.
    Minimizes the negative of the minimum area.
    """
    bary = bary_flat.reshape(N, 2)
    # Ensure barycentric coords are valid (u>=0, v>=0, u+v<=1)
    # DE handles bounds, but let's clamp for safety in local steps
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1 - u)
    bary = np.column_stack([u, v])
    
    min_area = calc_min_area_bary(bary)
    return -min_area

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to valid region."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = calc_min_area_bary(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Bounds for barycentric coords (u, v)
    # u in [0, 1], v in [0, 1-u]. 
    # DE requires fixed bounds. We use [0,1] for both and clamp in objective.
    bounds = [(0, 1)] * (N * 2)

    # 1. Generate diverse initial points
    # Strategy: Hexagonal lattice + Random + Vertices/Midpoints
    
    # Hexagonal lattice init
    def hex_init():
        points = []
        # Try a few grid densities
        for s in np.linspace(0.1, 0.4, 10):
            for i in range(int(1/s)):
                for j in range(int(1/s)):
                    u = i * s
                    v = j * s
                    if u + v <= 1 and len(points) < N:
                        points.append([u, v])
            if len(points) == N:
                return np.array(points)
        # Fallback random
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        return np.column_stack([u, v])
    
    # Random init
    def rand_init(seed):
        rng = np.random.default_rng(seed)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        return np.column_stack([u, v])
    
    # "Pushed" init: Start with points at vertices and midpoints, then perturb
    def pushed_init():
        # 3 vertices, 3 midpoints, 3 centroid-ish, 2 random
        pts = [
            [0.01, 0.01], [0.98, 0.01], [0.01, 0.98], # Corners (approx in bary)
            [0.5, 0.01], [0.01, 0.5], [0.49, 0.49], # Midpoints/Center
            [0.33, 0.33], [0.33, 0.66], [0.66, 0.33], # Trisection points
            [0.2, 0.2], [0.8, 0.1]
        ]
        return np.array(pts[:N])

    init_candidates = [
        hex_init(),
        rand_init(1),
        rand_init(2),
        pushed_init()
    ]
    
    for ic in init_candidates:
        update_best(ic)

    # 2. Differential Evolution
    # Run DE for a limited number of iterations to explore globally
    try:
        # Use a larger population for better exploration
        de_result = differential_evolution(
            objective,
            bounds,
            popsize=40,
            maxiter=100,
            tol=1e-6,
            mut=(0.5, 1.5),
            recomb=0.8,
            seed=42,
            polish=False, # We will polish manually
            init='latinhypercube',
            update_interval=1
        )
        if de_result.x is not None:
            update_best(de_result.x.reshape(N, 2))
    except Exception as e:
        print(f"DE failed: {e}")

    # Check time
    if time.time() > deadline - 20:
        if best_bary is None:
            best_bary = rand_init(42)
        return bary_to_cart(clamp_bary(best_bary))

    # 3. Local Refinement (Nelder-Mead)
    # Nelder-Mead is robust for non-smooth functions like min-area
    # We will run it on the best DE result and the best initial candidates
    
    local_starts = []
    if best_bary is not None:
        local_starts.append(best_bary.copy())
    
    # Add a few perturbed versions of the best
    for i in range(5):
        rng = np.random.default_rng(100 + i)
        perturbed = best_bary.copy()
        perturbed += rng.normal(0, 0.05, perturbed.shape)
        local_starts.append(clamp_bary(perturbed))
        
    # Add the initial candidates that were good
    for ic in init_candidates:
        local_starts.append(ic.copy())

    # Run Nelder-Mead on each start
    for start in local_starts:
        if time.time() > deadline - 5:
            break
            
        try:
            nm_result = minimize(
                objective,
                start.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if nm_result.x is not None:
                update_best(nm_result.x.reshape(N, 2))
        except Exception:
            pass

    # 4. Final Polish with L-BFGS-B using Soft Min
    # If we have time, try to smooth the landscape for gradient-based polish
    if best_bary is not None and time.time() < deadline - 10:
        def soft_min_obj(bary_flat, beta=50.0):
            bary = bary_flat.reshape(N, 2)
            bary = clamp_bary(bary)
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
            areas = np.abs(det) * TOTAL_AREA
            
            # Soft min
            scaled = -beta * areas
            max_scaled = np.max(scaled)
            if max_scaled > 700:
                return -np.min(areas)
            log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            soft_min = -log_sum / beta
            return -soft_min

        try:
            lbfgs_result = minimize(
                soft_min_obj,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if lbfgs_result.x is not None:
                update_best(lbfgs_result.x.reshape(N, 2))
        except Exception:
            pass

    if best_bary is None:
        best_bary = rand_init(42)
        
    final_bary = clamp_bary(best_bary)
    return bary_to_cart(final_bary)
