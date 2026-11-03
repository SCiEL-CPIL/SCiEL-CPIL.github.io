import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all unique triples (i, j, k) with i < j < k
# There are C(11, 3) = 165 triples
_triples = []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_TRIPLES = np.array(_triples, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLES)

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates.
    Returns the minimum normalized triangle area.
    """
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    
    # Cross product r2 x r3
    cross_r2_r3 = np.cross(r2, r3)
    
    # Dot product r1 . (r2 x r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    
    areas = 0.5 * np.abs(dets)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric (u,v,w) to Cartesian (x,y) for equilateral triangle."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary):
    """
    Project barycentric coordinates to be valid (non-negative, sum to 1).
    Simple clip and normalize.
    """
    b = bary.copy()
    b = np.clip(b, 1e-8, 1.0)
    b_sum = np.sum(b, axis=1, keepdims=True)
    b = b / b_sum
    return b

def _objective_flat(x_flat):
    """
    Objective function for optimizers: negative of the min area.
    x_flat: array of shape (N*3,)
    """
    bary = x_flat.reshape(N, 3)
    bary = _project_bary(bary)
    return -_min_area_bary(bary)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def _is_time_up():
        return time.time() > deadline - 2.0

    best_area = 0.0
    best_bary = None

    rng = np.random.default_rng(42)

    # Phase 1: Global Search with Differential Evolution
    # DE is robust to non-convexity and doesn't require gradients.
    # We use a moderate popsize to balance exploration and computation time.
    
    # Bounds: each coordinate in [0, 1]
    bounds = [(0.0, 1.0)] * (N * 3)

    try:
        # DE options:
        # maxiter: limit to ensure we finish in time.
        # popsize: 15-20 is usually sufficient for 33 dimensions.
        # tol: stop when improvement is small.
        # seed: fixed for reproducibility in this block, but rng used for init.
        
        de_res = differential_evolution(
            _objective_flat,
            bounds,
            maxiter=50,  # Limit iterations to save time for SA
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False, # We will do our own refinement
            init='latinhypercube',
            updating='deferred'
        )
        
        if _is_time_up():
            pass # Just skip refinement if time is up

        current_bary = _project_bary(de_res.x.reshape(N, 3))
        current_area = _min_area_bary(current_bary)
        
        if current_area > best_area:
            best_area = current_area
            best_bary = current_bary.copy()
            
    except Exception as e:
        # Fallback if DE fails
        best_bary = rng.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(best_bary)
        best_area = _min_area_bary(best_bary)

    # Phase 2: Simulated Annealing Refinement
    # SA can navigate the rugged landscape of the exact min-area objective.
    # It accepts worse solutions with probability exp(-delta/E), helping escape local minima.
    
    if best_bary is not None and not _is_time_up():
        current_bary = best_bary.copy()
        current_area = best_area
        T = 1e-3  # Initial temperature
        T_min = 1e-8
        cooling_rate = 0.995
        max_steps = 20000
        step_size = 0.05
        
        # We work in barycentric coordinates directly for simplicity in SA
        # Perturbation: add Gaussian noise to barycentric coords, then project
        
        for i in range(max_steps):
            if _is_time_up():
                break
            if T < T_min:
                break
            
            # Generate neighbor
            noise = rng.normal(0, step_size, size=(N, 3))
            neighbor_bary = current_bary + noise
            neighbor_bary = _project_bary(neighbor_bary)
            
            neighbor_area = _min_area_bary(neighbor_bary)
            
            # Acceptance criterion
            if neighbor_area > current_area:
                current_bary = neighbor_bary
                current_area = neighbor_area
                if current_area > best_area:
                    best_area = current_area
                    best_bary = current_bary.copy()
            else:
                delta = current_area - neighbor_area
                # Probability of accepting worse solution
                # If T is very small, this becomes 0
                if T > 0:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_bary = neighbor_bary
                        current_area = neighbor_area
                        if current_area > best_area:
                            best_area = current_area
                            best_bary = current_bary.copy()
            
            T *= cooling_rate

    # Phase 3: Final Polish with Nelder-Mead
    # NM is good for final convergence when close to optimum.
    if best_bary is not None and not _is_time_up():
        x0 = best_bary.flatten()
        try:
            res_nm = minimize(
                _objective_flat,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if not _is_time_up():
                refined_bary = _project_bary(res_nm.x.reshape(N, 3))
                refined_area = _min_area_bary(refined_bary)
                if refined_area > best_area:
                    best_area = refined_area
                    best_bary = refined_bary.copy()
        except Exception:
            pass

    if best_bary is None:
        # Fallback
        best_bary = rng.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(best_bary)

    return _bary_to_cart(best_bary)

def _fallback():
    """Fallback if no valid result found."""
    rng = np.random.default_rng(0)
    bary = rng.dirichlet([1, 1, 1], size=N)
    bary = _project_bary(bary)
    return _bary_to_cart(bary)
