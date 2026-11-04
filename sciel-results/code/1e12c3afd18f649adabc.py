"""Combined program: iterative radius increase with energy minimization using jellyfish parameterization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5
N_ANCHORS = 8
LATTICE_SHAPE = (5, 4, 5)

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return float(r)

def _jellyfish_to_centers(params):
    """
    Convert jellyfish parameters to sphere centers.
    params: (N_ANCHORS*3 + N, 3) array
    First N_ANCHORS*3 entries are anchor positions (flattened).
    Next N*3 entries are local offsets from base lattice.
    
    Each center = weighted blend of:
      - its base lattice position (with local offset)
      - nearest 8 anchors (weighted by inverse distance)
    """
    anchors = params[:N_ANCHORS*3].reshape(N_ANCHORS, 3)
    offsets = params[N_ANCHORS*3:].reshape(N, 3)
    
    # Base lattice positions
    nx, ny, nz = LATTICE_SHAPE
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    base_lattice = np.array([(x, y, z) for z in zs for y in ys for x in xs])[:N]
    
    # Add local offsets to base lattice
    lattice_positions = base_lattice + offsets
    
    # Compute weights from each lattice point to each anchor
    # Use inverse distance squared with a small epsilon to avoid division by zero
    centers = np.zeros((N, 3))
    for i in range(N):
        lp = lattice_positions[i]
        dists_to_anchors = np.sqrt(np.sum((anchors - lp)**2, axis=1)) + 1e-6
        weights = 1.0 / (dists_to_anchors**2)
        weights /= np.sum(weights)
        
        # Blend: 70% lattice, 30% anchor-influenced
        anchor_blend = np.sum(weights[:, np.newaxis] * anchors, axis=0)
        centers[i] = 0.7 * lp + 0.3 * anchor_blend
    
    return centers

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Uses a smooth soft-penalty: quadratic when violated, zero otherwise.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms - smooth penalty for d_ij < 2R
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists + 1e-15)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(sq_dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Only process pairs that are close enough to potentially violate
    close_mask = d_ij < 2.5 * R
    if np.any(close_mask):
        i_close = i_idx[close_mask]
        j_close = j_idx[close_mask]
        d_close = d_ij[close_mask]
        
        # Smooth penalty: (max(0, 2R - d))^2
        violation = np.maximum(0.0, 2.0 * R - d_close)
        energy += np.sum(violation**2)
        
        # Gradient computation
        active = violation > 0
        if np.any(active):
            i_a = i_close[active]
            j_a = j_close[active]
            d_a = d_close[active]
            v_a = violation[active]
            
            factors = 2.0 * v_a / d_a
            vecs = centers[i_a] - centers[j_a]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_a, -forces)
            np.add.at(grad, j_a, forces)

    # 2. Wall terms - smooth penalty
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    wall_violation = np.maximum(0.0, R - wall_dists)
    energy += np.sum(wall_violation**2)
    
    active_wall = wall_violation > 0
    if np.any(active_wall):
        for idx in np.where(active_wall)[0]:
            v = wall_violation[idx]
            max_k = np.argmax(abs_centers[idx])
            sign_val = np.sign(centers[idx, max_k])
            if sign_val == 0:
                sign_val = 1.0
            grad[idx, max_k] += 2.0 * v * sign_val
            
    return energy, grad.flatten()

def _jellyfish_energy_and_gradient(params, R):
    """
    Energy and gradient in jellyfish parameter space.
    Uses finite differences for the Jacobian of the jellyfish mapping.
    """
    centers = _jellyfish_to_centers(params)
    energy, center_grad = _energy_and_gradient(centers, R)
    
    # Compute Jacobian of centers w.r.t. params via finite differences
    n_params = len(params)
    jac = np.zeros((N * 3, n_params))
    eps = 1e-6
    
    for k in range(n_params):
        params_plus = params.copy()
        params_plus[k] += eps
        centers_plus = _jellyfish_to_centers(params_plus).flatten()
        jac[:, k] = (centers_plus - centers.flatten()) / eps
    
    # Chain rule: dE/dparams = J^T * dE/dcenters
    param_grad = jac.T @ center_grad
    
    return energy, param_grad

def _minimize_energy_jellyfish(init_params, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy in jellyfish parameter space for a given target radius.
    """
    def objective_and_grad(x):
        val, g = _jellyfish_energy_and_gradient(x, target_R)
        return val, g

    n_params = init_params.shape[0]
    # Bounds: anchors in [-0.5, 0.5], offsets in [-0.15, 0.15]
    bounds = [(-0.5, 0.5)] * (N_ANCHORS * 3) + [(-0.15, 0.15)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_params,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_params = res.x
        new_centers = _jellyfish_to_centers(new_params)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_params, new_centers, final_energy, res.success
    except Exception:
        init_centers = _jellyfish_to_centers(init_params)
        return init_params, init_centers, np.inf, False

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy for a given target radius (direct center optimization).
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def _initialize_jellyfish_params(rng):
    """
    Initialize jellyfish parameters: anchors at cube corners, small random offsets.
    """
    # Anchors near cube corners
    corners = np.array([
        [-0.4, -0.4, -0.4], [0.4, -0.4, -0.4],
        [-0.4, 0.4, -0.4], [0.4, 0.4, -0.4],
        [-0.4, -0.4, 0.4], [0.4, -0.4, 0.4],
        [-0.4, 0.4, 0.4], [0.4, 0.4, 0.4]
    ])
    anchors = corners + rng.normal(0, 0.02, size=(N_ANCHORS, 3))
    
    # Small random offsets for each sphere
    offsets = rng.normal(0, 0.01, size=(N, 3))
    
    params = np.concatenate([anchors.flatten(), offsets.flatten()])
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # 1. Initialize with jellyfish parameterization
    init_params = _initialize_jellyfish_params(rng)
    init_centers = _jellyfish_to_centers(init_params)
    
    # Initial optimization with conservative radius
    initial_r = 0.1
    init_params, init_centers, energy, _ = _minimize_energy_jellyfish(init_params, initial_r, max_iter=200)
    best_r = _compute_radius(init_centers)
    best_params = init_params.copy()
    best_pos = init_centers.copy()
    
    if best_r <= 0:
        # Fallback to direct optimization
        best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = _compute_radius(best_pos)
        best_params = None
    
    # 2. Iterative Radius Increase with jellyfish optimization
    current_r = best_r
    step_size = 0.001
    
    consecutive_failures = 0
    max_failures = 4
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 15.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.6
            consecutive_failures = 0
            if step_size < 1e-7:
                break
        
        success = False
        
        # Try jellyfish optimization with perturbed parameters
        n_attempts = 3
        for attempt in range(n_attempts):
            if time.time() > deadline - 15.0:
                break
            
            if attempt == 0:
                start_params = best_params.copy() if best_params is not None else _initialize_jellyfish_params(rng_search)
            else:
                start_params = best_params + rng_search.normal(0, 0.01, size=best_params.shape)
            
            start_params = np.clip(start_params, -0.5, 0.5)
            
            max_iter = 400 if current_r < 0.11 else 600
            
            new_params, new_pos, energy, opt_success = _minimize_energy_jellyfish(
                start_params, target_r, max_iter=max_iter
            )
            
            if energy < 1e-10:
                actual_r = _compute_radius(new_pos)
                if actual_r >= target_r - 1e-6:
                    best_params = new_params
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    success = True
                    consecutive_failures = 0
                    break
            else:
                # Check if we improved even without perfect energy
                actual_r = _compute_radius(new_pos)
                if actual_r > current_r + 1e-8:
                    best_params = new_params
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
        
        if not success:
            consecutive_failures += 1
            
            # Try direct center optimization as fallback
            if time.time() < deadline - 20.0 and consecutive_failures % 2 == 0:
                polished, _, _ = _minimize_energy(best_pos, current_r, max_iter=300)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_pos = polished
                    best_r = polished_r
                    current_r = best_r
                    # Re-initialize jellyfish params from this position
                    # (approximate: set offsets to match, anchors at corners)
                    base_lattice = np.array([(x, y, z) for z in np.linspace(-0.4, 0.4, 5) 
                                           for y in np.linspace(-0.4, 0.4, 4) 
                                           for x in np.linspace(-0.4, 0.4, 5)])[:N]
                    offsets = best_pos - base_lattice
                    corners = np.array([
                        [-0.4, -0.4, -0.4], [0.4, -0.4, -0.4],
                        [-0.4, 0.4, -0.4], [0.4, 0.4, -0.4],
                        [-0.4, -0.4, 0.4], [0.4, -0.4, 0.4],
                        [-0.4, 0.4, 0.4], [0.4, 0.4, 0.4]
                    ])
                    best_params = np.concatenate([corners.flatten(), offsets.flatten()])

    # 3. Final High-Precision Polish
    if time.time() < deadline - 2.0:
        # Try to push slightly beyond current best with very tight tolerances
        for _ in range(5):
            if time.time() > deadline - 1.0:
                break
            target_r = best_r + 1e-6
            if best_params is not None:
                polished_params, polished_pos, energy, _ = _minimize_energy_jellyfish(
                    best_params, target_r, max_iter=2000, ftol=1e-16, gtol=1e-15
                )
                polished_r = _compute_radius(polished_pos)
                if polished_r > best_r:
                    best_params = polished_params
                    best_pos = polished_pos
                    best_r = polished_r
            else:
                polished, energy, _ = _minimize_energy(best_pos, target_r, max_iter=2000, ftol=1e-16, gtol=1e-15)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_pos = polished
                    best_r = polished_r
    
    # Final validation and clipping
    best_pos = np.clip(best_pos, -0.499999, 0.499999)
    
    return best_pos
