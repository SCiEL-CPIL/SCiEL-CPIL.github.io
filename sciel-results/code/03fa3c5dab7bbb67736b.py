"""Contact-graph sphere packing: sparse nearest-neighbor optimization with repulsion forces."""
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree

N = 100

def _compute_radius(centers):
    """Compute the maximum common radius for given centers using pdist for speed."""
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    d = pdist(centers)
    min_pair_dist = np.min(d)
    min_wall_dist = np.min(0.5 - np.abs(centers))
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _build_contact_graph(centers, k_neighbors=6):
    """Build contact graph using k-nearest neighbors via KDTree."""
    n = len(centers)
    tree = cKDTree(centers)
    dists, indices = tree.query(centers, k=k_neighbors + 1)
    
    # Build set of unique pairs
    pairs = set()
    for i in range(n):
        for j in indices[i, 1:]:  # skip self
            if i < j:
                pairs.add((i, j))
            else:
                pairs.add((j, i))
    
    pair_list = sorted(pairs)
    i_idx = np.array([p[0] for p in pair_list])
    j_idx = np.array([p[1] for p in pair_list])
    return i_idx, j_idx

def _contact_graph_objective_and_grad(x, k_neighbors=8, softness=50.0):
    """
    Objective: maximize min distance considering only contact graph edges + wall distances.
    Uses a soft-min over the contact graph edges for smoothness.
    """
    centers = x.reshape(N, 3)
    n = len(centers)
    
    # Rebuild contact graph
    i_idx, j_idx = _build_contact_graph(centers, k_neighbors)
    n_edges = len(i_idx)
    
    # Compute edge distances
    edge_vecs = centers[i_idx] - centers[j_idx]
    edge_dists = np.sqrt(np.sum(edge_vecs**2, axis=1))
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists, axis=1)
    
    # Soft-min over all constraints (edges + walls)
    all_dists = np.concatenate([edge_dists, min_wall])
    
    # Soft min: -1/softness * log(sum exp(-softness * d))
    shifted = -softness * all_dists
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    r_soft = -log_sum_exp / softness
    
    # Gradient
    weights = np.exp(shifted - max_shifted)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(centers)
    
    # Edge gradients
    edge_weights = weights[:n_edges]
    for idx in range(n_edges):
        w = edge_weights[idx]
        if w < 1e-12:
            continue
        i = i_idx[idx]
        j = j_idx[idx]
        vec = edge_vecs[idx]
        dist = edge_dists[idx]
        if dist < 1e-12:
            continue
        # d(r_soft)/d(c_i) += w * vec/dist
        # d(r_soft)/d(c_j) += w * (-vec)/dist
        grad[i] += w * vec / dist
        grad[j] += w * (-vec) / dist
    
    # Wall gradients
    wall_weights = weights[n_edges:]
    for i in range(n):
        w = wall_weights[i]
        if w < 1e-12:
            continue
        min_coord = np.argmin(wall_dists[i])
        sign = np.sign(centers[i, min_coord])
        if sign == 0:
            sign = 1.0
        grad[i, min_coord] += w * (-sign)
    
    return -r_soft, -grad.flatten()

def _optimize_contact_graph(centers, time_limit=10.0, k_neighbors=8, softness=50.0):
    """Optimize using contact graph with L-BFGS-B."""
    from scipy.optimize import minimize
    
    start = time.time()
    
    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return 1e10, np.zeros_like(x)
        return _contact_graph_objective_and_grad(x, k_neighbors, softness)
    
    x0 = centers.flatten()
    bounds = [(-0.4999999, 0.4999999)] * (N * 3)
    
    try:
        result = minimize(
            obj_and_grad, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30, 'maxfun': 1000}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return centers

def _repulsion_force_optimize(centers, time_limit=10.0, k_neighbors=10, n_iters=200):
    """
    Direct repulsion-based optimization: push spheres apart based on nearest neighbors.
    This is a gradient-free approach that directly models physical repulsion.
    """
    start = time.time()
    c = centers.copy()
    n = len(c)
    
    for it in range(n_iters):
        if time.time() - start > time_limit:
            break
        
        # Build KDTree for fast neighbor queries
        tree = cKDTree(c)
        dists, indices = tree.query(c, k=k_neighbors + 1)
        
        # Compute repulsion forces
        forces = np.zeros_like(c)
        for i in range(n):
            for k in range(1, k_neighbors + 1):
                j = indices[i, k]
                d = dists[i, k]
                if d < 1e-10:
                    continue
                # Repulsion force proportional to 1/d^2 (Coulomb-like)
                vec = c[i] - c[j]
                force_mag = 1.0 / (d * d)
                forces[i] += force_mag * vec / d
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        for i in range(n):
            for coord in range(3):
                wd = wall_dists[i, coord]
                if wd < 0.05:  # Only push back when close to wall
                    forces[i, coord] += (0.05 - wd) * 10.0 * np.sign(c[i, coord])
        
        # Adaptive step size
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 1e-10:
            break
        step = 0.01 / (1.0 + max_force)
        
        c += step * forces
        c = np.clip(c, -0.4999999, 0.4999999)
    
    return c

def _simulated_annealing(centers, time_limit=20.0, seed=42, k_neighbors=8):
    """
    Simulated annealing with adaptive step size for sphere packing.
    Uses contact graph to compute energy (negative min distance).
    """
    start = time.time()
    rng = np.random.RandomState(seed)
    c = centers.copy()
    n = len(c)
    
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling = 0.995
    n_steps = 5000
    
    for step in range(n_steps):
        if time.time() - start > time_limit:
            break
        
        # Perturb random subset of spheres
        n_perturb = max(5, n // 10)
        idx = rng.choice(n, size=n_perturb, replace=False)
        
        # Adaptive step size based on temperature
        step_size = T * 0.01
        perturbation = rng.normal(0, step_size, size=(n_perturb, 3))
        
        c_new = c.copy()
        c_new[idx] += perturbation
        c_new = np.clip(c_new, -0.4999999, 0.4999999)
        
        r_old = best_r  # Use best radius as reference for acceptance
        r_new = _compute_radius(c_new)
        
        # Accept if better or with probability exp((r_new - r_old)/T)
        if r_new > r_old:
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        else:
            delta = r_new - r_old
            if delta > -0.001:  # Only consider small degradations
                prob = np.exp(delta / max(T, 1e-10))
                if rng.random() < prob:
                    c = c_new
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_c

def _grid_init(n):
    """Grid initialization with slight jitter to break symmetry."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:n].copy()
    # Add small random jitter
    rng = np.random.RandomState(123)
    grid += rng.normal(0, 0.001, size=grid.shape)
    return np.clip(grid, -0.4999999, 0.4999999)

def _random_init(n, seed=None):
    """Random initialization inside cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _face_centered_init(n):
    """
    Initialize with spheres clustered near face centers and corners.
    This exploits the cube symmetry for better initial packing.
    """
    rng = np.random.RandomState(777)
    centers = np.zeros((n, 3))
    
    # Place some spheres near each face center
    face_centers = [
        [0.4, 0, 0], [-0.4, 0, 0],
        [0, 0.4, 0], [0, -0.4, 0],
        [0, 0, 0.4], [0, 0, -0.4]
    ]
    per_face = n // 6
    
    idx = 0
    for fc in face_centers:
        for i in range(per_face):
            centers[idx] = fc + rng.normal(0, 0.08, size=3)
            idx += 1
    
    # Fill remaining with random
    if idx < n:
        centers[idx:] = rng.uniform(-0.4, 0.4, size=(n - idx, 3))
    
    return np.clip(centers, -0.4999999, 0.4999999)

def construct():
    """Construct optimal packing of N spheres in unit cube using contact graph optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Multiple initialization strategies
    inits = [
        ("grid", _grid_init(N)),
        ("face_centered", _face_centered_init(N)),
        ("random", _random_init(N, seed=42)),
    ]
    
    best_centers = None
    best_radius = -1.0
    
    # Phase 1: Evaluate and select best initial configuration
    for name, init_c in inits:
        if time.time() > deadline - 100.0:
            break
        init_c = np.clip(init_c, -0.4999999, 0.4999999)
        r = _compute_radius(init_c)
        if r > best_radius:
            best_radius = r
            best_centers = init_c.copy()
    
    if best_centers is None:
        best_centers = _grid_init(N)
        best_radius = _compute_radius(best_centers)
    
    # Phase 2: Repulsion force optimization (fast, gradient-free)
    if time.time() < deadline - 70.0:
        time_remaining = deadline - time.time() - 65.0
        if time_remaining > 5.0:
            c_rep = _repulsion_force_optimize(best_centers, time_remaining, k_neighbors=10, n_iters=300)
            c_rep = np.clip(c_rep, -0.4999999, 0.4999999)
            r_new = _compute_radius(c_rep)
            if r_new > best_radius:
                best_radius = r_new
                best_centers = c_rep.copy()
    
    # Phase 3: Contact graph optimization with L-BFGS-B
    if time.time() < deadline - 40.0:
        time_remaining = deadline - time.time() - 35.0
        if time_remaining > 5.0:
            c_cg = _optimize_contact_graph(best_centers, time_remaining, k_neighbors=8, softness=50.0)
            c_cg = np.clip(c_cg, -0.4999999, 0.4999999)
            r_new = _compute_radius(c_cg)
            if r_new > best_radius:
                best_radius = r_new
                best_centers = c_cg.copy()
    
    # Phase 4: Simulated annealing for global exploration
    if time.time() < deadline - 20.0:
        time_remaining = deadline - time.time() - 15.0
        if time_remaining > 5.0:
            c_sa = _simulated_annealing(best_centers, time_remaining, seed=42, k_neighbors=8)
            c_sa = np.clip(c_sa, -0.4999999, 0.4999999)
            r_new = _compute_radius(c_sa)
            if r_new > best_radius:
                best_radius = r_new
                best_centers = c_sa.copy()
    
    # Phase 5: Final contact graph polish with tighter settings
    if time.time() < deadline - 5.0:
        time_remaining = deadline - time.time() - 3.0
        if time_remaining > 2.0:
            c_final = _optimize_contact_graph(best_centers, time_remaining, k_neighbors=12, softness=100.0)
            c_final = np.clip(c_final, -0.4999999, 0.4999999)
            r_new = _compute_radius(c_final)
            if r_new > best_radius:
                best_radius = r_new
                best_centers = c_final.copy()
    
    # Final safety clip
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    # Verify no overlaps and all inside cube
    d = pdist(best_centers)
    min_d = np.min(d)
    min_wall = np.min(0.5 - np.abs(best_centers))
    final_r = min(min_d / 2.0, min_wall)
    
    # If something went wrong, fall back to grid
    if final_r < 0.01:
        best_centers = _grid_init(N)
    
    return best_centers
