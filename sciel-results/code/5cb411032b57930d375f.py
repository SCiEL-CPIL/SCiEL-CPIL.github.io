"""Rewritten program: optimize 50 circle centres in the unit disk via contact-graph force-directed layout."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin for 120s limit

def _score(centers):
    """Compute the packing score: largest common radius."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_bound)

def _hex_lattice_init(n, seed=42):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    r_est = 0.125
    s = 2 * r_est
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
    points = np.array(points)
    if len(points) < n:
        return _random_init(n, seed)
    # Shuffle and take n
    idx = rng.permutation(len(points))[:n]
    return points[idx]

def _concentric_ring_init(n, seed=42):
    """Ring-based initialization matching known optimal structure 1+6+12+18+13."""
    rng = np.random.default_rng(seed)
    centers = []
    centers.append([0.0, 0.0])
    rings = [
        (6, 0.25),
        (12, 0.50),
        (18, 0.75),
        (13, 0.92),
    ]
    count = 1
    for cnt, radius in rings:
        angles = np.linspace(0, 2 * np.pi, cnt, endpoint=False)
        # Add small asymmetric perturbation
        angles += rng.uniform(-0.02, 0.02, size=cnt)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
        if count >= n:
            break
    return np.array(centers[:n])

def _random_init(n, seed=42):
    """Random initialization with rejection sampling."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x*x + y*y <= 0.9:
                break
        centers.append([x, y])
    return np.array(centers)

def _project_to_feasible(centers):
    """Project centers to be inside the unit disk."""
    norms = np.linalg.norm(centers, axis=1)
    mask = norms > 0.999
    if np.any(mask):
        centers = centers.copy()
        centers[mask] *= 0.999 / norms[mask]
    return centers

def _contact_graph_optimize(centers, start_time, max_iters=3000):
    """
    Force-directed layout with 1/d^4 repulsive potential and hard constraint projection.
    Nodes represent circles, edges represent near-contact constraints.
    """
    current = centers.copy()
    best_score = _score(current)
    best_centers = current.copy()
    n = len(current)
    
    # Parameters
    repulsive_power = 4
    attractive_strength = 0.001
    boundary_strength = 0.01
    lr = 0.002
    lr_decay = 0.999
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Compute pairwise distances
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(current, axis=1)
        
        # Force-directed update
        forces = np.zeros_like(current)
        
        # Repulsive forces: 1/d^4 for all pairs (approximate with top-k for speed)
        # Use all pairs since n=50 is small
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                if d < 1e-10:
                    # Random push for coincident points
                    forces[i] += rng_uniform(2) * 0.01
                    forces[j] -= rng_uniform(2) * 0.01
                    continue
                # Repulsive force magnitude
                f_mag = 1.0 / (d ** repulsive_power)
                vec = (current[i] - current[j]) / d
                forces[i] += f_mag * vec
                forces[j] -= f_mag * vec
        
        # Boundary repulsion: push points toward center if close to boundary
        for i in range(n):
            if norms[i] > 0.8:
                f_mag = boundary_strength * (norms[i] - 0.8) ** 2
                if norms[i] > 1e-10:
                    forces[i] -= f_mag * current[i] / norms[i]
        
        # Attractive force toward centroid (keeps cluster compact)
        centroid = np.mean(current, axis=0)
        for i in range(n):
            forces[i] += attractive_strength * (centroid - current[i])
        
        # Update positions
        current += lr * forces
        
        # Hard constraint projection: ensure inside unit disk
        current = _project_to_feasible(current)
        
        # Check score
        new_score = _score(current)
        if new_score > best_score:
            best_score = new_score
            best_centers = current.copy()
        
        # Decay learning rate
        lr *= lr_decay
        if lr < 1e-7:
            break
    
    return best_centers, best_score

def rng_uniform(size):
    """Simple random uniform for reproducibility within a call."""
    return np.random.uniform(-1, 1, size=size)

def _soft_min_objective(params, n, beta):
    """
    Objective: minimize -soft_min(pair_dists/2, boundary_dists)
    Using log-sum-exp for smoothness.
    """
    centers = params.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    # Soft min via log-sum-exp
    m = np.min(vals)
    exp_terms = np.exp(-beta * (vals - m))
    soft_min = m - (1.0 / beta) * np.log(np.sum(exp_terms))
    
    return -soft_min

def _optimize_from_start(centers, start_time):
    """
    Multi-stage optimization:
    1. L-BFGS-B with soft min
    2. Contact-graph force-directed polish
    """
    current = centers.copy()
    
    # Stage 1: L-BFGS-B with soft min
    betas = [50.0, 200.0, 500.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 25:
            break
        try:
            res = minimize(
                _soft_min_objective,
                current.flatten(),
                args=(N, beta),
                method='L-BFGS-B',
                options={'maxiter': 80, 'gtol': 1e-5}
            )
            if np.isfinite(res.fun):
                candidate = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate)):
                    current = candidate
        except Exception:
            pass
    
    # Stage 2: Contact-graph force-directed polish
    if time.time() - start_time < DEADLINE - 5:
        current, _ = _contact_graph_optimize(current, start_time, max_iters=2000)
    
    return current

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Diverse initializations
    initializations = [
        _concentric_ring_init(N, seed=42),
        _concentric_ring_init(N, seed=7),
        _hex_lattice_init(N, seed=42),
        _hex_lattice_init(N, seed=13),
        _random_init(N, seed=1),
        _random_init(N, seed=99),
    ]
    
    # Optimize each initialization
    for init in initializations:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Pre-process: ensure inside
        init = _project_to_feasible(init)
        
        optimized = _optimize_from_start(init, start_time)
        
        if np.all(np.isfinite(optimized)):
            r = _score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
    
    if best_centers is None:
        return initializations[0]
    
    # Final polish on the absolute best
    if time.time() - start_time < DEADLINE - 3:
        best_centers, best_score = _contact_graph_optimize(best_centers, start_time, max_iters=5000)
    
    # Final feasibility check
    best_centers = _project_to_feasible(best_centers)
    
    return best_centers
