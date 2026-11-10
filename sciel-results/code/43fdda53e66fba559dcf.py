"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree


def _compute_score(points):
    """Compute the score: min(min pairwise dist / 2, min (1 - |c_i|))."""
    n = len(points)
    if n < 2:
        return 0.0
    tree = cKDTree(points)
    # Find nearest neighbor distances efficiently
    dists, _ = tree.query(points, k=2)
    min_pair_dist = dists[:, 1].min()
    # Boundary distance
    radii = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - radii.max()
    return min(min_pair_dist / 2.0, min_boundary)


def _forces_and_score(points, soft_beta=50.0):
    """
    Compute gradient-like forces and a smooth surrogate score.
    Returns: (forces array (N,2), smooth_score)
    Forces point in direction that increases the minimum distance.
    """
    n = len(points)
    tree = cKDTree(points)
    # Query k nearest neighbors (k=6 gives good local picture)
    k = min(6, n)
    dists, indices = tree.query(points, k=k)
    # dists[:,0] is 0 (self), dists[:,1:] are neighbor distances
    neighbor_dists = dists[:, 1:]  # shape (n, k-1)
    neighbor_indices = indices[:, 1:]  # shape (n, k-1)
    
    # For each point, compute force from neighbors:
    # Force pushes point away from neighbors, weighted by proximity
    forces = np.zeros((n, 2))
    for i in range(n):
        for j_idx in range(k - 1):
            j = neighbor_indices[i, j_idx]
            d = neighbor_dists[i, j_idx]
            if d < 1e-12:
                continue
            # Inverse-distance weighting: closer neighbors exert more force
            weight = 1.0 / (d + 1e-6)
            direction = points[i] - points[j]
            direction = direction / d
            forces[i] += weight * direction
    
    # Boundary force: push points inward if near boundary
    radii = np.linalg.norm(points, axis=1)
    boundary_margin = 1.0 - radii
    for i in range(n):
        if boundary_margin[i] < 0.1:
            # Strong inward force when close to boundary
            pull = max(0, 0.1 - boundary_margin[i]) / 0.1
            if radii[i] > 1e-12:
                forces[i] -= pull * 2.0 * points[i] / radii[i]
    
    # Smooth surrogate score using soft-min
    # min_pair_dist / 2 and min_boundary
    min_pair = dists[:, 1].min() / 2.0
    min_bnd = 1.0 - radii.max()
    # Soft min: -log(exp(-a/beta) + exp(-b/beta)) * beta approximates min(a,b)
    a, b = min_pair, min_bnd
    m = min(a, b)
    if m < -50:
        soft_min = m
    else:
        soft_min = m - np.log(np.exp(-a / soft_beta) + np.exp(-b / soft_beta)) * soft_beta
    
    return forces, soft_min


def _local_optimize(points, deadline, max_steps=20000):
    """
    Gradient-based local optimization with adaptive step size.
    Uses forces from nearest neighbors and boundary.
    """
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)
    current = points.copy()
    
    step_size = 0.005
    lr_decay = 0.999
    momentum = 0.5
    vel = np.zeros((n, 2))
    
    rng = np.random.default_rng(123)
    
    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
        
        # Compute forces
        forces, _ = _forces_and_score(current)
        
        # Update velocity with momentum
        vel = momentum * vel + (1 - momentum) * forces
        vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = vel / np.maximum(vel_norm, 1e-12)  # Normalize to unit vectors
        
        # Take step
        current = current + step_size * vel
        
        # Clamp to unit circle
        radii = np.linalg.norm(current, axis=1)
        mask = radii > 0.999
        if np.any(mask):
            current[mask] = current[mask] / radii[mask, None] * 0.999
        
        # Evaluate score
        score = _compute_score(current)
        
        if score > best_score:
            best_score = score
            best_points = current.copy()
            step_size *= 1.01  # Slightly increase step
            step_size = min(step_size, 0.02)
        else:
            # Revert and decrease step
            current = best_points.copy()
            vel *= 0.5
            step_size *= lr_decay
        
        # Occasionally add small random perturbation to escape local minima
        if step % 200 == 0:
            n_perturb = max(1, n // 20)
            idxs = rng.choice(n, size=n_perturb, replace=False)
            perturb = rng.normal(scale=step_size * 0.5, size=(n_perturb, 2))
            current[idxs] += perturb
            # Clamp
            radii = np.linalg.norm(current[idxs], axis=1)
            mask = radii > 0.999
            if np.any(mask):
                current[idxs][mask] = current[idxs][mask] / radii[mask, None] * 0.999
            score = _compute_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
        
        # Adaptive step size based on progress
        if step % 100 == 0:
            # If we've made no progress recently, shrink step more aggressively
            pass
    
    return best_points, best_score


def _shell_init():
    """
    Initialize with concentric shells matching known optimal packing structure.
    For 300 circles: center + shells with counts summing to 299.
    Known good structure: 1 + 6 + 12 + 18 + 24 + 30 + 36 + 42 + 48 + ... 
    but adjusted to sum to 300.
    """
    points = [np.array([0.0, 0.0])]
    remaining = N - 1
    
    # Shell structure: try to use hexagonal close packing shells
    # Shell k has approximately 6k circles
    shell_counts = []
    for k in range(1, 20):
        count = 6 * k
        if remaining <= 0:
            break
        if remaining < count:
            shell_counts.append(remaining)
            remaining = 0
        else:
            shell_counts.append(count)
            remaining -= count
    
    if remaining > 0:
        # Add remaining to last shell
        shell_counts[-1] += remaining
    
    # Distribute circles on each shell
    for shell_idx, count in enumerate(shell_counts):
        if count <= 0:
            continue
        # Radius of this shell (approximate)
        # For hexagonal packing, shell k radius ≈ k * spacing
        # We'll optimize radii later
        r_shell = (shell_idx + 1) * 0.055  # Initial guess
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        # Add small angular offset for alternating shells
        if shell_idx % 2 == 1:
            angles += np.pi / count
        for angle in angles:
            x = r_shell * np.cos(angle)
            y = r_shell * np.sin(angle)
            points.append(np.array([x, y]))
    
    points = np.array(points[:N])
    return points


def _hex_init():
    """Initialize with hexagonal lattice clipped to unit disk."""
    spacing = 0.106  # Approximate optimal spacing for 300 circles
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append(np.array([x, y]))
    
    points = np.array(points)
    # Keep closest to center
    radii = np.linalg.norm(points, axis=1)
    if len(points) > N:
        idx = np.argsort(radii, kind="stable")[:N]
        points = points[idx]
    elif len(points) < N:
        # If not enough, add more by shrinking spacing
        for _ in range(10):
            spacing *= 0.97
            points = []
            k = int(1.0 / spacing) + 2
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x * x + y * y <= 1.0:
                        points.append(np.array([x, y]))
            points = np.array(points)
            radii = np.linalg.norm(points, axis=1)
            if len(points) >= N:
                idx = np.argsort(radii, kind="stable")[:N]
                points = points[idx]
                break
    
    return points


def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Try multiple initializations and keep the best
    initializations = [
        _shell_init(),
        _hex_init(),
    ]
    
    best_points = None
    best_score = -1
    
    for init_points in initializations:
        if time.time() > deadline - 60.0:
            break
        
        # Optimize
        opt_points, opt_score = _local_optimize(init_points, deadline - 10.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points
    
    # Final polish: run a few more steps with smaller step size
    if best_points is not None and time.time() < deadline - 5.0:
        # Tighten with smaller steps
        current = best_points.copy()
        for _ in range(5000):
            if time.time() > deadline - 2.0:
                break
            forces, _ = _forces_and_score(current, soft_beta=100.0)
            step_size = 0.001
            current = current + step_size * forces
            radii = np.linalg.norm(current, axis=1)
            mask = radii > 0.9995
            if np.any(mask):
                current[mask] = current[mask] / radii[mask, None] * 0.9995
            score = _compute_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
    
    return best_points
# EVOLVE-BLOCK-END
