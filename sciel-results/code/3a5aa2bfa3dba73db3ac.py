"""Pack 50 equal circles in the unit circle using multi-start optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

def _compute_score(centers):
    """Compute the packing score (largest common radius)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    dists = np.sqrt(np.sum(centers**2, axis=1))
    min_boundary = 1.0 - np.max(dists)
    if min_boundary < 0:
        return 0.0
    pairwise = cdist(centers, centers)
    np.fill_diagonal(pairwise, np.inf)
    min_pair = np.min(pairwise)
    score = min(min_boundary, min_pair / 2.0)
    return max(score, 0.0)

def _objective(x, beta):
    """Objective function: negative soft-min of constraints."""
    n = N
    centers = x.reshape(n, 2)
    dists = np.sqrt(np.sum(centers**2, axis=1))
    boundary = 1.0 - dists
    pairwise = cdist(centers, centers)
    np.fill_diagonal(pairwise, np.inf)
    pair_half = pairwise / 2.0
    iu = np.triu_indices(n, k=1)
    pair_constraints = pair_half[iu]
    all_constraints = np.concatenate([boundary, pair_constraints])
    m = np.min(all_constraints)
    exp_vals = np.exp(beta * (all_constraints - m))
    sum_exp = np.sum(exp_vals)
    soft_min_val = m - np.log(sum_exp) / beta
    return -soft_min_val

def _gradient(x, beta):
    """Gradient of the objective function."""
    n = N
    centers = x.reshape(n, 2)
    dists = np.sqrt(np.sum(centers**2, axis=1))
    boundary = 1.0 - dists
    pairwise = cdist(centers, centers)
    np.fill_diagonal(pairwise, np.inf)
    pair_half = pairwise / 2.0
    iu = np.triu_indices(n, k=1)
    pair_constraints = pair_half[iu]
    all_constraints = np.concatenate([boundary, pair_constraints])
    m = np.min(all_constraints)
    exp_vals = np.exp(beta * (all_constraints - m))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    grad = np.zeros_like(x)
    for i in range(n):
        if dists[i] > 1e-12:
            grad[2*i] = -weights[i] * (-centers[i, 0] / dists[i])
            grad[2*i+1] = -weights[i] * (-centers[i, 1] / dists[i])
    for k in range(len(pair_constraints)):
        i, j = iu[0][k], iu[1][k]
        d = pairwise[i, j]
        if d > 1e-12:
            w = weights[n + k]
            diff = centers[i] - centers[j]
            grad[2*i] += -w * (diff[0] / d) * 0.5
            grad[2*i+1] += -w * (diff[1] / d) * 0.5
            grad[2*j] += -w * (-diff[0] / d) * 0.5
            grad[2*j+1] += -w * (-diff[1] / d) * 0.5
    return grad

def _optimize_from_start(start_centers, beta_start=50.0, beta_end=1000.0, time_limit=20.0):
    """Optimize using L-BFGS-B with increasing beta."""
    start = time.time()
    x0 = start_centers.flatten().astype(np.float64)
    best_x = x0.copy()
    best_score = _compute_score(start_centers)
    n_betas = 8
    betas = np.geomspace(beta_start, beta_end, n_betas)
    for beta in betas:
        if time.time() - start > time_limit - 2.0:
            break
        try:
            result = minimize(_objective, best_x, jac=lambda x: _gradient(x, beta), 
                            args=(beta,), method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100})
            candidate = result.x.reshape(N, 2)
            cand_score = _compute_score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_x = candidate.flatten()
        except Exception:
            break
    return best_x.reshape(N, 2), best_score

def _force_directed_relax(centers, time_limit=5.0, target_radius=None):
    """Force-directed relaxation with velocity Verlet."""
    start = time.time()
    centers = centers.copy().astype(np.float64)
    if target_radius is None:
        target_radius = _compute_score(centers)
    r = target_radius * 1.005
    dt = 0.002
    damping = 0.85
    velocities = np.zeros_like(centers)
    
    def _compute_forces(pos, r):
        forces = np.zeros_like(pos)
        for i in range(N):
            for j in range(i + 1, N):
                diff = pos[i] - pos[j]
                dist = np.sqrt(np.sum(diff**2))
                min_dist = 2 * r
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    force_mag = 50.0 * overlap * overlap
                    force_dir = diff / dist
                    forces[i] += force_mag * force_dir
                    forces[j] -= force_mag * force_dir
        for i in range(N):
            dist_from_center = np.sqrt(np.sum(pos[i]**2))
            max_dist = 1.0 - r
            if dist_from_center > max_dist and dist_from_center > 1e-12:
                overlap = dist_from_center - max_dist
                force_mag = 100.0 * overlap * overlap
                force_dir = -pos[i] / dist_from_center
                forces[i] += force_mag * force_dir
        return forces
    
    forces = _compute_forces(centers, r)
    while time.time() - start < time_limit - 1.0:
        # Velocity Verlet
        centers += velocities * dt + 0.5 * forces * dt * dt
        new_forces = _compute_forces(centers, r)
        velocities = damping * velocities + 0.5 * (forces + new_forces) * dt
        forces = new_forces
        # Clamp to unit disk
        for i in range(N):
            d = np.sqrt(np.sum(centers[i]**2))
            if d > 1.0 - r:
                centers[i] = centers[i] * (1.0 - r) / d
    return centers

def _get_hex_lattice(n, radius, angle=0.0):
    """Generate hexagonal lattice points within unit circle."""
    spacing = 2 * radius
    sqrt3 = np.sqrt(3)
    points = []
    max_k = int(np.ceil(1.0 / spacing)) + 5
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            if np.sqrt(x**2 + y**2) <= 1.0 - radius + 1e-9:
                points.append([x, y])
    if abs(angle) > 1e-9:
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        points = [np.dot(rot, p) for p in points]
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    return np.array(points[:n])

def _get_ring_initialization(n, radii=None):
    """Initialize with concentric rings."""
    if radii is None:
        rings = [1, 6, 12, 18, 13]
        ring_radii = [0.0, 0.25, 0.50, 0.75, 0.90]
    else:
        rings = [1, 6, 12, 18, 13]
        ring_radii = radii
    points = []
    for ring_idx, (count, r) in enumerate(zip(rings, ring_radii)):
        if count == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points[:n])

def _get_fibonacci_initialization(n):
    """Initialize using Fibonacci spiral."""
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        r = 0.95 * np.sqrt((i + 0.5) / n)
        theta = i * golden_angle
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)

def _get_random_initialization(n, seed):
    """Random initialization within unit circle."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(n):
        r = np.sqrt(rng.random()) * 0.95
        theta = rng.random() * 2 * np.pi
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)

def _nelder_mead_polish(centers, time_limit=10.0):
    """Polish with Nelder-Mead on the exact score."""
    start = time.time()
    x0 = centers.flatten()
    def neg_score(x):
        c = x.reshape(N, 2)
        return -_compute_score(c)
    best_x = x0.copy()
    best_val = neg_score(x0)
    for _ in range(3):
        if time.time() - start > time_limit - 1.0:
            break
        try:
            result = minimize(neg_score, best_x, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            break
    return best_x.reshape(N, 2), -best_val

def _local_search(centers, time_limit=10.0):
    """Local search: try moving each point slightly in random directions."""
    start = time.time()
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    rng = np.random.RandomState(42)
    step_sizes = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    for step in step_sizes:
        if time.time() - start > time_limit - 1.0:
            break
        for _ in range(500):
            if time.time() - start > time_limit - 1.0:
                break
            idx = rng.randint(N)
            direction = rng.randn(2)
            direction /= np.linalg.norm(direction)
            candidate = best_centers.copy()
            candidate[idx] += step * direction
            d = np.linalg.norm(candidate[idx])
            if d > 1.0:
                candidate[idx] = candidate[idx] * (1.0 / d) * 0.999
            cand_score = _compute_score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_centers = candidate
    return best_centers, best_score

def _simulated_annealing(centers, time_limit=15.0):
    """Simulated annealing to escape local minima."""
    start = time.time()
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    current_centers = centers.copy()
    current_score = best_score
    T = 0.01
    cooling = 0.995
    rng = np.random.RandomState(7)
    
    while time.time() - start < time_limit - 1.0 and T > 1e-6:
        idx = rng.randint(N)
        move_size = T * 0.5
        direction = rng.randn(2)
        direction /= np.linalg.norm(direction)
        candidate = current_centers.copy()
        candidate[idx] += move_size * direction
        d = np.linalg.norm(candidate[idx])
        if d > 1.0:
            candidate[idx] = candidate[idx] * (1.0 / d) * 0.999
        cand_score = _compute_score(candidate)
        delta = cand_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_centers = candidate
            current_score = cand_score
            if cand_score > best_score:
                best_score = cand_score
                best_centers = candidate.copy()
        T *= cooling
    return best_centers, best_score

def construct():
    """Construct the best packing found within the time budget."""
    deadline = time.time() + 145.0
    best_centers = None
    best_score = 0.0
    
    # Generate multiple initializations
    initializations = []
    
    # Ring initialization
    initializations.append(_get_ring_initialization(N))
    
    # Hexagonal lattice with various angles and radii
    for angle in [0.0, np.pi/6, np.pi/3, np.pi/4, np.pi/2, 2*np.pi/3, np.pi/12, np.pi/8]:
        for radius in [0.12, 0.125, 0.128, 0.13]:
            try:
                init = _get_hex_lattice(N, radius, angle)
                if len(init) == N:
                    initializations.append(init)
            except Exception:
                pass
    
    # Fibonacci spiral
    initializations.append(_get_fibonacci_initialization(N))
    
    # Random initializations
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
        initializations.append(_get_random_initialization(N, seed))
    
    # Process each initialization
    for init_idx, init in enumerate(initializations):
        if time.time() > deadline - 50.0:
            break
        
        # Step 1: Force-directed relaxation
        try:
            relaxed = _force_directed_relax(init, time_limit=3.0)
            score = _compute_score(relaxed)
            if score > best_score:
                best_score = score
                best_centers = relaxed
        except Exception:
            relaxed = init
            score = _compute_score(relaxed)
            if score > best_score:
                best_score = score
                best_centers = relaxed
        
        # Step 2: Soft-min optimization with L-BFGS-B
        if time.time() > deadline - 35.0:
            break
        try:
            optimized, opt_score = _optimize_from_start(relaxed, beta_start=50.0, 
                                                        beta_end=1000.0, 
                                                        time_limit=8.0)
            if opt_score > best_score:
                best_score = opt_score
                best_centers = optimized
        except Exception:
            pass
        
        # Step 3: Simulated annealing to escape local minima
        if time.time() > deadline - 20.0:
            break
        try:
            sa_result, sa_score = _simulated_annealing(best_centers, time_limit=5.0)
            if sa_score > best_score:
                best_score = sa_score
                best_centers = sa_result
        except Exception:
            pass
        
        # Step 4: Nelder-Mead polish
        if time.time() > deadline - 10.0:
            break
        try:
            polished, pol_score = _nelder_mead_polish(best_centers, time_limit=4.0)
            if pol_score > best_score:
                best_score = pol_score
                best_centers = polished
        except Exception:
            pass
    
    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 15.0:
        try:
            remaining = min(15.0, deadline - time.time() - 5.0)
            polished, pol_score = _nelder_mead_polish(best_centers, time_limit=remaining)
            if pol_score > best_score:
                best_score = pol_score
                best_centers = polished
        except Exception:
            pass
    
    # Local search as final refinement
    if best_centers is not None and time.time() < deadline - 8.0:
        try:
            remaining = min(8.0, deadline - time.time() - 3.0)
            refined, ref_score = _local_search(best_centers, time_limit=remaining)
            if ref_score > best_score:
                best_score = ref_score
                best_centers = refined
        except Exception:
            pass
    
    # Final Nelder-Mead
    if best_centers is not None and time.time() < deadline - 3.0:
        try:
            remaining = max(3.0, deadline - time.time() - 1.0)
            polished, pol_score = _nelder_mead_polish(best_centers, time_limit=remaining)
            if pol_score > best_score:
                best_score = pol_score
                best_centers = polished
        except Exception:
            pass
    
    if best_centers is None:
        best_centers = _get_hex_lattice(N, 0.12)
    
    return best_centers
