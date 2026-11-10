"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    x = centers[:, 0]
    y = centers[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.hypot(dx, dy)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate a hexagonal lattice packing inside the unit circle."""
    best_points = None
    best_radius = 0.0
    
    for spacing in np.linspace(0.15, 0.25, 30):
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            points = points[np.argsort(radius, kind='stable')[:N]]
            r = _compute_radius(points)
            if r > best_radius:
                best_radius = r
                best_points = points.copy()
    
    if best_points is not None:
        return best_points
    
    spacing = 0.2
    k = int(1.0 / spacing) + 3
    rows = [(i * spacing, j * spacing) for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    radius = radius[mask]
    if len(points) >= N:
        return points[np.argsort(radius, kind='stable')[:N]]
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

def _concentric_rings(n_rings=5, center_frac=0.1):
    """Generate concentric ring initial configuration."""
    points = []
    if center_frac > 0:
        points.append([0.0, 0.0])
    
    remaining = N - len(points)
    if remaining <= 0:
        return np.array(points[:N])
    
    # Distribute points among rings
    ring_counts = []
    r = 1
    while remaining > 0 and r <= n_rings:
        count = min(remaining, int(2 * np.pi * r * 0.15 / 0.03))
        count = max(count, 3)
        if r == n_rings:
            count = remaining
        ring_counts.append(count)
        remaining -= count
        r += 1
    
    # Adjust if we have too many or too few
    total = sum(ring_counts)
    if total != N - len(points):
        ring_counts = []
        per_ring = max(1, (N - len(points)) // n_rings)
        for i in range(n_rings):
            if i < n_rings - 1:
                ring_counts.append(per_ring)
            else:
                ring_counts.append(N - len(points) - sum(ring_counts))
    
    # Place points on rings
    for ring_idx, count in enumerate(ring_counts):
        radius = (ring_idx + 1) * 0.18
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for a in angles:
            points.append([radius * np.cos(a), radius * np.sin(a)])
    
    return np.array(points[:N])

def _symmetric_sectors(n_sectors=5):
    """Generate symmetric sector initial configuration."""
    points = []
    points.append([0.0, 0.0])
    
    remaining = N - 1
    per_sector = remaining // n_sectors
    
    for s in range(n_sectors):
        base_angle = 2 * np.pi * s / n_sectors
        for i in range(per_sector):
            r = 0.2 + 0.15 * (i % 3)
            angle = base_angle + 0.2 * (i // 3)
            points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Add any remaining points
    extra = N - len(points)
    if extra > 0:
        angles = np.linspace(0, 2 * np.pi, extra, endpoint=False)
        for a in angles:
            points.append([0.85 * np.cos(a), 0.85 * np.sin(a)])
    
    return np.array(points[:N])

def _softmin_obj_and_grad(flat_centers, beta=50.0):
    """
    Objective: minimize -softmin of (pairwise dists/2, boundary dists)
    Gradient computed analytically.
    """
    centers = flat_centers.reshape(N, 2)
    
    # Boundary distances
    r_norm = np.hypot(centers[:, 0], centers[:, 1])
    boundary_dists = 1.0 - r_norm
    
    # Pairwise distances / 2
    x = centers[:, 0]
    y = centers[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.hypot(dx, dy)
    np.fill_diagonal(dists, 0.0)
    pair_dists = dists / 2.0
    
    # Collect all constraints
    all_dists = np.concatenate([
        boundary_dists,
        pair_dists[np.triu_indices(N, k=1)]
    ])
    
    # Soft-min computation
    shifted = -beta * all_dists
    shifted -= np.max(shifted)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp <= 0:
        return 0.0, np.zeros_like(flat_centers)
    
    weights = exp_shifted / sum_exp
    grad_all_dists = -weights
    
    # Map back to centers
    grad_centers = np.zeros((N, 2))
    
    # Boundary part
    for i in range(N):
        if r_norm[i] > 1e-12:
            grad_centers[i] += grad_all_dists[i] * (-centers[i] / r_norm[i])
    
    # Pairwise part
    idx_i, idx_j = np.triu_indices(N, k=1)
    for k in range(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        d = dists[i, j]
        if d < 1e-12:
            continue
        w = grad_all_dists[N + k]
        diff = centers[i] - centers[j]
        grad_centers[i] += w * diff / (2.0 * d)
        grad_centers[j] += w * (-diff) / (2.0 * d)
    
    obj = np.log(sum_exp) / beta
    grad_flat = grad_centers.flatten()
    
    return obj, grad_flat

def _polish(centers, max_time=30.0, beta_start=50.0, beta_end=500.0, n_betas=4):
    """Polish using L-BFGS-B with adaptive beta schedule."""
    start = time.time()
    current_centers = centers.copy()
    
    betas = np.geomspace(beta_start, beta_end, n_betas)
    
    for beta in betas:
        if time.time() - start > max_time:
            break
        
        def objective(flat, b=beta):
            obj, grad = _softmin_obj_and_grad(flat, beta=b)
            return obj, grad
        
        flat_init = current_centers.flatten()
        
        try:
            result = minimize(
                objective,
                flat_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000}
            )
            
            if np.all(np.isfinite(result.x)):
                polished = result.x.reshape(N, 2)
                new_radius = _compute_radius(polished)
                old_radius = _compute_radius(current_centers)
                if new_radius >= old_radius:
                    current_centers = polished
        except Exception:
            pass
    
    return current_centers

def _high_precision_polish(centers, max_time=10.0):
    """Final high-precision polish."""
    start = time.time()
    
    def objective(flat):
        obj, grad = _softmin_obj_and_grad(flat, beta=1000.0)
        return obj, grad
    
    flat_init = centers.flatten()
    
    try:
        result = minimize(
            objective,
            flat_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-14, 'maxfun': 500}
        )
        
        if np.all(np.isfinite(result.x)):
            polished = result.x.reshape(N, 2)
            new_radius = _compute_radius(polished)
            old_radius = _compute_radius(centers)
            if new_radius >= old_radius:
                return polished
    except Exception:
        pass
    
    return centers

def construct():
    """Construct the packing of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate diverse initial configurations
    inits = []
    
    # Hexagonal lattice
    hex_init = _hexagonal_packing()
    inits.append(hex_init)
    
    # Concentric rings
    if time.time() < deadline - 60:
        rings_init = _concentric_rings()
        inits.append(rings_init)
    
    # Symmetric sectors
    if time.time() < deadline - 60:
        sectors_init = _symmetric_sectors()
        inits.append(sectors_init)
    
    # Perturbed hexagonal
    if time.time() < deadline - 60:
        pert_hex = hex_init + np.random.normal(0, 0.005, (N, 2))
        inits.append(pert_hex)
    
    # Best result tracking
    best_centers = None
    best_radius = 0.0
    
    # Polish each initial configuration
    for idx, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - time.time() - 10
        max_polish_time = min(25.0, remaining)
        
        polished = _polish(init, max_time=max_polish_time)
        radius = _compute_radius(polished)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = polished.copy()
    
    # If we have time, do more perturbations from best
    if best_centers is not None and time.time() < deadline - 15:
        for restart in range(3):
            if time.time() > deadline - 12:
                break
            perturbed = best_centers + np.random.normal(0, 0.002, (N, 2))
            remaining = deadline - time.time() - 8
            polished = _polish(perturbed, max_time=min(15.0, remaining))
            radius = _compute_radius(polished)
            if radius > best_radius:
                best_radius = radius
                best_centers = polished.copy()
    
    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        polished = _high_precision_polish(best_centers, max_time=min(8.0, remaining))
        radius = _compute_radius(polished)
        if radius > best_radius:
            best_radius = radius
            best_centers = polished.copy()
    
    # Final validation
    if best_centers is not None:
        final_radius = _compute_radius(best_centers)
        if final_radius > 0:
            return best_centers
    
    return hex_init
# EVOLVE-BLOCK-END
