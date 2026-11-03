"""Pack 50 equal circles in the unit disk via ring-based parametric optimization + L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 140.0  # safety margin before 150s limit


def _compute_score(pts):
    """Compute the packing radius: min of half-min-pair-dist and min-container-margin."""
    n = len(pts)
    # Container constraint
    container_margin = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    min_container = np.min(container_margin)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_container, min_pair)


def _ring_init(n_rings, radii, angles_offsets, seed=42):
    """
    Generate points on concentric rings.
    n_rings: list of number of points per ring (including center if first is 1)
    radii: list of radii for each ring
    angles_offsets: list of angular offsets for each ring
    """
    pts = []
    for k, (n_pts, r, offset) in enumerate(zip(n_rings, radii, angles_offsets)):
        if n_pts == 1:
            # Center point
            pts.append(np.array([0.0, 0.0]))
        else:
            angles = offset + 2 * np.pi * np.arange(n_pts) / n_pts
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            pts.append(np.column_stack([x, y]))
    return np.vstack(pts)


def _smooth_energy(pts, r_target, beta=80.0):
    """
    Smooth surrogate for constraint violation.
    Energy = soft sum of violations for pairs and container.
    """
    n = len(pts)
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Soft penalty for pairs closer than 2*r_target
    pair_viol = np.maximum(0, 2 * r_target - dist)
    # Use log-sum-exp style soft sum (smooth approximation of max/sum)
    # We want to penalize the worst violations, so use a soft-max
    # Actually, for optimization, sum of soft violations works well
    pair_energy = np.sum(np.logaddexp(0, -beta * pair_viol)) / beta
    
    # Container
    container_margin = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    cont_viol = np.maximum(0, r_target - container_margin)
    cont_energy = np.sum(np.logaddexp(0, -beta * cont_viol)) / beta
    
    return pair_energy + cont_energy


def _lbfgs_polish(pts, r_target, maxiter=200, beta=80.0):
    """Polish using L-BFGS-B to minimize smooth energy for a target radius."""
    n = len(pts)
    
    def objective(flat):
        p = flat.reshape(n, 2)
        return _smooth_energy(p, r_target, beta=beta)
    
    def gradient(flat):
        p = flat.reshape(n, 2)
        # Analytical gradient
        n_pts = n
        grad = np.zeros_like(flat)
        
        # Pairwise gradient
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        pair_viol = np.maximum(0, 2 * r_target - dist)
        # Derivative of logaddexp(0, -beta*v) w.r.t. v is -beta * sigmoid(-beta*v)
        # But we have sum over pairs, and each pair (i,j) contributes to grad_i and grad_j
        # d/dp_i of logaddexp(0, -beta*(2r - d_ij)) = -beta * sigmoid(-beta*(2r-d_ij)) * d/dp_i(2r-d_ij)
        # d/dp_i (2r - d_ij) = -(p_i - p_j)/d_ij for i != j
        # So contribution to grad_i from pair (i,j) = beta * sigmoid(-beta*(2r-d_ij)) * (p_i-p_j)/d_ij
        
        # This is O(n^2) which is fine for n=50
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = dist[i, j]
                if d < 1e-10:
                    continue
                v = 2 * r_target - d
                if v <= 0:
                    # sigmoid(-beta*v) is ~1 when v<0, so sigmoid(-beta*v) = 1/(1+exp(beta*v))
                    # When v < 0, exp(beta*v) < 1, so sigmoid is > 0.5
                    # Actually let's just compute it
                    pass
                sig = 1.0 / (1.0 + np.exp(beta * v)) if v > -50 else 0.0  # sigmoid(-beta*v)
                # Contribution to grad_i: beta * sig * (p_i - p_j) / d
                # Contribution to grad_j: beta * sig * (p_j - p_i) / d = -beta * sig * (p_i - p_j) / d
                factor = beta * sig * (p[i] - p[j]) / d
                grad[2*i:2*i+2] += factor
                grad[2*j:2*j+2] -= factor
        
        # Container gradient
        norms = np.sqrt(np.sum(p ** 2, axis=1))
        cont_margin = 1.0 - norms
        cont_viol = np.maximum(0, r_target - cont_margin)
        for i in range(n_pts):
            v = r_target - cont_margin[i]
            if norms[i] < 1e-10:
                continue
            sig = 1.0 / (1.0 + np.exp(beta * v)) if v > -50 else 0.0
            factor = beta * sig * p[i] / norms[i]
            grad[2*i:2*i+2] += factor
        
        return grad
    
    result = minimize(
        objective,
        pts.flatten(),
        method='L-BFGS-B',
        jac=gradient,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    return result.x.reshape(n, 2)


def _parametric_ring_optimize(seed=42, max_iter_de=300):
    """
    Optimize a ring-based packing using differential evolution on ring parameters.
    Structure: center + rings with varying counts.
    We'll use a flexible ring structure and optimize radii and angular offsets.
    """
    t0 = time.time()
    
    # Define a ring structure that sums to 50
    # Try several structures
    structures = [
        [1, 6, 12, 18, 13],   # center + 4 rings
        [1, 6, 12, 14, 17],
        [1, 6, 10, 14, 19],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 18, 13],
        [1, 5, 10, 15, 19],
        [1, 6, 11, 15, 17],
        [1, 6, 12, 13, 18],
    ]
    
    best_score = 0
    best_pts = None
    
    for struct in structures:
        if time.time() - t0 > DEADLINE - 40:
            break
        
        n_rings = len(struct)
        # Parameters: radii (n_rings-1 values, center is 0) + angular offsets (n_rings-1 values)
        n_params = 2 * (n_rings - 1)
        
        # Bounds for radii: [0.05, 0.95]
        # Bounds for offsets: [0, 2*pi]
        bounds = []
        for i in range(n_rings - 1):
            bounds.append((0.05, 0.95))  # radius
        for i in range(n_rings - 1):
            bounds.append((0.0, 2 * np.pi))  # offset
        
        def objective(params):
            radii = params[:n_rings-1]
            offsets = params[n_rings-1:]
            pts = _ring_init(struct, np.concatenate([[0.0], radii]), np.concatenate([[0.0], offsets]))
            return -_compute_score(pts)
        
        # Use differential evolution
        try:
            result = differential_evolution(
                objective,
                bounds,
                seed=seed,
                maxiter=max_iter_de,
                tol=1e-8,
                polish=True,
                updating='immediate',
                workers=1,
            )
            
            if result.success or result.fun < 0:
                radii = result.x[:n_rings-1]
                offsets = result.x[n_rings-1:]
                pts = _ring_init(struct, np.concatenate([[0.0], radii]), np.concatenate([[0.0], offsets]))
                score = _compute_score(pts)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
        except Exception:
            pass
    
    return best_pts, best_score


def _local_search(pts, r_target, n_iters=5000, seed=12345):
    """
    Local search: perturb one point at a time, accept if score improves.
    """
    rng = np.random.default_rng(seed)
    best_pts = pts.copy()
    best_score = _compute_score(best_pts)
    
    step_size = 0.02
    temperature = 0.005
    
    for it in range(n_iters):
        idx = rng.integers(0, N)
        old_pos = best_pts[idx].copy()
        new_pos = old_pos + step_size * rng.normal(size=2)
        
        # Keep inside container
        new_r = np.sqrt(np.sum(new_pos ** 2))
        if new_r > 0.99:
            new_pos *= 0.99 / new_r
        
        test_pts = best_pts.copy()
        test_pts[idx] = new_pos
        new_score = _compute_score(test_pts)
        
        delta = new_score - best_score
        if delta > 0 or rng.random() < np.exp(delta / temperature):
            best_pts[idx] = new_pos
            if new_score > best_score:
                best_score = new_score
        
        temperature *= 0.99995
        step_size = max(0.001, step_size * 0.99995)
    
    return best_pts, best_score


def construct():
    t0 = time.time()
    
    # Step 1: Parametric ring optimization via differential evolution
    ring_pts, ring_score = _parametric_ring_optimize(seed=42, max_iter_de=200)
    
    if ring_pts is None:
        # Fallback: simple ring
        ring_pts = _ring_init([1, 6, 12, 18, 13], [0, 0.2, 0.45, 0.65, 0.85], [0, 0, np.pi/12, np.pi/18, 0])
        ring_score = _compute_score(ring_pts)
    
    best_pts = ring_pts.copy()
    best_score = ring_score
    
    # Step 2: Local search to refine
    if time.time() - t0 < DEADLINE - 30:
        best_pts, best_score = _local_search(best_pts, best_score, n_iters=3000, seed=12345)
    
    # Step 3: L-BFGS-B polish
    if time.time() - t0 < DEADLINE - 15:
        r_target = best_score * 1.0005
        polished = _lbfgs_polish(best_pts, r_target, maxiter=150, beta=80.0)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # Step 4: Another local search round
    if time.time() - t0 < DEADLINE - 10:
        best_pts, best_score = _local_search(best_pts, best_score, n_iters=2000, seed=99999)
    
    # Step 5: Final L-BFGS-B polish with tighter tolerance
    if time.time() - t0 < DEADLINE - 5:
        r_target = best_score * 1.0002
        polished = _lbfgs_polish(best_pts, r_target, maxiter=100, beta=100.0)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # Validate
    radii = np.sqrt(np.sum(best_pts ** 2, axis=1))
    if np.any(~np.isfinite(best_pts)) or np.any(radii >= 1.0):
        # Fallback
        best_pts = _ring_init([1, 6, 12, 18, 13], [0, 0.2, 0.45, 0.65, 0.85], [0, 0, np.pi/12, np.pi/18, 0])
    
    return best_pts
