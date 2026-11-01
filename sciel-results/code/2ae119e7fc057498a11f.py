"""Packing 50 equal circles in a unit circle - simulated annealing optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Pairwise distances using broadcasting
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _energy(pts, r_target):
    """
    Energy function: penalizes overlaps and boundary violations.
    Lower is better. We want to maximize r, so we minimize violations.
    """
    pts = np.asarray(pts)
    n = len(pts)
    
    # Boundary violation
    dists = np.hypot(pts[:, 0], pts[:, 1])
    boundary_viol = np.maximum(0, dists + r_target - 1.0)
    
    # Pairwise overlap violation
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    d = np.sqrt(dx * dx + dy * dy)
    overlap_viol = np.maximum(0, 2 * r_target - d)
    
    # Sum of squared violations (smooth, differentiable)
    energy = np.sum(boundary_viol ** 2) + np.sum(overlap_viol ** 2)
    
    return energy


def _concentric_ring_init(n, seed=42):
    """
    Generate initial positions using concentric rings.
    Ring 0: 1 point at center
    Ring k: 6*k points on circle of radius r_k
    Choose radii to maximize spacing.
    """
    rng = np.random.RandomState(seed)
    pts = []
    
    # Center point
    pts.append([0.0, 0.0])
    
    # Add rings until we have n points
    ring = 1
    while len(pts) < n:
        n_on_ring = 6 * ring
        # Radius of this ring: approximate optimal spacing
        # For 6k points on a ring, radius ~ k * (2/6) * sqrt(3) * r_circle
        # Simplified: r_k ≈ k / 3
        r_k = ring / 3.5  # Tuned for n=50
        if r_k >= 0.95:
            r_k = 0.95
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        for a in angles:
            pts.append([r_k * np.cos(a), r_k * np.sin(a)])
        ring += 1
    
    pts = np.array(pts[:n])
    
    # Add small random perturbation to break symmetry
    pts += rng.randn(n, 2) * 0.01
    
    # Project inside unit circle
    r = np.hypot(pts[:, 0], pts[:, 1])
    mask = r > 0.999
    if np.any(mask):
        pts[mask] = pts[mask] / r[mask, None] * 0.999
    
    return pts


def _random_init(n, seed=42):
    """Random initial positions inside unit circle."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(rng.uniform(0, 1, n))
    pts = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
    return pts


def _simulated_annealing(pts, r_target, max_iters=50000, T_init=0.05, T_min=1e-6, cooling=0.9998, seed=42):
    """
    Simulated annealing to minimize energy for a given target radius.
    """
    rng = np.random.RandomState(seed)
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    T = T_init
    
    best_energy = _energy(pts, r_target)
    best_pts = pts.copy()
    current_energy = best_energy
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Propose a move: perturb one or two random points
        n_moves = rng.choice([1, 2], p=[0.7, 0.3])
        idx = rng.randint(0, n, size=n_moves)
        
        new_pts = pts.copy()
        step = 0.02 * T / T_init  # Scale step with temperature
        new_pts[idx] += rng.randn(n_moves, 2) * step
        
        # Project inside unit circle
        r = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / r[mask, None] * 0.9999
        
        new_energy = _energy(new_pts, r_target)
        
        # Accept if better or with probability exp(-dE/T)
        dE = new_energy - current_energy
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-12)):
            pts = new_pts
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = pts.copy()
        
        T *= cooling
        if T < T_min:
            break
    
    return best_pts, best_energy


def _local_refine(pts, r_target, max_iters=2000):
    """
    Local refinement using gradient-based optimization to push r higher.
    """
    from scipy.optimize import minimize
    
    def objective(x):
        pts_2d = x.reshape(-1, 2)
        # Penalty for violations
        e = _energy(pts_2d, r_target)
        # Small regularization to avoid degenerate solutions
        r = np.hypot(pts_2d[:, 0], pts_2d[:, 1])
        e += 0.001 * np.sum(r ** 2)
        return e
    
    def jac(x):
        pts_2d = x.reshape(-1, 2)
        n = len(pts_2d)
        grad = np.zeros_like(x)
        
        # Boundary gradient
        dists = np.hypot(pts_2d[:, 0], pts_2d[:, 1])
        boundary_viol = np.maximum(0, dists + r_target - 1.0)
        for i in range(n):
            if boundary_viol[i] > 0:
                r_safe = max(dists[i], 1e-12)
                grad[2*i] += 2 * boundary_viol[i] * pts_2d[i, 0] / r_safe
                grad[2*i+1] += 2 * boundary_viol[i] * pts_2d[i, 1] / r_safe
        
        # Pairwise gradient
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = pts_2d[i_idx, 0] - pts_2d[j_idx, 0]
        dy = pts_2d[i_idx, 1] - pts_2d[j_idx, 1]
        d = np.sqrt(dx * dx + dy * dy)
        d = np.maximum(d, 1e-12)
        overlap_viol = np.maximum(0, 2 * r_target - d)
        
        for k in range(len(i_idx)):
            if overlap_viol[k] > 0:
                # d/dx_i of (2r - d)^2 = -2(2r-d) * (dx/dx_i)
                # dx/dx_i = -1, so contribution = +2(2r-d) * dx/d
                factor = 2 * overlap_viol[k] / d
                grad[2*i_idx[k]] += factor * dx[k]
                grad[2*i_idx[k]+1] += factor * dy[k]
                grad[2*j_idx[k]] -= factor * dx[k]
                grad[2*j_idx[k]+1] -= factor * dy[k]
        
        # Regularization gradient
        r = np.hypot(pts_2d[:, 0], pts_2d[:, 1])
        for i in range(n):
            r_safe = max(r[i], 1e-12)
            grad[2*i] += 0.002 * pts_2d[i, 0]
            grad[2*i+1] += 0.002 * pts_2d[i, 1]
        
        return grad
    
    x0 = pts.flatten()
    try:
        result = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                         options={'maxiter': max_iters, 'maxfun': 100000})
        refined_pts = result.x.reshape(-1, 2)
        # Project inside
        r = np.hypot(refined_pts[:, 0], refined_pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            refined_pts[mask] = refined_pts[mask] / r[mask, None] * 0.9999
        return refined_pts
    except Exception:
        return pts


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0
    
    # Strategy: binary search on radius r, using simulated annealing + local refinement
    # to find a valid packing for each r. Find the maximum r that works.
    
    # Initial guess for r
    r_low = 0.10
    r_high = 0.14  # Best known is ~0.1258
    
    # First, try to find a good packing at a moderate r using SA
    initial_r = 0.11
    
    # Try multiple initializations
    inits = []
    inits.append(_concentric_ring_init(N, seed=42))
    inits.append(_random_init(N, seed=123))
    inits.append(_concentric_ring_init(N, seed=999))
    
    best_init_pts = None
    best_init_score = 0.0
    
    for init_pts in inits:
        if time.time() > _DEADLINE - 20.0:
            break
        
        # Run simulated annealing at initial_r
        sa_pts, sa_energy = _simulated_annealing(
            init_pts, initial_r,
            max_iters=30000, T_init=0.05, T_min=1e-5, cooling=0.9999,
            seed=42
        )
        
        # Local refinement
        refined_pts = _local_refine(sa_pts, initial_r, max_iters=1000)
        score = _score(refined_pts)
        
        if score > best_init_score:
            best_init_score = score
            best_init_pts = refined_pts
    
    if best_init_pts is not None:
        best_pts = best_init_pts
        best_score = best_init_score
    
    # Now binary search for higher r
    if best_score > 0 and time.time() < _DEADLINE - 30.0:
        r_low = best_score
        r_high = min(best_score + 0.02, 0.14)
        
        # Try a few increasing r values
        for trial in range(5):
            if time.time() > _DEADLINE - 10.0:
                break
            
            r_mid = (r_low + r_high) / 2
            
            # Try to find a packing at r_mid
            trial_pts = best_pts.copy() + np.random.RandomState(42 + trial).randn(N, 2) * 0.01
            r = np.hypot(trial_pts[:, 0], trial_pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                trial_pts[mask] = trial_pts[mask] / r[mask, None] * 0.9999
            
            sa_pts, sa_energy = _simulated_annealing(
                trial_pts, r_mid,
                max_iters=20000, T_init=0.03, T_min=1e-5, cooling=0.9999,
                seed=100 + trial
            )
            
            refined_pts = _local_refine(sa_pts, r_mid, max_iters=500)
            score = _score(refined_pts)
            
            if score >= r_mid - 1e-9:
                # Success! Update best
                if score > best_score:
                    best_score = score
                    best_pts = refined_pts.copy()
                r_low = r_mid
            else:
                r_high = r_mid
    
    # Final local refinement on the best found
    if best_pts is not None and time.time() < _DEADLINE - 5.0:
        final_pts = _local_refine(best_pts, best_score, max_iters=500)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score
    
    if best_pts is None:
        # Fallback
        best_pts = _concentric_ring_init(N, seed=42)
    
    # Final validation and cleanup
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r[mask, None] * 0.9999
    
    return best_pts
