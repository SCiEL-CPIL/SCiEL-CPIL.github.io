"""Packing 50 equal circles in a unit circle - soft-min surrogate with L-BFGS-B."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


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
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _hex_lattice_init(n, jitter=0.0):
    """Generate initial positions from hexagonal lattice, trimmed to n points."""
    for spacing in np.linspace(0.14, 0.35, 40):
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= n:
            idx = np.random.permutation(len(pts))[:n]
            pts = pts[idx]
            if jitter > 0:
                pts += np.random.randn(n, 2) * jitter
            # Project back inside unit circle
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] /= r[mask, None] * 1.001
            return pts
    # Fallback: random
    theta = np.random.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _random_init(n):
    """Generate random initial positions within the unit disk."""
    theta = np.random.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _concentric_ring_init(n):
    """Generate initial positions on concentric rings."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    # Rings
    remaining = n - 1
    ring_radius = 0.2
    while remaining > 0:
        # How many points fit on this ring?
        circumference = 2 * np.pi * ring_radius
        # Minimum spacing ~ 0.25 for reasonable packing
        n_on_ring = max(1, int(circumference / 0.25))
        n_on_ring = min(n_on_ring, remaining)
        if n_on_ring == 0:
            ring_radius += 0.15
            continue
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        for a in angles:
            pts.append([ring_radius * np.cos(a), ring_radius * np.sin(a)])
        remaining -= n_on_ring
        ring_radius += 0.18
    pts = np.array(pts[:n])
    # Ensure inside unit circle
    r = np.hypot(pts[:, 0], pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        pts[mask] /= r[mask, None] * 1.001
    return pts


def _soft_min_surrogate(pts, beta):
    """
    Compute a soft-min surrogate for the packing radius.
    
    The packing radius is min(r_cont, min_pair/2).
    We approximate min(a, b) using the log-sum-exp trick:
    min(a,b) ≈ -(1/beta) * log(exp(-beta*a) + exp(-beta*b))
    
    This gives a smooth, differentiable approximation that converges to min as beta→∞.
    """
    if len(pts) != N:
        return 1e6, np.zeros(2*N)
    if not np.all(np.isfinite(pts)):
        return 1e6, np.zeros(2*N)
    
    # Container constraint: r_cont = 1 - max(|c_i|)
    dists = np.hypot(pts[:, 0], pts[:, 1])
    r_cont = 1.0 - np.max(dists)
    
    if r_cont < -0.1:
        return 1e6, np.zeros(2*N)
    
    # Pairwise distances
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    d_sq = dx * dx + dy * dy
    d = np.sqrt(d_sq + 1e-12)
    
    # We want to maximize the minimum of {r_cont, d_ij/2 for all i<j}
    # Using soft-min: -softmin = -(-(1/beta)*log(sum(exp(-beta*x))))
    # Let's work with the negative to minimize
    
    # All "radius candidates": r_cont and all d_ij/2
    half_d = d / 2.0
    # Combine into one array
    all_radii = np.concatenate([[r_cont], half_d])
    
    # Soft min: min(x) ≈ -(1/beta) * log(sum(exp(-beta*x)))
    # For numerical stability, subtract max first
    x_max = np.max(all_radii)
    exp_terms = np.exp(-beta * (all_radii - x_max))
    soft_min_val = x_max - (1.0 / beta) * np.log(np.sum(exp_terms))
    
    # Gradient with respect to pts
    # d(soft_min)/d(all_radii[k]) = -exp(-beta*(x_k - x_max)) / sum(exp(-beta*(x_j - x_max)))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros(2*N)
    
    # Gradient for r_cont = 1 - max(|c_i|)
    # d(r_cont)/d(c_i) = -c_i / |c_i| for the max point
    max_idx = np.argmax(dists)
    if dists[max_idx] > 1e-10:
        grad[2*max_idx] = -weights[0] * pts[max_idx, 0] / dists[max_idx]
        grad[2*max_idx + 1] = -weights[0] * pts[max_idx, 1] / dists[max_idx]
    
    # Gradient for each d_ij/2
    # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    # d(d_ij/2)/d(c_i) = (c_i - c_j) / (2*d_ij)
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        w = weights[k+1]
        if d[k] > 1e-10:
            grad[2*i] += w * (pts[i, 0] - pts[j, 0]) / (2.0 * d[k])
            grad[2*i+1] += w * (pts[i, 1] - pts[j, 1]) / (2.0 * d[k])
            grad[2*j] += w * (pts[j, 0] - pts[i, 0]) / (2.0 * d[k])
            grad[2*j+1] += w * (pts[j, 1] - pts[i, 1]) / (2.0 * d[k])
    
    # We want to MINIMIZE -soft_min, so return (-soft_min, -grad)
    return -soft_min_val, -grad


def _soft_min_optimize(pts, beta_start=10.0, beta_end=5000.0, n_phases=8):
    """
    Optimize using soft-min surrogate with L-BFGS-B, gradually increasing beta.
    
    Args:
        pts: initial positions (N, 2)
        beta_start: initial sharpness of soft-min
        beta_end: final sharpness
        n_phases: number of beta refinement phases
    
    Returns:
        best_pts, best_score
    """
    pts = pts.copy().astype(np.float64)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    betas = np.geomspace(beta_start, beta_end, n_phases)
    
    for beta in betas:
        if time.time() > _DEADLINE - 5.0:
            break
        
        def objective_and_grad(flat):
            p = flat.reshape(N, 2)
            val, grad = _soft_min_surrogate(p, beta)
            return val, grad
        
        flat = pts.flatten()
        
        try:
            result = minimize(
                objective_and_grad,
                flat,
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'gtol': 1e-8,
                    'ftol': 1e-10,
                    'maxfun': 5000
                }
            )
            new_pts = result.x.reshape(N, 2)
        except Exception:
            continue
        
        # Project inside container
        r = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            new_pts[mask] /= r[mask, None] * 1.0001
        
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
        else:
            # Keep old pts if no improvement
            pass
    
    return best_pts, best_score


def _simulated_annealing(pts, max_iters=5000, T_init=0.1, T_min=1e-4, cooling=0.999):
    """Simulated annealing to escape local minima."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    T = T_init
    best_score = _score(pts)
    best_pts = pts.copy()
    current_score = best_score
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Random perturbation
        scale = 0.05 * T / T_init + 0.001
        new_pts = pts + np.random.randn(n, 2) * scale
        
        # Keep inside container
        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * 1.0001
        
        new_score = _score(new_pts)
        
        # Accept or reject
        if new_score >= current_score:
            pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            # Metropolis criterion
            delta = current_score - new_score
            if delta > 0 and np.random.random() < np.exp(-delta / max(T, 1e-10)):
                pts = new_pts
                current_score = new_score
        
        T = max(T * cooling, T_min)
    
    return best_pts, best_score


def _polish(pts):
    """
    Final high-precision polish using very high beta soft-min.
    """
    if time.time() > _DEADLINE - 3.0:
        return pts, _score(pts)
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for beta in [2000, 5000, 10000]:
        if time.time() > _DEADLINE - 2.0:
            break
        
        def obj_and_grad(flat):
            p = flat.reshape(N, 2)
            val, grad = _soft_min_surrogate(p, beta)
            return val, grad
        
        flat = best_pts.flatten()
        try:
            result = minimize(
                obj_and_grad,
                flat,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12, 'maxfun': 2000}
            )
            new_pts = result.x.reshape(N, 2)
            r = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                new_pts[mask] /= r[mask, None] * 1.0001
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            pass
    
    return best_pts, best_score


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0
    
    # Try multiple initializations and optimization strategies
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Choose initialization strategy
        if restart == 0:
            pts = _hex_lattice_init(N, jitter=0.0)
        elif restart == 1:
            pts = _hex_lattice_init(N, jitter=0.02)
        elif restart == 2:
            pts = _concentric_ring_init(N)
        elif restart == 3:
            pts = _random_init(N)
        elif restart == 4:
            pts = _hex_lattice_init(N, jitter=0.05)
        elif restart == 5:
            pts = _concentric_ring_init(N) + np.random.randn(N, 2) * 0.03
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] /= r[mask, None] * 1.001
        elif restart == 6:
            pts = _hex_lattice_init(N, jitter=0.01)
        elif restart == 7:
            pts = _random_init(N)
        else:
            # Random perturbations of hex lattice
            base = _hex_lattice_init(N, jitter=0.0)
            pts = base + np.random.randn(N, 2) * 0.03 * (restart - 6)
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] /= r[mask, None] * 1.001
        
        # Run soft-min L-BFGS-B optimization
        optimized_pts, score = _soft_min_optimize(
            pts,
            beta_start=10.0,
            beta_end=5000.0,
            n_phases=6
        )
        
        if score > best_score:
            best_score = score
            best_pts = optimized_pts.copy()
        
        # If we have time, try simulated annealing from the best point
        if time.time() < _DEADLINE - 25.0 and restart % 3 == 0:
            sa_pts, sa_score = _simulated_annealing(
                best_pts,
                max_iters=2000,
                T_init=0.06,
                T_min=1e-4,
                cooling=0.9995
            )
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts.copy()
    
    if best_pts is None:
        # Fallback
        best_pts = _hex_lattice_init(N, jitter=0.0)
        best_score = _score(best_pts)
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 3.0:
        polished_pts, polished_score = _polish(best_pts)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
    
    # Final validation and cleanup
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] /= r[mask, None] * 1.0001
    
    return best_pts
