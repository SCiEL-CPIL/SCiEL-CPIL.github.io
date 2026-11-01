"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
# Safety margin: Evaluator kills at 150s. We aim to finish well before 120s.
_DEADLINE = _START_TIME + 115.0


def _score(positions):
    """
    Compute the valid packing radius (min of pairwise dist/2 and dist to boundary).
    Returns 0 if invalid (non-finite, outside container).
    """
    n = len(positions)
    if not np.all(np.isfinite(positions)):
        return 0.0
    
    radii = np.hypot(positions[:, 0], positions[:, 1])
    if np.any(radii > 1.0):
        return 0.0
        
    min_boundary = 1.0 - np.max(radii)
    if min_boundary < 0:
        return 0.0
        
    if n < 2:
        return min_boundary

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_boundary)


def _soft_min(values, beta):
    """Compute the soft minimum of an array of values."""
    v = np.array(values)
    m = np.min(v)
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))


def _objective_and_gradient(params, n, beta):
    """
    Contact-graph-weighted objective: focus gradient on binding constraints.
    Uses a sharpened soft-min that heavily weights the minimum constraints.
    """
    centers = params.reshape(n, 2)
    
    # Pairwise distances
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = centers[i_idx, :] - centers[j_idx, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    boundary_dists = 1.0 - norms
    
    vals = np.concatenate([dists / 2.0, boundary_dists])
    
    # Sharpened soft-min: use higher effective beta for contact identification
    m = np.min(vals)
    # Use a very sharp soft-min to identify critical contacts
    beta_sharp = max(beta * 5.0, 500.0)
    exp_vals = np.exp(-beta_sharp * (vals - m))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    # Renormalize to sum to 1 (they already do)
    obj = -(m - (1.0 / beta_sharp) * np.log(S))
    
    # Vectorized gradient
    grad = np.zeros_like(centers)
    
    # Pairwise part
    pair_weights = weights[:len(i_idx)]
    
    mask = pair_weights > 1e-10
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        w_m = pair_weights[mask]
        d_m = dists[mask]
        
        safe_d = np.where(d_m > 1e-8, d_m, 1.0)
        vec = (centers[i_m, :] - centers[j_m, :]) / safe_d[:, np.newaxis]
        coeff = w_m * 0.5 / safe_d
        
        np.add.at(grad, i_m, coeff[:, np.newaxis] * vec)
        np.add.at(grad, j_m, -coeff[:, np.newaxis] * vec)
    
    # Boundary part
    bound_weights = weights[len(i_idx):]
    mask_b = bound_weights > 1e-10
    if np.any(mask_b):
        idx_b = np.where(mask_b)[0]
        w_b = bound_weights[mask_b]
        norm_b = norms[idx_b]
        safe_norm = np.where(norm_b > 1e-8, norm_b, 1.0)
        grad[idx_b] -= (w_b / safe_norm)[:, np.newaxis] * centers[idx_b, :]
    
    return obj, -grad


def _optimize_lbfgs(centers, beta, maxiter, gtol):
    """
    Optimize using L-BFGS-B.
    """
    try:
        res = minimize(
            _objective_and_gradient, 
            centers.flatten(), 
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'disp': False}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers


def _get_hex_lattice(n, seed_offset=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(42 + seed_offset)
    s_est = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    
    points = []
    max_range = int(2.0 / s_est) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s_est + (j % 2) * s_est / 2.0
            y = j * s_est * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) < n:
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    points = points + rng.normal(0, 0.01, points.shape)
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]


def _get_ring_init(n, seed=0):
    """Generate concentric ring initialization matching known optimal structure."""
    rng = np.random.default_rng(seed)
    
    # Known optimal packing of 50 circles has a ring structure.
    # Use rings with radii tuned for ~50 circles.
    # Ring radii approximately: 0, 0.25, 0.5, 0.75, 0.95
    # Points per ring: 1, 6, 12, 18, 13 (total=50)
    ring_configs = [
        (0.0, 1),    # center
        (0.25, 6),   # ring 1
        (0.50, 12),  # ring 2
        (0.75, 18),  # ring 3
        (0.95, 13),  # ring 4
    ]
    
    points = []
    for ring_r, count in ring_configs:
        for k in range(count):
            angle = 2 * np.pi * k / count + np.pi / count  # offset to avoid alignment
            r = ring_r
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
    
    # Add small perturbation to break symmetry
    points = np.array(points[:n])
    points += rng.normal(0, 0.01, points.shape)
    
    return points


def _get_random_init(n, seed):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.85)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)


def _polish(centers, start_time, deadline_margin=10.0):
    """
    Progressively polish the solution with increasing beta.
    """
    current = centers.copy()
    best_score = _score(current)
    
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0]
    
    for beta in betas:
        if time.time() - start_time > _DEADLINE - deadline_margin:
            break
        
        maxiter = 100
        gtol = 1e-7
        
        polished = _optimize_lbfgs(current, beta, maxiter, gtol)
        
        if np.all(np.isfinite(polished)):
            radii = np.hypot(polished[:, 0], polished[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                polished[mask] *= (0.999 / radii[mask])[:, None]
            
            r = _score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score


def _perturb_and_polish(centers, best_score, start_time, n_perturbations=5, max_beta=50000.0):
    """
    Try small perturbations of the current solution and polish each one.
    This helps escape shallow local minima.
    """
    current = centers.copy()
    best = centers.copy()
    best_sc = best_score
    
    rng = np.random.default_rng(12345)
    
    for _ in range(n_perturbations):
        if time.time() - start_time > _DEADLINE - 15:
            break
        
        # Small random perturbation
        perturbed = current + rng.normal(0, 0.005, current.shape)
        
        # Clamp to unit disk
        radii = np.hypot(perturbed[:, 0], perturbed[:, 1])
        mask = radii > 0.99
        if np.any(mask):
            perturbed[mask] *= (0.99 / radii[mask])[:, None]
        
        # Quick polish with moderate betas
        quick_betas = [500.0, 1000.0, 2000.0]
        for beta in quick_betas:
            if time.time() - start_time > _DEADLINE - 15:
                break
            
            polished = _optimize_lbfgs(perturbed, beta, 150, 1e-8)
            
            if np.all(np.isfinite(polished)):
                radii_p = np.hypot(polished[:, 0], polished[:, 1])
                mask_p = radii_p > 1.0
                if np.any(mask_p):
                    polished[mask_p] *= (0.999 / radii_p[mask_p])[:, None]
                
                sc = _score(polished)
                if sc > best_sc:
                    best_sc = sc
                    best = polished.copy()
                    current = polished
    
    return best, best_sc


def _deep_polish(centers, start_time):
    """
    Deep optimization with contact-graph-aware gradient and progressive refinement.
    """
    current = centers.copy()
    best_score = _score(current)
    
    # Progressive beta schedule with more iterations at high beta
    betas = [100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]
    
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        # More iterations at higher beta for precision
        if beta < 1000:
            maxiter = 200
            gtol = 1e-8
        elif beta < 10000:
            maxiter = 300
            gtol = 1e-10
        else:
            maxiter = 500
            gtol = 1e-12
        
        polished = _optimize_lbfgs(current, beta, maxiter, gtol)
        
        if np.all(np.isfinite(polished)):
            radii = np.hypot(polished[:, 0], polished[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                polished[mask] *= (0.999 / radii[mask])[:, None]
            
            score = _score(polished)
            if score > best_score:
                best_score = score
                current = polished
            else:
                # If no improvement, try with slightly perturbed current
                if time.time() - start_time < _DEADLINE - 10:
                    rng = np.random.default_rng(int(beta))
                    perturbed = current + rng.normal(0, 0.002, current.shape)
                    radii_p = np.hypot(perturbed[:, 0], perturbed[:, 1])
                    mask_p = radii_p > 0.999
                    if np.any(mask_p):
                        perturbed[mask_p] *= (0.999 / radii_p[mask_p])[:, None]
                    polished2 = _optimize_lbfgs(perturbed, beta, maxiter, gtol)
                    if np.all(np.isfinite(polished2)):
                        radii2 = np.hypot(polished2[:, 0], polished2[:, 1])
                        mask2 = radii2 > 1.0
                        if np.any(mask2):
                            polished2[mask2] *= (0.999 / radii2[mask2])[:, None]
                        score2 = _score(polished2)
                        if score2 > best_score:
                            best_score = score2
                            current = polished2
    
    return current, best_score


def _final_finetune(centers, start_time):
    """
    Final fine-tuning: identify contact graph, then do exact local solve.
    """
    current = centers.copy()
    best_score = _score(current)
    
    # Step 1: Very high beta to identify critical contacts
    beta_high = 50000.0
    polished = _optimize_lbfgs(current, beta_high, 500, 1e-12)
    
    if np.all(np.isfinite(polished)):
        radii = np.hypot(polished[:, 0], polished[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            polished[mask] *= (0.999 / radii[mask])[:, None]
        
        score = _score(polished)
        if score > best_score:
            best_score = score
            current = polished
    
    # Step 2: If time allows, do multiple restarts from current with tiny perturbations
    if time.time() - start_time < _DEADLINE - 3:
        rng = np.random.default_rng(999)
        for attempt in range(3):
            if time.time() - start_time > _DEADLINE - 3:
                break
            
            perturbed = current + rng.normal(0, 0.001, current.shape)
            radii_p = np.hypot(perturbed[:, 0], perturbed[:, 1])
            mask_p = radii_p > 0.9999
            if np.any(mask_p):
                perturbed[mask_p] *= (0.9999 / radii_p[mask_p])[:, None]
            
            polished = _optimize_lbfgs(perturbed, 100000.0, 300, 1e-12)
            
            if np.all(np.isfinite(polished)):
                radii2 = np.hypot(polished[:, 0], polished[:, 1])
                mask2 = radii2 > 1.0
                if np.any(mask2):
                    polished[mask2] *= (0.9999 / radii2[mask2])[:, None]
                
                score2 = _score(polished)
                if score2 > best_score:
                    best_score = score2
                    current = polished
    
    return current, best_score


def construct():
    """
    Construct positions for N circles in the unit disk.
    Returns an array of shape (N, 2).
    """
    start_time = time.time()
    
    # Generate multiple diverse initializations
    inits = []
    
    # Hex lattice
    inits.append(_get_hex_lattice(N, seed_offset=0))
    
    # Ring init
    inits.append(_get_ring_init(N, seed=42))
    
    # Random inits
    for seed in [1, 2, 3, 7]:
        inits.append(_get_random_init(N, seed))
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Quick screening pass on all inits
    # Use low beta and few iterations to quickly identify the best candidate
    quick_betas = [50.0, 100.0, 200.0]
    
    for init in inits:
        if time.time() - start_time > _DEADLINE - 40:
            break
        
        current = init.copy()
        quick_score = _score(current)
        
        for beta in quick_betas:
            if time.time() - start_time > _DEADLINE - 40:
                break
            
            polished = _optimize_lbfgs(current, beta, 80, 1e-6)
            
            if np.all(np.isfinite(polished)):
                radii = np.hypot(polished[:, 0], polished[:, 1])
                mask = radii > 1.0
                if np.any(mask):
                    polished[mask] *= (0.999 / radii[mask])[:, None]
                
                r = _score(polished)
                if r > quick_score:
                    quick_score = r
                    current = polished
        
        if quick_score > best_score:
            best_score = quick_score
            best_pts = current
    
    if best_pts is None:
        best_pts = inits[0]
        best_score = _score(best_pts)
    
    # Phase 2: Deep optimization on the best candidate
    if time.time() - start_time < _DEADLINE - 15:
        best_pts, best_score = _deep_polish(best_pts, start_time)
    
    # Phase 3: Perturbation-based local search to escape local minima
    if time.time() - start_time < _DEADLINE - 8:
        best_pts, best_score = _perturb_and_polish(best_pts, best_score, start_time, n_perturbations=8)
    
    # Phase 4: Final fine-tuning with contact graph awareness
    if time.time() - start_time < _DEADLINE - 2:
        best_pts, best_score = _final_finetune(best_pts, start_time)
    
    # Final validation and correction: ensure all circles are strictly inside
    # Use a small safety margin to avoid boundary violations in double precision
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    # The actual radius is min(pairwise/2, 1-|c|), so we need |c| + r <= 1
    # Use a tiny epsilon to stay safely inside
    max_allowed_radius = 1.0 - best_score - 1e-15
    
    mask = radii > max_allowed_radius
    if np.any(mask):
        best_pts[mask] *= (max_allowed_radius / radii[mask])[:, None]
    
    # Re-score after correction to ensure validity
    final_score = _score(best_pts)
    if final_score <= 0:
        # Fallback: use a simple ring configuration
        best_pts = _get_ring_init(N, seed=42)
    
    return best_pts

# EVOLVE-BLOCK-END
