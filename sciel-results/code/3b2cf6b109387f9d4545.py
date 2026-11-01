"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    # Efficient pairwise distance computation
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    return min(r_cont, r_pair)


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = 0.0
    # Try a range of spacings to find the one that fits N=50 best
    for spacing in np.linspace(0.15, 0.35, 80):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points that can fit a circle of radius spacing/2 inside unit circle
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Select the N closest to the center to form a compact core
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    if best_pts is not None:
        return best_pts
    # Fallback: random points in unit disk
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _soft_min_objective(x, beta):
    """
    Objective function for progressive polishing.
    Minimizes the negative soft-min of distances.
    Soft-min approximates the true min, allowing gradient-based methods to work.
    """
    pts = x.reshape(N, 2)
    
    # Boundary constraint: penalty if outside
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists_origin > 1.0):
        # Heavy penalty for being outside
        penalty = np.sum((dists_origin[dists_origin > 1.0] - 1.0) ** 2) * 100.0
        return 10.0 + penalty
    
    r_cont = 1.0 - np.max(dists_origin)
    
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    
    # We want to maximize the minimum distance.
    # Let d_min be the min pairwise distance. We want to maximize d_min/2.
    # Soft min: -beta * log(sum(exp(-d/beta))) approximates min(d) for small beta.
    # We apply this to the pairwise distances.
    # To avoid numerical issues with large N, we use a shifted log-sum-exp.
    
    # Only consider finite distances (off-diagonal)
    finite_dists = dist_mat[np.isfinite(dist_mat)]
    
    if len(finite_dists) == 0:
        return -r_cont
        
    # Scale distances to be positive and reasonable for exp
    # We are minimizing -soft_min. So we want soft_min to be large.
    # soft_min = -beta * log(sum(exp(-d/beta)))
    # Let's compute log-sum-exp of -d/beta
    val = -finite_dists / beta
    max_val = np.max(val)
    log_sum_exp = max_val + np.log(np.sum(np.exp(val - max_val)))
    soft_min_val = -beta * log_sum_exp
    
    # The effective radius is the min of boundary constraint and half the soft min distance
    r_eff = min(r_cont, 0.5 * soft_min_val)
    
    return -r_eff


def _progressive_polish(pts0, time_limit):
    """
    Apply progressive beta polishing.
    Start with high beta (smooth landscape) and decrease to low beta (sharp min).
    """
    pts = pts0.copy()
    
    # Define beta schedule
    betas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    # Allocate time roughly equally, but leave some for final exact polish
    steps_per_beta = 200
    
    for beta in betas:
        if time.time() > time_limit:
            break
            
        def obj(x):
            return _soft_min_objective(x, beta)
            
        try:
            res = minimize(
                obj,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': steps_per_beta, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_pts = res.x.reshape(N, 2)
            
            # Project back to unit disk if needed (L-BFGS-B doesn't strictly enforce custom bounds like r<=1 easily without transformation, 
            # but we can project. Actually, the penalty in objective handles it, but projection is safer for validity)
            radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / radii[mask, np.newaxis]
                
            # Only accept if valid and score improved (or comparable, to avoid getting stuck in invalid states)
            old_score = _score(pts)
            new_score = _score(new_pts)
            
            if new_score >= old_score:
                pts = new_pts
            else:
                # If it got worse, keep old, but maybe try a smaller step? 
                # For now, just keep old.
                pass
                
        except Exception:
            pass
            
    return pts


def _final_polish(pts0, time_limit):
    """
    Final high-precision polish using Nelder-Mead on the exact objective.
    """
    def exact_obj(x):
        pts = x.reshape(N, 2)
        dists = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(dists > 1.0):
            return 1e6 + np.sum(dists[dists > 1.0] - 1.0)
        r_cont = 1.0 - np.max(dists)
        if N < 2:
            return -r_cont
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        dist_mat = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist_mat, np.inf)
        r_pair = 0.5 * np.min(dist_mat)
        return -min(r_cont, r_pair)
    
    def callback_polish(xk):
        return time.time() > time_limit
    
    try:
        res = minimize(
            exact_obj,
            pts0.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            callback=callback_polish
        )
        best_pts = res.x.reshape(N, 2)
        
        # Project back to unit disk
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            best_pts[mask] = best_pts[mask] / radii[mask, np.newaxis]
        
        return best_pts
    except Exception:
        return pts0


def _repulsion_optimize(pts0, max_time=30.0):
    """
    Fast vectorized repulsion optimization to get out of bad local minima.
    Used as a coarse global optimizer before polishing.
    """
    pts = pts0.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    step_size = 0.05
    momentum = 0.8
    v = np.zeros_like(pts)
    stuck_count = 0
    
    start_time = time.time()
    
    while time.time() < start_time + max_time and time.time() < _DEADLINE - 20.0:
        # Compute Forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        dists = np.linalg.norm(diff, axis=-1)  # (N, N)
        
        np.fill_diagonal(dists, 1.0)
        
        # 1/d^2 repulsion, capped to prevent explosion
        weights = 1.0 / (dists * dists + 1e-6)
        # Cap weights to avoid huge forces
        weights = np.minimum(weights, 100.0)
        np.fill_diagonal(weights, 0.0)
        
        forces = np.sum(diff * weights[..., np.newaxis], axis=1)
        
        # Boundary force: push in if too close to edge
        radii = np.linalg.norm(pts, axis=1)
        # Target radius is roughly 1 - current_min_dist/2. 
        # Simpler: just ensure they don't go out. 
        # Add a gentle outward force if they are in the center to spread them?
        # Actually, repulsion handles spreading. Boundary just needs to be a wall.
        
        # Update Velocity and Position
        v = momentum * v + (1 - momentum) * forces
        # Normalize velocity to prevent huge jumps
        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        v_norms[v_norms == 0] = 1.0
        v = v / v_norms
        
        new_pts = pts + step_size * v
        
        # Project to Unit Disk
        new_radii = np.linalg.norm(new_pts, axis=1)
        mask = new_radii > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / new_radii[mask, np.newaxis]
            
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            stuck_count = 0
            # Increase step slightly if improving
            step_size = min(0.1, step_size * 1.02)
        else:
            stuck_count += 1
            step_size *= 0.95
            if stuck_count > 20:
                # Random kick
                pts = best_pts + np.random.normal(0, 0.05, (N, 2))
                radii = np.linalg.norm(pts, axis=1)
                mask = radii > 1.0
                if np.any(mask):
                    pts[mask] = pts[mask] / radii[mask, np.newaxis]
                v = np.zeros_like(pts)
                stuck_count = 0
    
    return best_pts


def construct():
    start_time = time.time()
    
    # 1. Initialize with hexagonal lattice
    pts0 = _hex_lattice_init()
    best_pts = pts0
    best_score = _score(pts0)
    
    # 2. Coarse optimization with repulsion to escape initial lattice structure
    # This is fast and helps find a better local minimum
    if time.time() < _DEADLINE - 60.0:
        try:
            rep_pts = _repulsion_optimize(best_pts, max_time=30.0)
            rep_score = _score(rep_pts)
            if rep_score > best_score:
                best_pts = rep_pts
                best_score = rep_score
        except Exception:
            pass
    
    # 3. Progressive Beta Polishing
    # This is the key improvement from "Knowledge from this run"
    # We allocate most of the remaining time to this
    polish_deadline = _DEADLINE - 10.0
    if time.time() < polish_deadline:
        try:
            polished_pts = _progressive_polish(best_pts, polish_deadline)
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
    
    # 4. Final Nelder-Mead Polish for precision
    if time.time() < _DEADLINE - 2.0:
        try:
            final_pts = _final_polish(best_pts, _DEADLINE - 2.0)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_pts = final_pts
                best_score = final_score
        except Exception:
            pass
    
    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        return pts0
    
    return best_pts

# EVOLVE-BLOCK-END
