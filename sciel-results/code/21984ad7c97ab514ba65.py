"""Improved packer: grid seed + BFGS optimization on penalty energy."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100


def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)


def _energy_and_grad(centers, target_r):
    """
    Energy function for BFGS optimization.
    Penalizes pairwise distances < 2*target_r and wall clearances < target_r.
    Returns (energy, gradient).
    """
    c = centers.reshape(-1, 3)
    
    # --- Pairwise Part ---
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)  # Avoid division by zero
    
    # Active pairs: dists < 2 * target_r
    active = dists < 2.0 * target_r
    active[np.diag_indices_from(active)] = False
    
    # Violations: (2r - d) for active, 0 otherwise
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    energy_pair = 0.5 * np.sum(violations**2)
    
    # Gradient calculation
    # safe_dists to avoid div by 0 in unit vectors
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # weights[i,j] = 2 * violations[i,j]
    weights = 2.0 * violations
    
    # grad_i = - sum_j weights[i,j] * unit_vecs[i,j]
    grad_pair = -np.sum(weights[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # --- Wall Part ---
    wall_dists = 0.5 - np.abs(c)
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    energy_wall = np.sum(wall_violations**2)
    
    # Wall gradient: 2 * (r - w) * sign(c)
    signs = np.sign(c)
    grad_wall = 2.0 * wall_violations * signs
    
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()
    
    return total_energy, total_grad


def construct():
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin under 120s

    # ---- Seed: simple cubic 5x5x5 grid ----
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()

    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)

    # Strategy: Run BFGS with increasing target radii
    # Start with a target slightly above the grid score (0.1)
    # Best known is ~0.11138, so 0.115 is a good stretch goal.
    
    targets = [0.11, 0.115, 0.118]
    
    for target_r in targets:
        if time.time() - t0 > DEADLINE:
            break
            
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.01, size=(N, 3))
        init_pts = np.clip(pts + noise, -0.49, 0.49)
        
        def objective(x):
            e, g = _energy_and_grad(x, target_r)
            return e, g
            
        try:
            result = minimize(
                objective,
                init_pts.flatten(),
                method='BFGS',
                jac=True,
                options={'maxiter': 1000, 'gtol': 1e-9, 'disp': False}
            )
            opt_pts = result.x.reshape(-1, 3)
            
            # Validate
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()
        except Exception:
            pass

    # If time permits, try a few random restarts with the best target
    if time.time() - t0 < DEADLINE - 10.0:
        for i in range(10):
            if time.time() - t0 > DEADLINE:
                break
            noise = rng.normal(0, 0.05, size=(N, 3))
            init_pts = np.clip(pts + noise, -0.49, 0.49)
            
            def objective_rand(x):
                e, g = _energy_and_grad(x, 0.115)
                return e, g
                
            try:
                result = minimize(
                    objective_rand,
                    init_pts.flatten(),
                    method='BFGS',
                    jac=True,
                    options={'maxiter': 500, 'gtol': 1e-8, 'disp': False}
                )
                opt_pts = result.x.reshape(-1, 3)
                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                    r_opt = _score(opt_pts)
                    if r_opt > best_r:
                        best_r = r_opt
                        best_pts = opt_pts.copy()
            except Exception:
                pass

    # Final safety
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_pts)):
        best_pts = grid[:N].copy()
        
    return best_pts
