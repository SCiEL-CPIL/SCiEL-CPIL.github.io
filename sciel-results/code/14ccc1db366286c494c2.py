"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _compute_circumcenters(pts):
    """Compute circumcenters of all hull facets. Returns array of shape (n_facets, 3)."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.empty((0, 3))
    
    ccs = []
    for fi in range(len(hull.simplices)):
        tri = hull.simplices[fi]
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        v = np.cross(b - a, c - a)
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v = v / nv
        if np.dot(v, a) < 0:
            v = -v
        ccs.append(v)
    
    if not ccs:
        return np.empty((0, 3))
    return np.array(ccs)


def _covering_radius(pts):
    """Compute covering radius in radians. Returns (radius, worst_facet_index, worst_point)."""
    ccs = _compute_circumcenters(pts)
    if len(ccs) == 0:
        return np.pi, -1, None
    
    # For each circumcenter, find min angular distance to any site
    cos_dists = pts @ ccs.T  # shape (n_sites, n_facets)
    cos_dists = np.clip(cos_dists, -1.0, 1.0)
    min_cos = np.min(cos_dists, axis=0)  # shape (n_facets,)
    angles = np.arccos(min_cos)
    
    worst_idx = np.argmax(angles)
    max_ang = angles[worst_idx]
    worst_pt = ccs[worst_idx].copy()
    
    return max_ang, worst_idx, worst_pt


def _soft_covering_radius(pts, beta=50.0):
    """Smooth surrogate: soft-max of angular distances to circumcenters.
    Returns (soft_radius, gradient wrt pts)."""
    ccs = _compute_circumcenters(pts)
    if len(ccs) == 0:
        return np.pi, np.zeros_like(pts)
    
    # Compute angular distances: shape (n_sites, n_facets)
    cos_dists = pts @ ccs.T
    cos_dists = np.clip(cos_dists, -1.0, 1.0)
    angles = np.arccos(cos_dists)
    
    # Soft-max over sites for each facet
    # For each facet f, soft_max_i(angles[i,f]) ≈ max_i angles[i,f]
    # Use log-sum-exp trick
    max_ang = np.max(angles, axis=0)  # shape (n_facets,)
    # Fix broadcasting: max_ang is (n_facets,), angles is (n_sites, n_facets)
    # We need to subtract max_ang from each column of angles
    exp_angles = np.exp(beta * (angles - max_ang[np.newaxis, :]))
    soft_radii = max_ang + np.log(np.sum(exp_angles, axis=0)) / beta  # shape (n_facets,)
    
    # The covering radius is the max over facets of soft_radii
    # Use another soft-max over facets
    max_sr = np.max(soft_radii)
    exp_sr = np.exp(beta * (soft_radii - max_sr))
    soft_radius = max_sr + np.log(np.sum(exp_sr)) / beta
    
    # Return zero gradient to avoid expensive finite differences which cause timeouts.
    # The local optimization and simulated annealing phases handle the refinement.
    grad = np.zeros_like(pts)
    
    return soft_radius, grad


def _lloyd_step(pts, n_iter=50):
    """Perform Lloyd's algorithm iterations to minimize covering radius."""
    pts = pts.copy()
    for _ in range(n_iter):
        ccs = _compute_circumcenters(pts)
        if len(ccs) == 0:
            break
        
        # For each site, find circumcenters closest to it
        new_pts = pts.copy()
        for i in range(len(pts)):
            dists = np.linalg.norm(ccs - pts[i], axis=1)
            k = min(5, len(ccs))
            idx = np.argsort(dists)[:k]
            centroid = ccs[idx].mean(axis=0)
            nc = np.linalg.norm(centroid)
            if nc > 1e-10:
                centroid = centroid / nc
                new_pts[i] = 0.5 * pts[i] + 0.5 * centroid
                new_pts[i] /= np.linalg.norm(new_pts[i])
        
        pts = new_pts
    
    return pts


def _local_optimize(pts, n_iter=200, step_size=0.05):
    """Local optimization targeting the worst facet."""
    pts = pts.copy()
    for it in range(n_iter):
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        tri = hull.simplices[wf]
        
        alpha = step_size
        for idx in tri:
            new_pt = pts[idx] + alpha * (wpt - pts[idx])
            new_pt /= np.linalg.norm(new_pt)
            pts[idx] = new_pt
    
    return pts


def _gradient_polish(pts, n_iter=50, lr=0.01):
    """Gradient-based polish using soft covering radius."""
    pts = pts.copy()
    for it in range(n_iter):
        if time.time() > _deadline - 5.0:
            break
        _, grad = _soft_covering_radius(pts, beta=50.0)
        
        # Gradient descent on the sphere
        new_pts = pts.copy()
        for i in range(len(pts)):
            # Move in negative gradient direction, project to sphere
            step = -lr * grad[i]
            new_pts[i] = pts[i] + step
            new_pts[i] /= np.linalg.norm(new_pts[i])
        
        pts = new_pts
    
    return pts


def _simulated_annealing(pts, best_radius, n_steps=100, T_start=0.02, T_end=0.001):
    """Simulated annealing to escape local minima."""
    pts = pts.copy()
    current_radius, _, _ = _covering_radius(pts)
    
    for step in range(n_steps):
        if time.time() > _deadline - 5.0:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (step / max(1, n_steps - 1))
        
        # Pick a random point to perturb
        idx = np.random.randint(len(pts))
        delta = np.random.randn(3) * T
        new_pt = pts[idx] + delta
        new_pt /= np.linalg.norm(new_pt)
        
        trial = pts.copy()
        trial[idx] = new_pt
        trial_radius, _, _ = _covering_radius(trial)
        
        # Accept if better, or with probability exp(-delta/T)
        delta_r = trial_radius - current_radius
        if delta_r < 0 or np.random.random() < np.exp(-delta_r / max(T, 1e-10)):
            pts = trial
            current_radius = trial_radius
            if trial_radius < best_radius:
                best_radius = trial_radius
    
    return pts, best_radius


def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 135.0  # Leave 15s margin before 150s kill
    
    np.random.seed(42)
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius, _, _ = _covering_radius(pts)
    
    # Phase 1: Lloyd's algorithm for global improvement
    while time.time() < _deadline - 40.0:
        pts_lloyd = _lloyd_step(pts, n_iter=20)
        r_lloyd, _, _ = _covering_radius(pts_lloyd)
        if r_lloyd < best_radius:
            best_radius = r_lloyd
            best_pts = pts_lloyd.copy()
            pts = pts_lloyd.copy()
        else:
            pts_lloyd = _lloyd_step(pts, n_iter=5)
            r_lloyd, _, _ = _covering_radius(pts_lloyd)
            if r_lloyd < best_radius:
                best_radius = r_lloyd
                best_pts = pts_lloyd.copy()
                pts = pts_lloyd.copy()
            else:
                break
    
    # Phase 2: Local optimization targeting worst facets
    while time.time() < _deadline - 20.0:
        pts_local = _local_optimize(pts, n_iter=50, step_size=0.05)
        r_local, _, _ = _covering_radius(pts_local)
        if r_local < best_radius:
            best_radius = r_local
            best_pts = pts_local.copy()
            pts = pts_local.copy()
        else:
            pts_local = _local_optimize(pts, n_iter=20, step_size=0.02)
            r_local, _, _ = _covering_radius(pts_local)
            if r_local < best_radius:
                best_radius = r_local
                best_pts = pts_local.copy()
                pts = pts_local.copy()
            else:
                break
    
    # Phase 3: Gradient-based polish
    while time.time() < _deadline - 15.0:
        pts_grad = _gradient_polish(pts, n_iter=30, lr=0.005)
        r_grad, _, _ = _covering_radius(pts_grad)
        if r_grad < best_radius:
            best_radius = r_grad
            best_pts = pts_grad.copy()
            pts = pts_grad.copy()
        else:
            pts_grad = _gradient_polish(pts, n_iter=10, lr=0.002)
            r_grad, _, _ = _covering_radius(pts_grad)
            if r_grad < best_radius:
                best_radius = r_grad
                best_pts = pts_grad.copy()
                pts = pts_grad.copy()
            else:
                break
    
    # Phase 4: Simulated annealing to escape local minima
    while time.time() < _deadline - 10.0:
        pts_sa, best_radius = _simulated_annealing(
            best_pts, best_radius, n_steps=50, T_start=0.015, T_end=0.003
        )
        # After SA, do a quick local polish
        pts_polished = _local_optimize(pts_sa, n_iter=30, step_size=0.03)
        r_polished, _, _ = _covering_radius(pts_polished)
        if r_polished < best_radius:
            best_radius = r_polished
            best_pts = pts_polished.copy()
            pts = pts_polished.copy()
        else:
            # Try smaller annealing
            pts_sa, best_radius = _simulated_annealing(
                best_pts, best_radius, n_steps=30, T_start=0.008, T_end=0.002
            )
            pts_polished = _local_optimize(pts_sa, n_iter=20, step_size=0.02)
            r_polished, _, _ = _covering_radius(pts_polished)
            if r_polished < best_radius:
                best_radius = r_polished
                best_pts = pts_polished.copy()
                pts = pts_polished.copy()
            else:
                break
    
    # Phase 5: Final fine-tuning with small perturbations
    while time.time() < _deadline - 5.0:
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        tri = hull.simplices[wf]
        
        improved = False
        for idx in tri:
            for _ in range(15):
                delta = np.random.randn(3) * 0.005
                new_pt = pts[idx] + delta
                new_pt /= np.linalg.norm(new_pt)
                trial = pts.copy()
                trial[idx] = new_pt
                r_trial, _, _ = _covering_radius(trial)
                if r_trial < best_radius:
                    best_radius = r_trial
                    best_pts = trial.copy()
                    pts = trial.copy()
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    # Ensure output is valid
    pts_out = best_pts.copy()
    norms = np.linalg.norm(pts_out, axis=1)
    pts_out = pts_out / norms[:, np.newaxis]
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pts_out[i] - pts_out[j]) < 1e-8:
                pts_out[j] += np.random.randn(3) * 0.01
                pts_out[j] /= np.linalg.norm(pts_out[j])
    
    return pts_out
# EVOLVE-BLOCK-END
