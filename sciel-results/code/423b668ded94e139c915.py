"""Logarithmic energy minimization of 150 points on the unit sphere.
Uses a multi-start strategy with Velocity-Verlet (with annealing), Adam, and L-BFGS-B polishing."""

import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its Riemannian gradient for points on the unit sphere.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Euclidean Gradient: g_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    Riemannian Gradient: grad_i = g_i - (g_i . x_i) x_i
    """
    n = pts.shape[0]
    # Compute pairwise distances squared
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist2, 1.0)
    
    # Compute energy
    # We only need the upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())
    
    # Compute Euclidean gradient
    # g_i = sum_j (x_j - x_i) / dist_ij^2
    # g = inv_dist2 @ pts - pts * row_sums
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    row_sums = np.sum(inv_dist2, axis=1)
    grad_euclid = inv_dist2 @ pts
    grad_euclid -= pts * row_sums[:, np.newaxis]
    
    # Project to tangent space to get Riemannian gradient
    dot = np.sum(grad_euclid * pts, axis=1, keepdims=True)
    grad = grad_euclid - dot * pts
    
    return energy, grad


def _energy_only(pts):
    """Compute only the energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _exp_map(pts, tangents):
    """
    Riemannian exponential map on the sphere.
    Given points p and tangent vectors v, computes exp_p(v).
    Formula: exp_p(v) = p * cos(||v||) + (v / ||v||) * sin(||v||)
    """
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    # Handle zero norms safely
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    cos_term = np.cos(norms)
    sin_term = np.sin(norms) / safe_norms
    # For very small norms, sin(norms)/norms -> 1, cos(norms) -> 1
    # The formula p*cos + v/sin_term works well.
    # Actually, for v=0, exp_p(0)=p.
    # p * cos(0) + 0/1 * sin(0) = p * 1 + 0 = p. Correct.
    
    return pts * cos_term + tangents * sin_term


def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    """Generate points using a Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _golden_spiral(n, seed=0):
    """
    Generate points using a golden-ratio spiral with slight random perturbation.
    """
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = golden_angle * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts += rng.randn(n, 3) * 0.01
    return _project_to_sphere(pts)


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron.
    """
    rng = np.random.RandomState(seed)
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    edges = [(0,11),(0,5),(0,1),(0,7),(0,10),(1,5),(5,11),(11,10),(10,7),
             (7,1),(1,6),(6,10),(2,11),(2,5),(2,9),(2,4),(2,8),(3,5),
             (3,11),(3,4),(3,9),(3,6),(4,9),(5,4),(11,4),(10,7),(6,1),
             (8,4),(9,8),(8,5),(6,9),(7,2),(8,3)]
    
    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])
    
    if len(verts) > n:
        idx = rng.choice(len(verts), size=n, replace=False)
        verts = verts[idx]
    
    return verts


def _random_points(n, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _riemannian_adam(pts0, max_iter=2000, lr_start=0.01, lr_end=0.0001,
                     beta1=0.9, beta2=0.999, eps=1e-8, deadline=None, check_interval=10):
    """
    Riemannian Adam optimization on the sphere.
    Uses exponential map for updates.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    
    # Cosine decay for learning rate
    def lr_schedule(t):
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t / max_iter))

    for t in range(1, max_iter + 1):
        if deadline is not None and (t % check_interval == 0 or t == 1):
            if time.time() > deadline:
                break

        energy, grad = _energy_and_grad(pts)
        
        # Adam update on tangent space
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        lr = lr_schedule(t)
        
        # Tangent vector for update
        tangent_update = -lr * update
        
        # Exponential map update
        pts_new = _exp_map(pts, tangent_update)
        # Ensure projection due to numerical errors
        pts_new = _project_to_sphere(pts_new)

        if not np.all(np.isfinite(pts_new)):
            break

        energy_new = _energy_only(pts_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
            
        pts = pts_new

    return best_pts, best_energy


def _lbfgs_polish(pts, deadline, maxiter=500):
    """
    L-BFGS-B polish using flat Cartesian coordinates with projection.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 30,
                                'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        if e_check < _energy_only(pts):
            return p_res
    except Exception:
        pass
    return pts


def _lbfgs_tight_polish(pts, deadline, maxiter=1000):
    """
    Ultra-tight L-BFGS-B polish.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 50,
                                'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 60})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        if e_check < _energy_only(pts):
            return p_res
    except Exception:
        pass
    return pts


def _local_perturb_restart(pts, rng, perturb_scale=0.05, n_perturb=10):
    """
    Create perturbed versions of pts by randomly moving n_perturb points.
    Returns list of perturbed point arrays.
    """
    results = []
    for _ in range(5):
        p = pts.copy()
        idx = rng.choice(N, size=n_perturb, replace=False)
        noise = rng.randn(n_perturb, 3) * perturb_scale
        p[idx] += noise
        p = _project_to_sphere(p)
        results.append(p)
    return results


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    # Define multiple initializations
    # 1. Fibonacci spiral
    init1 = _fibonacci_spiral(N)
    # 2. Fibonacci spiral with different twist
    init2 = _fibonacci_spiral(N, twist=1.5)
    # 3. Random points
    init3 = _random_points(N, seed=123)
    # 4. Random points different seed
    init4 = _random_points(N, seed=456)
    # 5. Perturbed Fibonacci
    rng = np.random.RandomState(999)
    init5 = _project_to_sphere(_fibonacci_spiral(N) + rng.randn(N, 3) * 0.1)
    # 6. Geodesic dome
    init6 = _geodesic_dome(N, seed=7)
    # 7. Golden spiral with perturbation
    init7 = _golden_spiral(N, seed=1)
    # 8. Golden spiral with different perturbation
    init8 = _golden_spiral(N, seed=2)

    inits = [init1, init2, init3, init4, init5, init6, init7, init8]
    
    best_pts_global = None
    best_energy_global = float('inf')

    # Time per restart
    # We have ~140s. Let's do 3 main restarts, each ~35s, and keep the best.
    # Then use remaining time for local perturbation restarts and final polish.
    
    time_per_restart = 35.0
    num_restarts = 3
    
    for i in range(num_restarts):
        current_deadline = start_time + (i + 1) * time_per_restart
        if current_deadline > deadline - 30.0: # Leave 30s for perturbation restarts + final polish
            current_deadline = deadline - 30.0
            
        pts_start = inits[i % len(inits)]
        
        # Phase 1: Riemannian Adam (basin finding and refinement)
        adam_deadline = current_deadline - 5.0
        if time.time() < adam_deadline:
            pts_adam, e_adam = _riemannian_adam(pts_start, max_iter=2000, lr_start=0.01, lr_end=0.0001, deadline=adam_deadline)
        else:
            pts_adam, e_adam = pts_start, _energy_only(pts_start)
            
        # Phase 2: L-BFGS-B (polish)
        if time.time() < current_deadline - 1.0:
            pts_lbfgs = _lbfgs_polish(pts_adam, current_deadline - 1.0, maxiter=300)
            e_lbfgs = _energy_only(pts_lbfgs)
        else:
            pts_lbfgs, e_lbfgs = pts_adam, e_adam
            
        if e_lbfgs < best_energy_global:
            best_energy_global = e_lbfgs
            best_pts_global = pts_lbfgs.copy()

    # Local Perturbation Restart Phase
    # Try perturbing the best solution to escape local minima
    if best_pts_global is not None and time.time() < deadline - 10.0:
        perturb_deadline = deadline - 10.0
        rng_perturb = np.random.RandomState(42)
        
        # Try several perturbations with decreasing scale
        for scale in [0.08, 0.05, 0.03]:
            if time.time() > perturb_deadline - 5.0:
                break
            
            perturbations = _local_perturb_restart(best_pts_global, rng_perturb, 
                                                   perturb_scale=scale, n_perturb=8)
            for pert_pts in perturbations:
                if time.time() > perturb_deadline - 2.0:
                    break
                
                # Quick Riemannian Adam refinement
                if time.time() < perturb_deadline - 3.0:
                    pts_ref, e_ref = _riemannian_adam(pert_pts, max_iter=500, 
                                                     lr_start=0.003, lr_end=0.0003,
                                                     deadline=perturb_deadline - 3.0)
                    # Quick L-BFGS polish
                    if time.time() < perturb_deadline - 1.0:
                        pts_pol = _lbfgs_polish(pts_ref, perturb_deadline - 1.0, maxiter=100)
                        e_pol = _energy_only(pts_pol)
                    else:
                        pts_pol, e_pol = pts_ref, e_ref
                    
                    if e_pol < best_energy_global:
                        best_energy_global = e_pol
                        best_pts_global = pts_pol.copy()

    # Final Polish Phase
    # Use remaining time to polish the global best
    remaining_deadline = deadline - 2.0
    
    if best_pts_global is None:
        best_pts_global = _fibonacci_spiral(N)
        best_energy_global = _energy_only(best_pts_global)
        
    # Tight L-BFGS with more iterations
    if time.time() < remaining_deadline - 3.0:
        best_pts_global = _lbfgs_tight_polish(best_pts_global, remaining_deadline - 3.0, maxiter=1500)
        best_energy_global = _energy_only(best_pts_global)
        
    # One more standard L-BFGS with different settings
    if time.time() < remaining_deadline - 0.5:
        best_pts_global = _lbfgs_polish(best_pts_global, remaining_deadline - 0.5, maxiter=300)
        best_energy_global = _energy_only(best_pts_global)

    best_pts_global = _project_to_sphere(best_pts_global)

    # Validation
    if not np.all(np.isfinite(best_pts_global)):
        return _fibonacci_spiral(N)

    diff = best_pts_global[:, np.newaxis, :] - best_pts_global[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_spiral(N)

    return best_pts_global

# EVOLVE-BLOCK-END
