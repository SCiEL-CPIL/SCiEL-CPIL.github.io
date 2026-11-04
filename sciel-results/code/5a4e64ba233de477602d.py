"""Logarithmic energy minimization of 150 points on the unit sphere.
Uses a multi-start strategy with Velocity-Verlet, Adam, L-BFGS-B polishing,
and a robust Riemannian line-search phase to escape shallow local minima."""

import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    The gradient is projected to the tangent space of the sphere.
    """
    n = pts.shape[0]
    # Compute pairwise distances squared
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist2, 1.0)
    
    # Compute energy using math.fsum for precision
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    # Filter out any zero distances (shouldn't happen but safety)
    dists_safe = dists[dists > 1e-15]
    if len(dists_safe) == 0:
        energy = 0.0
    else:
        energy = -math.fsum(np.log(dists_safe).tolist())
    
    # Compute gradient
    # grad_i = sum_j (x_j - x_i) / dist_ij^2
    # Note: 1/dist2 is symmetric
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad = inv_dist2 @ pts - pts * row_sums
    # row_sums[i] = sum_j inv_dist2[i,j]
    row_sums = np.sum(inv_dist2, axis=1)
    grad = inv_dist2 @ pts
    grad -= pts * row_sums[:, np.newaxis]
    
    # Project gradient to tangent space
    # tangent_grad = grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _energy_only(pts):
    """Compute only the energy with high precision."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    dists_safe = dists[dists > 1e-15]
    if len(dists_safe) == 0:
        return 0.0
    return -math.fsum(np.log(dists_safe).tolist())


def _energy_only_fast(pts):
    """Compute energy quickly (for inner loops where precision matters less)."""
    n = len(pts)
    dist = cdist(pts, pts, 'euclidean')
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = dist_upper[dist_upper > 1e-15]
    if len(dist_upper) == 0:
        return 0.0
    return -np.sum(np.log(dist_upper))


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    """Generate points using a Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron (geodesic dome),
    then project to sphere. Provides a different structure than Fibonacci spiral.
    """
    rng = np.random.RandomState(seed)
    # Start with icosahedron vertices
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Edges of icosahedron
    edges = [(0,11),(0,5),(0,1),(0,7),(0,10),(1,5),(5,11),(11,10),(10,7),
             (7,1),(1,6),(6,10),(2,11),(2,5),(2,9),(2,4),(2,8),(3,5),
             (3,11),(3,4),(3,9),(3,6),(4,9),(5,4),(11,4),(10,7),(6,1),
             (8,4),(9,8),(8,5),(6,9),(7,2),(8,3)]
    
    # Recursively subdivide until we have enough points
    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])
    
    # If we have more than n, randomly select n
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


def _icosahedron_based(n):
    """Generate points using icosahedron subdivision for better initial distribution."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    fib = _fibonacci_spiral(n)
    result = fib.copy()
    for i in range(12):
        dists = np.linalg.norm(fib - ico[i], axis=1)
        closest = np.argmin(dists)
        result[closest] = ico[i]
    return _project_to_sphere(result)


def _adam_optimize(pts0, max_iter=3000, lr_start=0.01, lr_end=0.001,
                   beta1=0.9, beta2=0.999, eps=1e-8, deadline=None, check_interval=20):
    """
    Adam optimization on the sphere with learning rate decay and adaptive step.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    
    # Cosine decay for learning rate
    def lr_schedule(t):
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t / max_iter))

    # Adaptive learning rate multiplier
    adaptive_lr_mult = 1.0

    for t in range(1, max_iter + 1):
        if deadline is not None and (t % check_interval == 0 or t == 1):
            if time.time() > deadline:
                break

        energy, grad = _energy_and_grad(pts)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        lr = lr_schedule(t) * adaptive_lr_mult
        
        # Step in tangent direction
        pts_new = pts - lr * update
        pts_new = _project_to_sphere(pts_new)

        if not np.all(np.isfinite(pts_new)):
            adaptive_lr_mult *= 0.5
            continue

        energy_new = _energy_only(pts_new)
        
        # Adaptive learning rate adjustment
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
            # Increase adaptive multiplier if we're making good progress
            if t > 100:
                adaptive_lr_mult = min(adaptive_lr_mult * 1.1, 3.0)
        else:
            # Decrease adaptive multiplier if we're not improving
            adaptive_lr_mult = max(adaptive_lr_mult * 0.8, 0.1)
            
        pts = pts_new

    return best_pts, best_energy


def _velocity_verlet_sphere(points, deadline, dt=0.005, n_steps=5000, damping=0.98):
    """
    Velocity-Verlet integration on the sphere.
    """
    pts = _project_to_sphere(points.copy())
    vel = np.zeros_like(pts)
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Initial force
    _, grad0 = _energy_and_grad(pts)
    force = -grad0
    
    for step in range(n_steps):
        if time.time() > deadline - 0.5:
            break
        
        # Velocity-Verlet half step
        vel_half = vel + 0.5 * dt * force
        # Project velocity to tangent space
        dot_v = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half = vel_half - dot_v * pts
        
        # Position update
        pts_new = pts + dt * vel_half
        pts_new = _project_to_sphere(pts_new)
        
        # New force
        _, grad_new = _energy_and_grad(pts_new)
        force_new = -grad_new
        # Project force to tangent space
        dot_f = np.sum(force_new * pts_new, axis=1, keepdims=True)
        force_new = force_new - dot_f * pts_new
        
        # Velocity update with damping
        vel = (vel_half + 0.5 * dt * force_new) * damping
        dot_v2 = np.sum(vel * pts_new, axis=1, keepdims=True)
        vel = vel - dot_v2 * pts_new
        
        pts = pts_new
        force = force_new
        
        energy = _energy_only(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Adaptive damping or step size if energy increases
        if step > 100 and energy > best_energy * 1.05:
            dt *= 0.9
            damping *= 0.99

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


def _riemannian_line_search(pts, deadline, max_iter=2000, initial_step=0.05,
                            armijo_c=1e-4, step_decay=0.5, max_backtrack=25):
    """
    Robust Riemannian line search with Armijo condition.
    This allows larger effective steps and better escape from shallow local minima
    compared to fixed-step methods.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Compute initial gradient
    _, grad = _energy_and_grad(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Backtracking line search with Armijo condition
        step = initial_step
        accepted = False
        for bt in range(max_backtrack):
            # Move in negative gradient direction (tangent space)
            pts_new = pts - step * grad
            pts_new = _project_to_sphere(pts_new)
            
            if not np.all(np.isfinite(pts_new)):
                step *= step_decay
                continue
            
            energy_new = _energy_only(pts_new)
            
            # Armijo condition
            if energy_new <= best_energy - armijo_c * step * grad_norm * grad_norm:
                accepted = True
                break
            
            step *= step_decay
        
        if accepted:
            pts = pts_new
            if energy_new < best_energy:
                best_energy = energy_new
                best_pts = pts.copy()
            
            # Recompute gradient
            _, grad = _energy_and_grad(pts)
            
            # Gradually reduce step size as we converge
            if it > 100:
                initial_step *= 0.995
                initial_step = max(initial_step, 0.001)
        else:
            # If no step was accepted, try a smaller step
            step = initial_step * 0.1
            pts_new = pts - step * grad
            pts_new = _project_to_sphere(pts_new)
            if np.all(np.isfinite(pts_new)):
                energy_new = _energy_only(pts_new)
                if energy_new < best_energy:
                    best_energy = energy_new
                    best_pts = pts_new.copy()
                    pts = pts_new
                    _, grad = _energy_and_grad(pts)
                initial_step *= 0.8
            else:
                break
    
    return best_pts, best_energy


def _final_grad_polish(pts, max_iter=5000, tol=1e-15, deadline=None):
    """
    High-precision gradient descent with backtracking line search on the sphere.
    Uses exact tangent-space projection and adaptive step sizes.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Compute initial gradient
    _, grad = _energy_and_grad(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        step = 0.1
        for _ in range(20):
            # Move in negative gradient direction (tangent space)
            pts_new = pts - step * grad
            pts_new = _project_to_sphere(pts_new)
            
            if not np.all(np.isfinite(pts_new)):
                step *= 0.5
                continue
            
            energy_new = _energy_only(pts_new)
            
            if energy_new < best_energy:
                best_energy = energy_new
                best_pts = pts_new.copy()
            
            # Armijo condition
            if energy_new <= best_energy - 1e-4 * step * grad_norm * grad_norm:
                break
            
            step *= 0.5
        
        # Update
        pts = _project_to_sphere(pts - step * grad)
        
        # Recompute gradient
        _, grad = _energy_and_grad(pts)
        
        # Project gradient to tangent space (already done in _energy_and_grad)
        
    return best_pts, best_energy


def _simulated_annealing(pts, time_limit=10.0, seed=42):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy = _energy_only(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    max_steps = 5000

    rng = np.random.RandomState(seed)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        noise_scale = T * 0.1
        perturbation = rng.randn(n, 3) * noise_scale
        new_pts = pts + perturbation
        new_pts = _project_to_sphere(new_pts)

        new_energy = _energy_only_fast(new_pts)

        if new_energy < current_energy or rng.rand() < math.exp(-(new_energy - current_energy) / T):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        T *= cooling
        if T < T_min:
            break

    return best_pts, best_energy


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
    # 7. Golden spiral (different from Fibonacci)
    def _golden_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = 2.399963229728653 * i  # Golden angle in radians
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    init7 = _golden_spiral(N)
    # 8. Icosahedron-based
    init8 = _icosahedron_based(N)

    inits = [init1, init2, init3, init4, init5, init6, init7, init8]
    
    best_pts_global = None
    best_energy_global = float('inf')

    # Time allocation:
    # - 4 main restarts, each ~25s
    # - Simulated annealing on best: ~10s
    # - Local perturbation restarts: ~20s
    # - Final polish: ~25s
    # Total: ~100s, leaving buffer
    
    time_per_restart = 25.0
    num_restarts = 4
    
    for i in range(num_restarts):
        current_deadline = start_time + (i + 1) * time_per_restart
        if current_deadline > deadline - 55.0:  # Leave 55s for SA + perturb + polish
            current_deadline = deadline - 55.0
            
        pts_start = inits[i % len(inits)]
        
        # Phase 1: Velocity Verlet (fast basin finding)
        vv_deadline = current_deadline - 10.0
        if time.time() < vv_deadline:
            pts_vv, e_vv = _velocity_verlet_sphere(pts_start, vv_deadline, dt=0.005, n_steps=2000)
        else:
            pts_vv, e_vv = pts_start, _energy_only(pts_start)
            
        # Phase 2: Adam (refine)
        adam_deadline = current_deadline - 5.0
        if time.time() < adam_deadline:
            pts_adam, e_adam = _adam_optimize(pts_vv, max_iter=1000, lr_start=0.005, lr_end=0.0005, deadline=adam_deadline)
        else:
            pts_adam, e_adam = pts_vv, e_vv
            
        # Phase 3: Riemannian line search (robust escape from local minima)
        if time.time() < current_deadline - 2.0:
            pts_rls, e_rls = _riemannian_line_search(pts_adam, current_deadline - 2.0, max_iter=500, initial_step=0.03)
            if e_rls < e_adam:
                pts_lbfgs = _lbfgs_polish(pts_rls, current_deadline - 0.5, maxiter=200)
                e_lbfgs = _energy_only(pts_lbfgs)
            else:
                pts_lbfgs = _lbfgs_polish(pts_adam, current_deadline - 0.5, maxiter=200)
                e_lbfgs = _energy_only(pts_lbfgs)
        else:
            pts_lbfgs, e_lbfgs = pts_adam, e_adam
            
        if e_lbfgs < best_energy_global:
            best_energy_global = e_lbfgs
            best_pts_global = pts_lbfgs.copy()

    # Simulated Annealing Phase
    # Try to escape local minima using SA on the best solution
    if best_pts_global is not None and time.time() < deadline - 45.0:
        sa_deadline = time.time() + 10.0
        if sa_deadline > deadline - 45.0:
            sa_deadline = deadline - 45.0
        pts_sa, e_sa = _simulated_annealing(best_pts_global, time_limit=10.0, seed=42)
        if e_sa < best_energy_global:
            # Polish the SA result
            if time.time() < deadline - 40.0:
                pts_sa_pol = _riemannian_line_search(pts_sa, deadline - 40.0, max_iter=300, initial_step=0.02)
                e_sa_pol = _energy_only(pts_sa_pol)
                if e_sa_pol < best_energy_global:
                    best_energy_global = e_sa_pol
                    best_pts_global = pts_sa_pol.copy()
            elif time.time() < deadline - 35.0:
                pts_sa_pol = _lbfgs_polish(pts_sa, deadline - 35.0, maxiter=300)
                e_sa_pol = _energy_only(pts_sa_pol)
                if e_sa_pol < best_energy_global:
                    best_energy_global = e_sa_pol
                    best_pts_global = pts_sa_pol.copy()

    # Local Perturbation Restart Phase
    # Try perturbing the best solution to escape local minima
    if best_pts_global is not None and time.time() < deadline - 25.0:
        perturb_deadline = deadline - 25.0
        rng_perturb = np.random.RandomState(42)
        
        # Try several perturbations with varying scales and point counts
        for scale, n_pert in [(0.1, 12), (0.06, 8), (0.03, 5), (0.015, 3)]:
            if time.time() > perturb_deadline - 5.0:
                break
            
            perturbations = _local_perturb_restart(best_pts_global, rng_perturb, 
                                                   perturb_scale=scale, n_perturb=n_pert)
            for pert_pts in perturbations:
                if time.time() > perturb_deadline - 2.0:
                    break
                
                # Quick Riemannian line search refinement
                if time.time() < perturb_deadline - 3.0:
                    pts_ref, e_ref = _riemannian_line_search(pert_pts, deadline - 3.0, max_iter=300, initial_step=0.02)
                    # Quick L-BFGS polish
                    if time.time() < perturb_deadline - 1.0:
                        pts_pol = _lbfgs_polish(pts_ref, perturb_deadline - 1.0, maxiter=150)
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
    if time.time() < remaining_deadline - 10.0:
        best_pts_global = _lbfgs_tight_polish(best_pts_global, remaining_deadline - 10.0, maxiter=2000)
        best_energy_global = _energy_only(best_pts_global)
        
    # Riemannian line search polish
    if time.time() < remaining_deadline - 5.0:
        best_pts_global, _ = _riemannian_line_search(best_pts_global, max_iter=1000, 
                                                     initial_step=0.01, deadline=remaining_deadline - 5.0)
        best_energy_global = _energy_only(best_pts_global)
        
    # High-precision gradient descent polish
    if time.time() < remaining_deadline - 2.0:
        best_pts_global, _ = _final_grad_polish(best_pts_global, max_iter=3000, 
                                                 tol=1e-16, deadline=remaining_deadline - 2.0)
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
