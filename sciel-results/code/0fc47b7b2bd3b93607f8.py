"""Logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill


def _remaining():
    return _DEADLINE - time.time()


def _hessian_energy(pts):
    """
    Compute the Hessian of the logarithmic energy on the tangent space.
    For f = -sum_{i<j} log(d_ij), where d_ij = ||x_i - x_j||
    
    The Hessian on R^3 projected to tangent space:
    H_ij = (1/d_ij^2) * (I - u_ij u_ij^T) for i != j, where u_ij = (x_i - x_j)/d_ij
    And diagonal terms accumulate contributions from all other points.
    
    We compute the full 3N x 3N Hessian projected to tangent space.
    """
    n = pts.shape[0]
    pts = _normalize(pts)
    
    # Compute all pairwise distances and unit vectors
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    
    # Unit vectors u_ij = (x_i - x_j) / d_ij
    # u[i,j] = (pts[i] - pts[j]) / dist[i,j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    u = diff / dist[:, :, np.newaxis]  # (n, n, 3)
    
    # Hessian block: for i != j, H_ij = (1/d_ij^2) * (I - u_ij u_ij^T)
    # Diagonal: H_ii = sum_{j != i} (1/d_ij^2) * (u_ij u_ij^T - I) ... wait, let me reconsider
    
    # Energy = -sum_{i<j} log(d_ij)
    # grad_i = sum_{j != i} (x_i - x_j) / d_ij^2  ... no
    # Let me recompute: f = -sum_{i<j} log(||x_i - x_j||)
    # df/dx_i = -sum_{j != i} (1/d_ij) * (x_i - x_j)/d_ij = -sum_{j != i} (x_i - x_j)/d_ij^2
    # Hmm, but our gradient was: grad_i = -sum_j W_ij * x_i + sum_j W_ij * x_j where W_ij = 1/d_ij^2
    # = sum_j W_ij (x_j - x_i) = -sum_j W_ij (x_i - x_j)
    # So grad_i = -sum_{j!=i} (x_i - x_j)/d_ij^2. That matches.
    
    # Hessian: d^2f/dx_i dx_j for i != j:
    # d/dx_j [-sum_k W_ik (x_i - x_k)] where only W_ij depends on x_j
    # = -dW_ij/dx_j * (x_i - x_j)
    # W_ij = 1/||x_i - x_j||^2, dW_ij/dx_j = 2(x_i - x_j)/||x_i - x_j||^4 = 2(x_i-x_j)/d^4
    # So H_ij = -2(x_i-x_j)/d^4 * (x_i - x_j)^T = -2 u_ij u_ij^T / d_ij^2
    # Wait, that gives negative definite off-diagonal blocks? Let me check.
    
    # Actually: f = -sum log(d_ij), so Hessian should be positive semi-definite at minimum.
    # Let me redo: 
    # grad_i = -sum_{j!=i} (x_i - x_j)/d_ij^2
    # d(grad_i)/d(x_j) for j != i: only term j contributes
    # d/dx_j [-(x_i - x_j)/d_ij^2]
    # = -[(-I)/d_ij^2 + (x_i - x_j) * d(1/d_ij^2)/d(x_j)]
    # d(1/d^2)/dx_j = d(d^-2)/dx_j = -2 d^-3 * d(d)/dx_j = -2 d^-3 * (x_j - x_i)/d = 2(x_i-x_j)/d^4
    # So: -[(-I)/d^2 + (x_i-x_j) * 2(x_i-x_j)^T / d^4]
    # = I/d^2 - 2(x_i-x_j)(x_i-x_j)^T/d^4
    # = (1/d^2)[I - 2 u_ij u_ij^T]
    
    # Diagonal H_ii: d(grad_i)/d(x_i)
    # grad_i = -sum_{j!=i} (x_i - x_j)/d_ij^2
    # d/dx_i [-(x_i-x_j)/d_ij^2] = -[I/d^2 + (x_i-x_j)*d(1/d^2)/dx_i]
    # d(1/d^2)/dx_i = -2(x_i-x_j)/d^4 ... wait
    # d/dx_i (d_ij^-2) = -2 d_ij^-3 * d(d_ij)/d(x_i) = -2 d^-3 * (x_i - x_j)/d = -2(x_i-x_j)/d^4
    # So: -[I/d^2 + (x_i-x_j)*(-2(x_i-x_j)^T/d^4)]
    # = -I/d^2 + 2(x_i-x_j)(x_i-x_j)^T/d^4
    # = (1/d^2)[-I + 2 u_ij u_ij^T]
    
    # H_ii = sum_{j!=i} (1/d_ij^2)[-I + 2 u_ij u_ij^T]
    
    # Build the 3N x 3N Hessian
    H = np.zeros((3*n, 3*n))
    for i in range(n):
        for j in range(i+1, n):
            d2 = dist[i, j] ** 2
            d2_safe = max(d2, 1e-16)
            uij = u[i, j]  # (3,)
            outer = np.outer(uij, uij)  # (3,3)
            
            # Off-diagonal block H_ij = H_ji = (1/d^2)[I - 2 u u^T]
            block = (1.0/d2_safe) * (np.eye(3) - 2.0 * outer)
            
            H[3*i:3*i+3, 3*j:3*j+3] = block
            H[3*j:3*j+3, 3*i:3*i+3] = block
            
            # Diagonal blocks
            H[3*i:3*i+3, 3*i:3*i+3] += (1.0/d2_safe) * (-np.eye(3) + 2.0 * outer)
            H[3*j:3*j+3, 3*j:3*j+3] += (1.0/d2_safe) * (-np.eye(3) + 2.0 * outer)
    
    return H


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _energy_fast(pts):
    """Fast energy using dot-product distance computation."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and tangent-space gradient using dot-product distances."""
    n = pts.shape[0]
    pts = _normalize(pts)
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))

    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)

    # grad_i = -sum_j W_ij * x_i + sum_j W_ij * x_j
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)

    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts

    return energy, grad


def _exp_map(pts, tangent):
    """Exponential map on the unit sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    # For small angles, use series expansion for better numerical stability
    mask_small = norms < 1e-6
    # General case: x' = x*cos(theta) + v*sin(theta)/|v|
    theta = norms_safe
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    v_normalized = tangent / norms_safe
    new_pts = pts * cos_theta + v_normalized * sin_theta
    # For very small steps, use Taylor series: cos(t) ~ 1-t^2/2, sin(t)/t ~ 1-t^2/6
    if np.any(mask_small):
        t2 = norms[mask_small] ** 2
        cos_t = 1.0 - t2 / 2.0
        sin_t_over_t = 1.0 - t2 / 6.0
        new_pts[mask_small] = pts[mask_small] * cos_t + tangent[mask_small] * sin_t_over_t
    return _normalize(new_pts)


def _riemannian_polish(pts, deadline, max_iter=1000, lr_init=1e-3):
    """
    Riemannian gradient descent with backtracking line search on the sphere.
    Uses exponential map for retraction with momentum for faster convergence.
    """
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)
    
    lr = lr_init
    momentum = np.zeros_like(best_pts)
    beta = 0.9  # momentum coefficient
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(best_pts)
        g_norm = np.linalg.norm(g)
        
        if g_norm < 1e-12:
            break
        
        # Update momentum
        momentum = beta * momentum - lr * g
        # Project momentum to tangent space
        dot_m = np.sum(momentum * best_pts, axis=1, keepdims=True)
        momentum = momentum - dot_m * best_pts
        
        # Backtracking line search
        alpha = 1.0
        accepted = False
        for _ in range(40):
            step = alpha * momentum
            new_pts = _exp_map(best_pts, step)
            new_e = _energy_fast(new_pts)
            # Armijo condition
            if new_e < e - 1e-4 * alpha * np.sum(momentum * g):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            if alpha < 1e-12:
                break
            new_pts = _exp_map(best_pts, alpha * momentum)
        
        new_e_precise = _precise_energy(new_pts)
        
        if new_e_precise < best_energy:
            best_pts = new_pts
            best_energy = new_e_precise
            # Slightly increase learning rate if we made good progress
            lr = min(lr * 1.02, 0.1)
            beta = min(beta * 1.001, 0.95)
        else:
            lr *= 0.8
            beta *= 0.9
    
    return best_pts, best_energy


def _precise_energy(pts):
    """Compute energy using math.fsum for final scoring accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _langevin_verlet(pts, deadline, rng, max_iter=3000, dt=0.003, gamma=0.5, T_init=0.5, T_min=1e-6):
    """
    Langevin dynamics with Velocity Verlet integrator on the sphere.
    Uses a slower cooling schedule for broader exploration.
    """
    current = _normalize(pts.copy())
    velocity = np.zeros_like(current)
    best_e = _energy_fast(current)
    best_pts = current.copy()
    n = N
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Slower cooling: power-law cooling that decays more slowly
        frac = it / max_iter
        T = T_init * (T_min / T_init) ** (frac ** 0.4)
        
        # 1. Half step velocity update with force
        e, g = _energy_grad(current)
        F = -g
        velocity += 0.5 * dt * F
        
        # 2. Full step position update
        new_pts = current + dt * velocity
        new_pts = _normalize(new_pts)
        
        # 3. Compute new force
        new_e, new_g = _energy_grad(new_pts)
        new_F = -new_g
        
        # 4. Half step velocity update with new force
        velocity += 0.5 * dt * new_F
        
        # 5. Apply friction and noise (Langevin kick)
        # Project velocity to tangent space of new_pts
        dot_v = np.sum(velocity * new_pts, axis=1, keepdims=True)
        velocity = velocity - dot_v * new_pts
        
        # Damping
        velocity *= (1.0 - gamma * dt)
        
        # Noise
        if T > T_min:
            noise_scale = np.sqrt(2.0 * gamma * T * dt)
            velocity += noise_scale * rng.standard_normal((N, 3))
            
            # Re-project after adding noise
            dot_v = np.sum(velocity * new_pts, axis=1, keepdims=True)
            velocity = velocity - dot_v * new_pts
        
        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()
            
        current = new_pts
        
    return best_pts, best_e


def _optimize_from_start(pts, deadline):
    """
    Optimize from a single starting configuration using multi-phase L-BFGS-B 
    followed by Riemannian polish.
    Returns (best_pts, best_precise_energy).
    """
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)

    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_grad(p)
        return e, g.reshape(-1)

    # Phase 1: Coarse L-BFGS-B
    if time.time() < deadline - 25:
        try:
            x0 = best_pts.reshape(-1)
            result = minimize(
                objective_and_grad, x0,
                method='L-BFGS-B', jac=True,
                options={'maxiter': 3000, 'maxfun': 15000, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if result.fun < best_energy:
                new_pts = _normalize(result.x.reshape(n, 3))
                e_p = _precise_energy(new_pts)
                if e_p < best_energy:
                    best_pts = new_pts
                    best_energy = e_p
        except Exception:
            pass

    # Phase 2: Fine L-BFGS-B with tighter tolerances
    if time.time() < deadline - 12:
        try:
            x0 = best_pts.reshape(-1)
            result = minimize(
                objective_and_grad, x0,
                method='L-BFGS-B', jac=True,
                options={'maxiter': 8000, 'maxfun': 40000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if result.fun < best_energy:
                new_pts = _normalize(result.x.reshape(n, 3))
                e_p = _precise_energy(new_pts)
                if e_p < best_energy:
                    best_pts = new_pts
                    best_energy = e_p
        except Exception:
            pass

    # Phase 3: Riemannian Polish (High precision with momentum)
    if time.time() < deadline - 3:
        try:
            r_pts, r_e = _riemannian_polish(best_pts, deadline, max_iter=2000, lr_init=1e-3)
            if r_e < best_energy:
                best_pts = r_pts
                best_energy = r_e
        except Exception:
            pass

    # Phase 4: Final ultra-precise L-BFGS-B
    if time.time() < deadline - 1:
        try:
            x0 = best_pts.reshape(-1)
            result = minimize(
                objective_and_grad, x0,
                method='L-BFGS-B', jac=True,
                options={'maxiter': 10000, 'maxfun': 50000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            if result.fun < best_energy:
                new_pts = _normalize(result.x.reshape(n, 3))
                e_p = _precise_energy(new_pts)
                if e_p < best_energy:
                    best_pts = new_pts
                    best_energy = e_p
        except Exception:
            pass

    return best_pts, best_energy


def _spectral_init(n, rng, l_max=8):
    """
    Generate a smooth initial configuration by sampling a random band-limited
    spherical function.
    """
    theta = rng.uniform(0, 2 * np.pi, n)
    cos_phi = 2.0 * rng.uniform(0, 1, n) - 1.0
    sin_phi = np.sqrt(1.0 - cos_phi ** 2)

    pts = np.column_stack([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        cos_phi
    ])

    # Add smooth low-frequency perturbation
    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            amp = rng.normal(0, 0.01 / (l + 1))
            phase = rng.uniform(0, 2 * np.pi)
            t = np.arctan2(pts[:, 1], pts[:, 0])
            c = pts[:, 2]
            mod = amp * np.cos(m * t + phase) * (c ** abs(m))
            pts[:, 2] += mod * 0.1
            pts[:, 0] += amp * 0.05 * np.sin(m * t + phase) * sin_phi

    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    rng = np.random.default_rng(12345)

    # Generate diverse initializations
    initializations = []

    # 1. Fibonacci spiral (best structured start)
    initializations.append(_fibonacci_spiral(N))

    # 2. Spectral/smooth initializations
    for seed in range(5):
        r = np.random.default_rng(seed * 100 + 7)
        initializations.append(_spectral_init(N, r, l_max=6))

    # 3. Random initializations
    for i in range(8):
        initializations.append(_random_sphere(N, rng))

    # 4. Perturbed Fibonacci
    base = _fibonacci_spiral(N)
    for scale in [0.003, 0.005, 0.01, 0.02]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))

    # 5. Additional structured: shifted golden spiral
    for offset in [0.0, 0.15, 0.3, 0.5]:
        initializations.append(_fibonacci_spiral(N, offset))

    # 6. Low-discrepancy-ish: stratified sampling
    try:
        from scipy.spatial import Delaunay
        # Use a grid-based approach for better coverage
        n_per_ring = 10
        pts_strat = []
        for ring in range(10):
            z = -1.0 + 2.0 * (ring + 0.5) / 10.0
            r = np.sqrt(max(0.0, 1.0 - z*z))
            for k in range(n_per_ring):
                theta = 2.0 * np.pi * (k + 0.5) / n_per_ring + 0.1 * ring
                pts_strat.append([r * np.cos(theta), r * np.sin(theta), z])
        pts_strat = np.array(pts_strat)
        initializations.append(_normalize(pts_strat))
    except Exception:
        pass

    # Sort by quick energy to prioritize good starts
    scored_inits = []
    for init in initializations:
        e_quick = _energy_fast(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])

    best_pts = None
    best_energy = float('inf')

    # Phase A: Quick optimization of top initializations to find good candidates
    # Use a shorter deadline for initial screening
    screen_deadline = start_time + 50.0
    
    for e_quick, init_pts in scored_inits[:8]:
        if time.time() > screen_deadline:
            break

        try:
            optimized_pts, e_precise = _optimize_from_start(init_pts, screen_deadline)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = optimized_pts
        except Exception:
            pass

    # Phase B: Langevin exploration from best candidates to find new basins
    if best_pts is not None and time.time() < deadline - 50:
        lang_deadline = deadline - 45
        for e_quick, init_pts in scored_inits[:3]:
            if time.time() > lang_deadline:
                break
            try:
                lang_pts, lang_e = _langevin_verlet(init_pts, lang_deadline, rng, 
                                                     max_iter=2000, dt=0.001, 
                                                     gamma=0.3, T_init=0.2, T_min=1e-7)
                if lang_e < best_energy:
                    # Polish the Langevin result
                    polished_pts, polished_e = _optimize_from_start(lang_pts, deadline - 10)
                    if polished_e < best_energy:
                        best_energy = polished_e
                        best_pts = polished_pts
            except Exception:
                pass

    # Phase C: Thorough optimization of remaining good starts
    for e_quick, init_pts in scored_inits:
        if time.time() > deadline - 25:
            break

        # Skip starts that are clearly worse than current best
        if e_quick > best_energy + 2.0:
            continue

        try:
            optimized_pts, e_precise = _optimize_from_start(init_pts, deadline - 5)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = optimized_pts
        except Exception:
            pass

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)

    # Final high-precision polish on the best configuration
    if time.time() < deadline - 5:
        try:
            final_pts, final_e = _optimize_from_start(best_pts, deadline)
            if final_e < best_energy:
                best_pts = final_pts
                best_energy = final_e
        except Exception:
            pass

    # Final normalization and validation
    best_pts = _normalize(best_pts)
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _normalize(_fibonacci_spiral(N))

    return best_pts
