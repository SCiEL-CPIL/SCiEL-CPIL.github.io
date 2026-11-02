import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE_MARGIN = 5.0  # seconds to leave before hard timeout

def _hessian_vector_product(pts, v):
    """Compute Hessian-vector product for the log energy on the sphere.
    
    The Hessian of E = sum_{i<j} log(1/|x_i - x_j|) projected to tangent space.
    For point i, the Hessian applied to v_i involves:
    - Diagonal: -sum_{j!=i} (1/d_ij^2) * v_i  (from second derivative of log(1/d))
    - Off-diagonal coupling: (1/d_ij^2) * (v_j - (v_j.x_i)x_i) projected terms
    
    We use the formula: H v = sum over pairs of the second-derivative contribution.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Hessian-vector product for the energy E = sum_{i<j} log(1/|xi-xj|)
    # dE/dxi = -sum_{j!=i} (xi - xj)/|xi-xj|^2
    # d2E/dxi^2 v = sum_{j!=i} [ (xi-xj)(xi-xj)^T / |xi-xj|^4 - delta_ij / |xi-xj|^2 ] v
    #            = sum_{j!=i} [ inv_d2_ij * (diff_ij * (diff_ij^T v) / d2_ij) - inv_d2_ij * v ]
    # But we need to be careful: the gradient is -sum (xi-xj)/d^2, so
    # Hessian of E at point i: sum_{j!=i} [ (xi-xj)(xi-xj)^T / d^4 - I/d^2 ]
    
    # For each i: H_i v_i = sum_{j!=i} [ (xi-xj)((xi-xj)^T v_i)/d^4 - v_i/d^2 ]
    #            = sum_{j!=i} [ (xi-xj)(dot_ij)/d^4 - v_i/d^2 ]
    # where dot_ij = (xi-xj)^T v_i
    
    # Vectorized:
    # term1[i] = sum_j inv_d2[i,j] * diff[i,j] * (diff[i,j] @ v_i) / d2[i,j]
    #          = sum_j diff[i,j] * (diff[i,j] @ v_i) * inv_d2[i,j]^2  ... wait
    # Actually: (xi-xj)(xi-xj)^T / d^4 = diff * (diff^T v) / d^4
    # And diff^T v = sum_k diff[i,j,k] * v_i[k]
    
    # Let's compute: for each pair (i,j), contribution to H_i v_i:
    #   diff[i,j] * (diff[i,j] dot v[i]) / dists_sq[i,j]^2 - v[i] / dists_sq[i,j]
    
    # Vectorized approach:
    # dot_v = diff @ v  (n x n matrix: dot_v[i,j] = diff[i,j] dot v[i])
    dot_v = np.sum(diff * v[np.newaxis, :, :], axis=2)  # (n, n): dot_v[i,j] = diff[i,j] . v[i]
    
    # term1[i] = sum_j diff[i,j] * dot_v[i,j] * inv_d2[i,j]^2
    # inv_d2^2 = 1/d^4
    inv_d4 = inv_d2 * inv_d2
    np.fill_diagonal(inv_d4, 0.0)
    
    # term1[i] = sum_j diff[i,j] * dot_v[i,j] * inv_d4[i,j]
    # diff is (n,n,3), dot_v is (n,n), inv_d4 is (n,n)
    term1 = np.sum(diff * dot_v[:, :, np.newaxis] * inv_d4[:, :, np.newaxis], axis=1)
    
    # term2[i] = v[i] * sum_j inv_d2[i,j]
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    term2 = v * sum_inv_d2[:, np.newaxis]
    
    # H v = term1 - term2  (this is for the unconstrained Hessian)
    hv = term1 - term2
    
    # Project to tangent space: hv - (hv . p) p
    dot_hp = np.sum(hv * pts, axis=1)
    hv -= dot_hp[:, np.newaxis] * pts
    
    return hv

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _lbfgs_sphere(pts, time_limit=10.0, max_iter=200, m=10):
    """L-BFGS on the sphere using exponential maps and parallel transport."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # L-BFGS history
    s_history = []  # step vectors (tangent at previous point)
    y_history = []  # gradient differences
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        q = grad.copy()
        rho = []
        
        # Two-loop recursion (backward)
        for k in range(len(s_history) - 1, -1, -1):
            s_k = s_history[k]
            y_k = y_history[k]
            
            ys = np.sum(y_k * s_k)
            if ys < 1e-30:
                rho.append(0.0)
                continue
            rho_k = 1.0 / ys
            rho.append(rho_k)
            alpha_k = rho_k * np.sum(s_k * q)
            q -= alpha_k * y_k
        
        # r = q (standard L-BFGS with H0 = I)
        r = q
        
        # Two-loop recursion (forward)
        d = r.copy()
        for k in range(len(rho)):
            s_k = s_history[k]
            y_k = y_history[k]
            
            if rho[k] == 0.0:
                continue
            ys = np.sum(y_k * s_k)
            if ys < 1e-30:
                continue
            beta_k = (1.0 / ys) * np.sum(y_k * r)
            d += (rho[k] - beta_k) * s_k
        
        # Ensure d is in tangent space
        dot_d = np.sum(d * current, axis=1)
        d -= dot_d[:, np.newaxis] * current
        
        # Check if direction is meaningful
        dir_norm = np.linalg.norm(d)
        if dir_norm < 1e-14:
            break
        
        # Line search (Armijo backtracking)
        alpha = 1.0
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        directional_deriv = np.sum(grad * d)
        
        # If directional derivative is positive, flip direction
        if directional_deriv > 0:
            d = -d
            directional_deriv = -directional_deriv
        
        for ls in range(35):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            # Armijo condition
            if new_energy > energy + c1 * alpha * directional_deriv:
                break
            
            alpha *= 0.5
            if alpha < 1e-14:
                break
        
        if best_alpha == 0.0:
            # Try smaller step
            alpha_try = 0.01
            for _ in range(8):
                new_pts = _exponential_map(current, alpha_try * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_grad = new_grad
                    best_alpha = alpha_try
                    break
                alpha_try *= 0.5
            
            if best_alpha == 0.0:
                break
        
        # Update L-BFGS history
        s_vec = best_pts - current  # approximate step
        y_vec = best_grad - grad
        
        # Project to tangent space
        dot_s = np.sum(s_vec * current, axis=1)
        s_vec -= dot_s[:, np.newaxis] * current
        
        dot_y = np.sum(y_vec * best_pts, axis=1)
        y_vec -= dot_y[:, np.newaxis] * best_pts
        
        s_history.append(s_vec)
        y_history.append(y_vec)
        
        if len(s_history) > m:
            s_history.pop(0)
            y_history.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-13:
            break
    
    return current, energy

def _newton_polish(pts, time_limit=5.0, max_iter=50, tol=1e-12):
    """Newton's method polish using Hessian-vector products and conjugate gradient.
    
    Uses a CG inner solver with the Hessian to achieve quadratic convergence
    near the minimum.
    """
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # Conjugate gradient on the Hessian to solve H d = -g
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # CG to solve H d = -grad
        r = -grad.copy()
        p = r.copy()
        rs = np.sum(r * r)
        
        for inner in range(30):
            Hp = _hessian_vector_product(current, p)
            pHp = np.sum(p * Hp)
            
            if abs(pHp) < 1e-30:
                break
            
            alpha_cg = rs / pHp
            
            # Ensure we're moving in a descent direction
            new_pts = _exponential_map(current, alpha_cg * p)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy >= energy:
                # Damped step
                alpha_cg *= 0.5
                new_pts = _exponential_map(current, alpha_cg * p)
                new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy >= energy:
                break
            
            r_new = new_grad - grad
            rs_new = np.sum(r_new * r_new)
            
            if rs_new < 1e-30:
                break
            
            beta_cg = rs_new / rs
            p = r_new + beta_cg * p
            r = r_new
            rs = rs_new
            current = new_pts
            energy = new_energy
            grad = new_grad
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
    
    return current, energy

def _basin_hopping(pts, time_limit=30.0, max_restarts=8):
    """Perform local optimization followed by basin hopping."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 5:
        current, energy = _lbfgs_sphere(current, time_limit=min(15, time_left - 2))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 0.5
    decay = 0.85
    
    for rh in range(max_restarts):
        time_left = time_limit - (time.time() - start)
        if time_left < 5:
            break
        
        # Perturb with adaptive scale - start larger, decay
        rng = np.random.default_rng(rh + 42)
        base_scale = 0.2 * (0.7 ** rh)
        scale = base_scale * (1.0 + 0.3 * math.sin(rh * 1.3))
        perturbations = rng.standard_normal((N, 3)) * scale
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point
        opt_time = min(12, time_left - 3)
        if opt_time < 3:
            break
        new_opt_pts, new_energy = _lbfgs_sphere(new_pts, time_limit=opt_time)
        
        # Metropolis Criterion
        delta_e = new_energy - energy
        accept = delta_e < 0
        if not accept and temp > 1e-6:
            try:
                accept = np.random.rand() < math.exp(-delta_e / temp)
            except OverflowError:
                accept = False
        
        if accept:
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
        
        temp *= decay
    
    # Final polish within basin hopping
    time_left = time_limit - (time.time() - start)
    if time_left > 3:
        try:
            polished, pol_energy = _lbfgs_sphere(best_pts, time_limit=max(2.0, time_left - 1.0), max_iter=200, m=15)
            if pol_energy < best_energy:
                best_energy = pol_energy
                best_pts = polished
        except Exception:
            pass
    
    return best_pts, best_energy

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN
    
    # Initializations - focus on high-quality diverse starts
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    
    # Perturbed Fibonacci with varying scales
    for s in range(2):
        rng = np.random.default_rng(s * 13 + 200)
        scale = 0.08 + 0.06 * s
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)
    
    # Farthest-point sampling init
    try:
        rng = np.random.default_rng(999)
        many_pts = rng.standard_normal((N * 3, 3))
        many_pts = _normalize(many_pts)
        selected = [many_pts[0].copy()]
        min_dists = np.full(N * 3, np.inf)
        for _ in range(N - 1):
            d = np.linalg.norm(many_pts[:, np.newaxis, :] - selected[-1][np.newaxis, :, :], axis=2)
            min_dists = np.minimum(min_dists, d)
            next_idx = np.argmax(min_dists)
            selected.append(many_pts[next_idx].copy())
        inits.append(np.array(selected))
    except Exception:
        pass
    
    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization with concentrated time budget
    n_inits = len(inits)
    # Reserve 20s for final Newton polish
    total_search_time = 150.0 - _DEADLINE_MARGIN - 20.0
    
    for idx, init_pts in enumerate(inits):
        remaining = deadline - time.time() - 20.0  # reserve 20s for polish
        if remaining < 10:
            break
        
        inits_remaining = n_inits - idx
        time_per_restart = remaining / inits_remaining
        time_per_restart = min(30.0, max(12.0, time_per_restart))
        
        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish - L-BFGS then Newton for high precision
    time_left = deadline - time.time()
    if time_left > 3:
        # First: L-BFGS polish
        polish_time = max(2.0, time_left - 1.0)
        try:
            final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=polish_time, max_iter=500, m=15)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
        
        # Second: Newton polish for quadratic convergence
        time_left2 = deadline - time.time()
        if time_left2 > 2:
            try:
                newton_time = max(1.5, time_left2 - 0.5)
                newton_pts, newton_energy = _newton_polish(best_pts, time_limit=newton_time, max_iter=50, tol=1e-13)
                if newton_energy < best_energy:
                    best_energy = newton_energy
                    best_pts = newton_pts
            except Exception:
                pass
        
        # Third: final L-BFGS micro-polish
        time_left3 = deadline - time.time()
        if time_left3 > 1.5:
            try:
                micro_pts, micro_energy = _lbfgs_sphere(best_pts, time_limit=max(1.0, time_left3 - 0.3), max_iter=200, m=20)
                if micro_energy < best_energy:
                    best_energy = micro_energy
                    best_pts = micro_pts
            except Exception:
                pass
            
    return _normalize(best_pts)
