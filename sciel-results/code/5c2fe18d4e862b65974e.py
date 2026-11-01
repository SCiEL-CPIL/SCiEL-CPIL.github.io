import time
import math
import numpy as np

N = 150
_DEADLINE_MARGIN = 5.0  # seconds to leave before hard timeout

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

def _project_to_tangent(pts, vecs):
    """Project vectors to tangent space at each point."""
    dot_prods = np.sum(vecs * pts, axis=1, keepdims=True)
    return vecs - dot_prods * pts

def _lbfgs_sphere(pts, time_limit=10.0, max_iter=200, m=10):
    """L-BFGS on the sphere using exponential maps and parallel transport."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # L-BFGS history
    s_history = []
    y_history = []
    
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
        
        r = q
        
        # Two-loop recursion (forward)
        d = r.copy()
        for k in range(len(rho)):
            s_k = s_history[k]
            y_k = y_history[k]
            
            ys = np.sum(y_k * s_k)
            if ys < 1e-30:
                continue
            rho_k = 1.0 / ys
            beta_k = rho_k * np.sum(y_k * r)
            d += (rho[k] - beta_k) * s_k
        
        # Ensure d is in tangent space
        d = _project_to_tangent(current, d)
        
        # Check if direction is meaningful
        dir_norm = np.linalg.norm(d)
        if dir_norm < 1e-14:
            break
        
        # Line search (Armijo backtracking)
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        directional_deriv = np.sum(grad * d)
        
        if directional_deriv > 0:
            d = -d
            directional_deriv = -directional_deriv
        
        for ls in range(30):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            if new_energy > energy + c1 * alpha * directional_deriv:
                break
            
            alpha *= 0.7
            if alpha < 1e-12:
                break
        
        if best_alpha == 0.0:
            alpha_try = 0.01
            for _ in range(5):
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
        s_vec = best_pts - current
        y_vec = best_grad - grad
        
        s_vec = _project_to_tangent(current, s_vec)
        y_vec = _project_to_tangent(best_pts, y_vec)
        
        s_history.append(s_vec)
        y_history.append(y_vec)
        
        if len(s_history) > m:
            s_history.pop(0)
            y_history.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-12:
            break
    
    return current, energy

def _riemannian_cg(pts, time_limit=15.0, max_iter=200):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Backtracking line search
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        for ls in range(30):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
            if new_energy > energy:
                break
            
            alpha *= 0.8
        
        if best_alpha == 0.0:
            break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Polak-Ribiere beta for conjugate gradient
        d_new = -grad
        beta = 0.0
        if it > 0:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
        
        d = d_new + beta * d
        prev_grad = grad.copy()
    
    return current, energy

def _cartesian_projected_grad(pts, time_limit=10.0, max_iter=100):
    """Project gradient steps in Cartesian space, then renormalize."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 0.01
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        _, grad = _energy_and_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-14:
            break
        
        # Try step in Cartesian space
        alpha = step_size
        accepted = False
        for _ in range(20):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < energy - 1e-15:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            step_size = min(alpha * 1.1, 0.05)
        else:
            step_size *= 0.5
            if step_size < 1e-16:
                break
    
    return current, energy

def _dual_representation_opt(pts, time_limit=30.0):
    """
    Alternate between Riemannian CG on sphere and Cartesian projected gradient.
    This leverages global dispersion for initialization and second-order curvature
    information for escaping shallow local minima.
    """
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    best_pts = current.copy()
    best_energy = energy
    
    phase = 0  # 0: Riemannian CG, 1: Cartesian PG
    max_phases = 8
    
    for p in range(max_phases):
        time_left = time_limit - (time.time() - start)
        if time_left < 3:
            break
        
        if phase % 2 == 0:
            # Riemannian CG phase
            phase_time = min(15.0, time_left - 2)
            if phase_time > 2:
                current, energy = _riemannian_cg(current, time_limit=phase_time, max_iter=150)
        else:
            # Cartesian projected gradient phase
            phase_time = min(10.0, time_left - 2)
            if phase_time > 2:
                current, energy = _cartesian_projected_grad(current, time_limit=phase_time, max_iter=80)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = current.copy()
        
        # Small perturbation to escape potential plateaus
        if p > 0 and p < max_phases - 1:
            rng = np.random.default_rng(p * 7 + 100)
            perturbation = rng.standard_normal((N, 3)) * 0.05
            perturbation = _project_to_tangent(current, perturbation)
            current = _exponential_map(current, perturbation)
            current = _normalize(current)
            energy = _log_energy_fast(current)
        
        phase += 1
    
    return best_pts, best_energy

def _basin_hopping(pts, time_limit=30.0, max_restarts=8):
    """Perform local optimization followed by basin hopping."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization with dual-representation
    time_left = time_limit - (time.time() - start)
    if time_left > 8:
        current, energy = _dual_representation_opt(current, time_limit=min(20, time_left - 3))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 0.5
    decay = 0.85
    
    for rh in range(max_restarts):
        time_left = time_limit - (time.time() - start)
        if time_left < 8:
            break
        
        # Perturb with adaptive scale based on gradient magnitude
        _, grad = _energy_and_grad(current)
        grad_mag = np.linalg.norm(grad)
        
        rng = np.random.default_rng(rh + 42)
        base_scale = 0.3 * (0.75 ** rh)
        # Scale perturbation inversely with gradient magnitude to explore more when stuck
        adaptive_factor = 1.0 + 0.5 * min(grad_mag / 0.1, 3.0)
        scale = base_scale * adaptive_factor * (1.0 + 0.2 * math.sin(rh * 1.7))
        
        perturbations = rng.standard_normal((N, 3)) * scale
        perturbations = _project_to_tangent(current, perturbations)
        
        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point with dual-representation
        opt_time = min(15, time_left - 4)
        if opt_time < 4:
            break
        new_opt_pts, new_energy = _dual_representation_opt(new_pts, time_limit=opt_time)
        
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

def _halton_3d_sphere(n, base_start=2):
    """Generate Halton sequence points using 3D coordinates mapped to sphere."""
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
        u3 = halton(i, base_start + 2)
        
        # Map to sphere using spherical coordinates
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
    
    # Initializations - more diverse starting points
    inits = []
    
    # Halton-based initializations (primary, low-discrepancy)
    inits.append(_halton_sphere(N, base_start=2))
    inits.append(_halton_sphere(N, base_start=3))
    inits.append(_halton_3d_sphere(N, base_start=2))
    inits.append(_halton_3d_sphere(N, base_start=5))
    
    # Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # Random inits with different seeds
    for s in range(3):
        rng = np.random.default_rng(s * 7 + 1)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Halton with varying scales
    for s in range(2):
        rng = np.random.default_rng(s * 13 + 200)
        scale = 0.1 + 0.08 * s
        pts = _halton_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)
    
    # Perturbed Fibonacci
    for s in range(2):
        rng = np.random.default_rng(s * 13 + 300)
        scale = 0.1 + 0.08 * s
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)
    
    # Spherical design-like init using greedy farthest-point sampling
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
    
    # Process each initialization
    n_inits = len(inits)
    for idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 18:
            break
        
        # Allocate time: give more time to earlier inits, reserve for final polish
        time_remaining_for_inits = remaining - 12
        inits_remaining = n_inits - idx
        time_per_restart = time_remaining_for_inits / inits_remaining
        time_per_restart = min(28.0, max(12.0, time_per_restart))
        
        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = _halton_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish - use remaining time for high-precision local optimization
    time_left = deadline - time.time()
    if time_left > 3:
        polish_time = max(3.0, time_left - 1.5)
        try:
            # Final dual-representation polish
            final_pts, final_energy = _dual_representation_opt(best_pts, time_limit=polish_time)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
        
        # Second polish pass with L-BFGS if time remains
        time_left2 = deadline - time.time()
        if time_left2 > 4:
            try:
                final_pts2, final_energy2 = _lbfgs_sphere(best_pts, time_limit=max(3.0, time_left2 - 1.5), max_iter=300, m=20)
                if final_energy2 < best_energy:
                    best_energy = final_energy2
                    best_pts = final_pts2
            except Exception:
                pass
            
    return _normalize(best_pts)
