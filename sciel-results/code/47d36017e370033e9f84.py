"""Dual-representation optimization for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, maximum_filter

N = 150


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


def _penalty_energy_and_grad(pts, min_sep=0.05, penalty_scale=1000.0):
    """Energy with penalty for points closer than min_sep, plus gradient."""
    e, g = _energy_and_grad(pts)
    
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Penalty: sum of (min_sep - d)^2 for d < min_sep
    viol = np.maximum(0, min_sep - dists)
    penalty = penalty_scale * np.sum(viol * viol)
    
    # Gradient of penalty: -2 * penalty_scale * (min_sep - d) * (x_i - x_j)/d for d < min_sep
    # Only for violating pairs
    pen_grad = np.zeros_like(pts)
    mask = dists < min_sep
    mask.fill_diagonal(False)
    for i in range(n):
        for j in range(i + 1, n):
            if mask[i, j]:
                d = dists[i, j]
                if d < 1e-12:
                    continue
                factor = -2 * penalty_scale * (min_sep - d) / d
                pen_grad[i] += factor * diff[i, j]
                pen_grad[j] -= factor * diff[i, j]
    
    # Project penalty gradient to tangent space
    dot_prod = np.sum(pen_grad * pts, axis=1)
    pen_grad -= dot_prod[:, np.newaxis] * pts
    
    return e + penalty, g + pen_grad


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _coulomb_gas_init(n, grid_res=128, n_iter=500, time_limit=20.0):
    """Mean-field Coulomb gas: solve self-consistent charge density on sphere,
    then extract n highest-density peaks as initial points."""
    start = time.time()
    
    # Create spherical grid using latitude/longitude
    theta = np.linspace(0, np.pi, grid_res)  # polar angle
    phi = np.linspace(0, 2 * np.pi, 2 * grid_res, endpoint=False)  # azimuthal
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian for distance calculations
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)
    
    # Compute pairwise squared chordal distances on the grid
    # Grid points as flat array
    grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    n_grid = len(grid_pts)
    
    # For memory efficiency, compute distances in blocks
    # But for grid_res=128, we have 128*256 = 32768 points, which is too many for full pairwise
    # Use a coarser approach: compute the mean-field potential iteratively
    
    # Start with uniform density
    rho = np.ones((grid_res, 2 * grid_res))
    
    # The self-consistent equation: rho(x) = exp(beta * V(x)) where V(x) = sum_j rho(y_j) log(1/d(x,y_j))
    # We solve this via iterative relaxation
    
    # Precompute the kernel: log(1/d) for each grid point to all others
    # This is O(n_grid^2) which is ~1e9 for 32768 points - too slow
    # Instead, use a smaller grid or approximate
    
    # Use a smaller effective grid
    eff_res = min(grid_res, 64)
    theta_eff = np.linspace(0, np.pi, eff_res)
    phi_eff = np.linspace(0, 2 * np.pi, 2 * eff_res, endpoint=False)
    TH, PH = np.meshgrid(theta_eff, phi_eff, indexing='ij')
    
    Xe = np.sin(TH) * np.cos(PH)
    Ye = np.sin(TH) * np.sin(PH)
    Ze = np.cos(TH)
    grid_eff = np.column_stack([Xe.ravel(), Ye.ravel(), Ze.ravel()])
    neff = len(grid_eff)
    
    # Compute pairwise distances on effective grid
    diff = grid_eff[:, np.newaxis, :] - grid_eff[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    kernel = np.log(1.0 / np.sqrt(dists_sq))
    
    # Iterative relaxation
    rho_eff = np.ones(neff) / neff
    beta = 50.0  # inverse temperature
    
    for it in range(n_iter):
        if time.time() - start > time_limit - 2:
            break
        
        # Compute potential V(x_i) = sum_j rho_j * kernel(i,j)
        V = kernel @ rho_eff
        
        # Update density: rho_i proportional to exp(beta * V_i)
        V_max = np.max(V)
        rho_new = np.exp(beta * (V - V_max))
        rho_new /= np.sum(rho_new)
        
        # Damping for stability
        rho_eff = 0.5 * rho_eff + 0.5 * rho_new
        
        # Check convergence
        if it % 50 == 0 and it > 0:
            change = np.max(np.abs(rho_eff - rho_new))
            if change < 1e-8:
                break
    
    # Find the n highest-density peaks
    # First, smooth the density to find broad peaks
    rho_2d = rho_eff.reshape(eff_res, 2 * eff_res)
    rho_smooth = gaussian_filter(rho_2d, sigma=2.0)
    
    # Find local maxima
    max_filter = maximum_filter(rho_smooth, size=5)
    is_max = (rho_smooth == max_filter) & (rho_smooth > np.percentile(rho_smooth, 80))
    
    # Get coordinates of local maxima
    max_indices = np.where(is_max)
    if len(max_indices[0]) == 0:
        max_indices = np.where(rho_smooth > np.percentile(rho_smooth, 90))
    
    max_theta = theta_eff[max_indices[0]]
    max_phi = phi_eff[max_indices[1]]
    max_vals = rho_smooth[max_indices]
    
    # Sort by density value (highest first)
    sort_idx = np.argsort(-max_vals)
    max_theta = max_theta[sort_idx]
    max_phi = max_phi[sort_idx]
    max_vals = max_vals[sort_idx]
    
    # If we don't have enough peaks, add points from remaining high-density regions
    if len(max_theta) < n:
        remaining = rho_smooth.copy()
        remaining[max_indices] = 0
        while len(max_theta) < n:
            flat_idx = np.argmax(remaining)
            i, j = np.unravel_index(flat_idx, remaining.shape)
            max_theta = np.append(max_theta, theta_eff[i])
            max_phi = np.append(max_phi, phi_eff[j])
            remaining[i, j] = 0
    
    # Take top n peaks
    max_theta = max_theta[:n]
    max_phi = max_phi[:n]
    
    # Convert to Cartesian
    pts = np.column_stack([
        np.sin(max_theta) * np.cos(max_phi),
        np.sin(max_theta) * np.sin(max_phi),
        np.cos(max_theta)
    ])
    pts = _normalize(pts)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    pts += rng.standard_normal((n, 3)) * 0.01
    pts = _normalize(pts)
    
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


def _riemannian_cg(pts, time_limit=15.0, max_iter=300):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        for ls in range(50):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
            if new_energy <= energy + 1e-4 * alpha * np.sum(grad * d):
                break
            
            alpha *= 0.7
        
        if best_alpha == 0.0:
            alpha = 1e-3
            for ls in range(20):
                new_pts = _exponential_map(current, alpha * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_alpha = alpha
                    best_grad = new_grad
                    break
                alpha *= 0.5
            if best_alpha == 0.0:
                break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        d_new = -grad
        beta = 0.0
        if it > 0:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
            if beta > 1.0:
                beta = 0.0
        
        d = d_new + beta * d
        prev_grad = grad.copy()
    
    return current, energy


def _lbfgs_optimize(pts, time_limit=30.0, use_penalty=False, min_sep=0.05, penalty_scale=1000.0):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        if use_penalty:
            energy, grad = _penalty_energy_and_grad(p, min_sep, penalty_scale)
        else:
            energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()
    
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 8000, 'ftol': 1e-20, 'gtol': 1e-15,
                            'maxfun': 30000, 'maxls': 100})
    
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy


def _penalty_lbfgs(pts, time_limit=30.0):
    """L-BFGS with penalty for minimum separation."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Start with moderate penalty, reduce over time
    min_sep = 0.08
    penalty_scale = 5000.0
    
    for phase in range(4):
        if time.time() - start > time_limit - 2:
            break
        
        time_left = time_limit - (time.time() - start) - 1
        if time_left < 3:
            break
        
        # Reduce penalty and separation as we converge
        min_sep_cur = min_sep * (0.7 ** phase)
        penalty_cur = penalty_scale * (0.5 ** phase)
        
        current, _ = _lbfgs_optimize(current, time_limit=min(time_left, 15.0),
                                      use_penalty=True, min_sep=min_sep_cur, penalty_scale=penalty_cur)
        energy = _log_energy_fast(current)
    
    return current, energy


def _dual_optimize(pts, time_limit=40.0):
    """Alternate between L-BFGS-B and Riemannian CG with Newton polish."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Phase 1: L-BFGS-B with penalty for coarse optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 10:
        current, energy = _penalty_lbfgs(current, time_limit=min(time_left * 0.4, 16.0))
    
    # Phase 2: Standard L-BFGS-B for fine optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 8:
        current, energy = _lbfgs_optimize(current, time_limit=min(time_left * 0.3, 12.0), use_penalty=False)
    
    # Phase 3: Riemannian CG for fine optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 5:
        current, energy = _riemannian_cg(current, time_limit=min(time_left * 0.4, 10.0))
    
    # Phase 4: Final L-BFGS-B micro-polish
    time_left = time_limit - (time.time() - start)
    if time_left > 2:
        current, energy = _lbfgs_optimize(current, time_limit=min(time_left, 6.0), use_penalty=False)
    
    return current, energy


def _polish(pts, time_limit=5.0):
    """High-precision polish using L-BFGS with tight tolerances."""
    start = time.time()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()
    
    x0 = pts.flatten()
    
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 5000, 'ftol': 1e-25, 'gtol': 1e-18,
                            'maxfun': 20000, 'maxls': 50, 'm': 30})
    
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    # Additional gradient-based steps if time remains
    if time.time() - start < time_limit - 0.5:
        current = x_opt
        for it in range(200):
            if time.time() - start > time_limit - 0.1:
                break
            e, grad = _energy_and_grad(current)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-17:
                break
            # Small gradient descent step
            alpha = 1e-6 / (grad_norm + 1e-15)
            new_pts = _exponential_map(current, -alpha * grad)
            new_e = _log_energy_fast(new_pts)
            if new_e < e:
                current = new_pts
                energy = new_e
    
    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Halton sequence (primary initialization per strategy)
    pts_halton = _halton_sphere(N)
    initializations.append(pts_halton.copy())
    
    # 2. Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    initializations.append(pts_fib.copy())
    
    # 3. Coulomb gas mean-field initialization
    try:
        pts_cg = _coulomb_gas_init(N, grid_res=64, n_iter=300, time_limit=15.0)
        initializations.append(pts_cg.copy())
    except Exception:
        pass
    
    # 4. Random initializations
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 5. Uniform random on sphere
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 321 + 99)
        pts_uniform = []
        for _ in range(N):
            while True:
                u1 = rng_local.uniform(-1, 1)
                u2 = rng_local.uniform(-1, 1)
                s = u1 * u1 + u2 * u2
                if s < 1.0 and s > 1e-15:
                    break
            scale = 2.0 * np.sqrt(1.0 - s)
            pts_uniform.append([u1 * scale, u2 * scale, 1.0 - s])
        pts_uniform = np.array(pts_uniform)
        pts_uniform = _normalize(pts_uniform)
        initializations.append(pts_uniform)
    
    # 6. Perturbed Halton
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 789 + 23)
        pts_pert = pts_halton + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 7. Perturbed Fibonacci
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = pts_fib + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    best_pts = initializations[0].copy()
    best_energy = _log_energy_fast(best_pts)
    
    # Optimize each initialization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        time_left = deadline - time.time() - 3
        if time_left < 8:
            break
        
        remaining = len(initializations) - idx
        base_time = min(time_left, max(12.0, 50.0 / remaining))
        
        try:
            opt_pts, opt_energy = _dual_optimize(init_pts, time_limit=base_time)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            try:
                time_left = deadline - time.time() - 3
                if time_left > 5:
                    opt_pts, opt_energy = _lbfgs_optimize(init_pts, time_limit=min(time_left, 15.0))
                    if opt_energy < best_energy:
                        best_energy = opt_energy
                        best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # Final high-precision polish on best solution
    if time.time() < deadline - 2:
        final_time = min(deadline - time.time() - 1, 12.0)
        if final_time > 1.0:
            try:
                if final_time > 6:
                    final_pts, final_energy = _riemannian_cg(best_pts, time_limit=min(final_time * 0.5, 6.0))
                    if final_energy < best_energy:
                        best_energy = final_energy
                        best_pts = final_pts
                        final_time -= min(final_time * 0.5, 6.0)
                
                if final_time > 2:
                    final_pts, final_energy = _polish(best_pts, time_limit=min(final_time * 0.9, 8.0))
                    if final_energy < best_energy:
                        best_energy = final_energy
                        best_pts = final_pts
            except Exception:
                pass
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                mid = (best_pts[i] + best_pts[j]) / 2
                perp = best_pts[i] - best_pts[j]
                perp = perp / (np.linalg.norm(perp) + 1e-15)
                cross = np.cross(best_pts[i], perp)
                cross = cross / (np.linalg.norm(cross) + 1e-15)
                best_pts[i] = _normalize(best_pts[i] + 1e-8 * cross)
                best_pts[j] = _normalize(best_pts[j] - 1e-8 * cross)
    
    return best_pts
