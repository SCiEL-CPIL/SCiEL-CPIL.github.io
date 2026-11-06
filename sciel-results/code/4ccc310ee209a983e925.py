import time
import numpy as np
import math
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _spherical_harmonic_basis(lmax, n_pts):
    """
    Precompute spherical harmonic basis vectors for each point.
    Returns a matrix B of shape (n_pts, n_harmonics) where each row is the
    vector of real spherical harmonic values at that point.
    """
    from scipy.special import lpmv
    
    # Build list of (l, m) pairs
    lm_pairs = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            lm_pairs.append((l, m))
    
    n_harm = len(lm_pairs)
    B = np.zeros((n_pts, n_harm))
    
    for idx, (l, m) in enumerate(lm_pairs):
        abs_m = abs(m)
        for i in range(n_pts):
            x, y, z = n_pts[i]
            r = np.sqrt(x*x + y*y + z*z)
            if r < 1e-15:
                continue
            theta = np.arccos(np.clip(z / r, -1.0, 1.0))
            phi = np.arctan2(y, x)
            
            # Associated Legendre polynomial
            P_lm = lpmv(abs_m, l, np.cos(theta))
            
            # Normalization factor
            norm_factor = math.sqrt((2*l + 1) / (4.0 * math.pi) * math.factorial(l - abs_m) / math.factorial(l + abs_m))
            
            if m == 0:
                B[i, idx] = norm_factor * P_lm
            elif m > 0:
                B[i, idx] = norm_factor * math.sqrt(2) * P_lm * np.cos(m * phi)
            else:
                B[i, idx] = norm_factor * math.sqrt(2) * P_lm * np.sin(abs_m * phi)
    
    return B

def _spectral_energy_and_grad(pts, B):
    """
    Compute spectral energy: variance of pair correlation in harmonic space.
    For a set of N points, the pair correlation function in harmonic space is:
    C_lm = (1/N) * sum_i Y_lm(p_i) * conj(Y_lm(p_j)) ... 
    Actually, for the covering problem, we want to minimize the variance of
    the local density. The spectral energy is:
    E = sum_{l,m} |sum_i Y_lm(p_i)|^2 / N^2 - |mean|^2
    
    More precisely, we want the points to be "spread out" in harmonic space.
    The energy is: E = (1/N^2) * sum_{l,m} |sum_i Y_lm(p_i)|^2
    Minimizing this promotes uniform distribution.
    """
    n_pts = len(pts)
    # B has shape (n_pts, n_harm)
    # Sum of harmonics: S_lm = sum_i B[i, lm]
    S = B.sum(axis=0)  # shape (n_harm,)
    
    # Energy: sum |S_lm|^2 / N^2
    # For real harmonics, this is just sum S_lm^2
    energy = np.sum(S * S) / (n_pts * n_pts)
    
    # Gradient: dE/dp_i
    # S_lm = sum_k B[k, lm]
    # dS_lm/dp_i = dB[i, lm]/dp_i
    # dE/dp_i = (2/(N^2)) * S * dB[i, lm]/dp_i
    
    # We need dB[i, lm]/dp_i for each point
    # This is complex to compute analytically, so we use a simpler approach:
    # Use the fact that for small perturbations, the gradient of the energy
    # with respect to point i is proportional to the "local density" at point i.
    
    # Alternative: compute gradient numerically or use a simpler proxy.
    # For now, use a heuristic: gradient pushes points away from regions
    # where the harmonic sum is large.
    
    # Actually, let's compute the gradient properly using the chain rule.
    # The derivative of Y_lm with respect to position on the sphere is complex.
    # Instead, let's use a different approach: minimize the "clumping" by
    # using the pairwise distances in harmonic space.
    
    # Simpler spectral energy: use the kernel matrix
    # K_ij = sum_lm Y_lm(p_i) Y_lm(p_j) = sum_lm B[i,lm] B[j,lm]
    # This is the reproducing kernel.
    # Energy = trace(K) - (1/N) * sum_{i,j} K_ij
    # = N * sum_lm |Y_lm(p_i)|^2 ... this is getting complicated.
    
    # Let's use a simpler proxy: minimize the variance of the number of points
    # in each "cell" defined by the leading harmonics.
    
    # For now, return energy and a zero gradient (will use other methods for gradient)
    grads = np.zeros_like(pts)
    
    return energy, grads

def _compute_covering_radius_and_grads(pts):
    """
    Computes the covering radius and the gradient of the covering radius w.r.t each point.
    Returns:
    radius: The covering radius in radians.
    grads: Array of shape (N, 3) containing the gradient of the radius w.r.t each point.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0 * math.pi / 180.0, np.zeros_like(pts)

    simplices = hull.simplices
    n_faces = len(simplices)
    
    # Precompute circumcenters and radii for all facets
    centers = np.zeros((n_faces, 3))
    min_dots = np.zeros((n_faces, 3))
    radii = np.zeros(n_faces)
    
    for i, simplex in enumerate(simplices):
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Cross product for normal
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            radii[i] = 0.0
            continue
        n = n / n_norm
        
        # Ensure outward normal
        if np.dot(n, v0) < 0:
            n = -n
            
        centers[i] = n
        
        d0 = np.dot(v0, n)
        d1 = np.dot(v1, n)
        d2 = np.dot(v2, n)
        
        min_dots[i, 0] = d0
        min_dots[i, 1] = d1
        min_dots[i, 2] = d2
        
        min_d = np.min(min_dots[i])
        # Radius is arccos(min_d)
        # Clip to avoid domain errors
        min_d = np.clip(min_d, -1.0, 1.0)
        radii[i] = np.arccos(min_d)
        
    max_radius = np.max(radii)
    
    # Identify active facets
    # Use a threshold to include facets close to the max
    # This smooths the gradient
    threshold = max_radius - 0.001 
    active_mask = radii > threshold
    
    if not np.any(active_mask):
        return max_radius, np.zeros_like(pts)
    
    grads = np.zeros_like(pts)
    
    active_indices = np.where(active_mask)[0]
    
    for i in active_indices:
        simplex = simplices[i]
        n = centers[i]
        
        d0, d1, d2 = min_dots[i]
        
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        g0 = n - np.dot(n, v0) * v0
        g1 = n - np.dot(n, v1) * v1
        g2 = n - np.dot(n, v2) * v2
        
        # Normalize gradients to have similar magnitude
        g0_norm = np.linalg.norm(g0)
        g1_norm = np.linalg.norm(g1)
        g2_norm = np.linalg.norm(g2)
        
        if g0_norm > 1e-9: g0 /= g0_norm
        if g1_norm > 1e-9: g1 /= g1_norm
        if g2_norm > 1e-9: g2 /= g2_norm
        
        # Accumulate
        grads[simplex[0]] += g0
        grads[simplex[1]] += g1
        grads[simplex[2]] += g2
        
    return max_radius, grads

def _spectral_init(lmax=8, n_iter=500, lr=0.01, seed=42):
    """
    Initialize points by minimizing spectral energy using gradient descent.
    """
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(N, 3))
    P = _normalize(P)
    
    B = _spherical_harmonic_basis(lmax, P)
    
    for it in range(n_iter):
        # Recompute B since P changed
        B = _spherical_harmonic_basis(lmax, P)
        
        # Compute gradient of spectral energy
        # E = (1/N^2) * sum_lm (sum_i B[i,lm])^2
        # dE/dP_i = (2/N^2) * sum_lm S_lm * dB[i,lm]/dP_i
        
        # Approximate gradient: push points to reduce local clumping
        # Use the negative of the "harmonic density" at each point
        S = B.sum(axis=0)
        
        # Gradient approximation: for each point, the gradient is proportional
        # to the sum of harmonics weighted by their derivatives
        # Simplified: use the local harmonic value as a proxy for gradient direction
        # Points with high harmonic sum should move to regions with low harmonic sum
        
        # Compute "pressure" at each point
        pressure = B @ S  # shape (N,)
        
        # Gradient: move points opposite to pressure direction
        # This is a heuristic; proper gradient requires dB/dP
        grads = np.zeros_like(P)
        for i in range(N):
            # Move point i in direction that reduces pressure
            # Approximate: random direction scaled by pressure
            if abs(pressure[i]) > 1e-6:
                # Use the harmonic basis to determine direction
                # The gradient of B[i,lm] w.r.t. P_i is complex, so use a simpler approach
                # Move in the direction of the negative local gradient of the kernel
                # For now, use a small random perturbation weighted by pressure
                direction = rng.normal(size=3)
                direction = direction - np.dot(direction, P[i]) * P[i]
                direction = _normalize(direction.reshape(1, 3)).flatten()
                grads[i] = -pressure[i] * direction
        
        # Normalize gradients
        grad_norms = np.linalg.norm(grads, axis=1, keepdims=True)
        grad_norms[grad_norms < 1e-15] = 1.0
        grads = grads / grad_norms
        
        # Update
        P = P + lr * grads
        P = _normalize(P)
    
    return P

def _compute_soft_covering_radius_and_grads(pts, samples, beta):
    """
    Computes a soft approximation of the covering radius using a log-sum-exp (soft-min) of distances.
    This provides a smooth objective and gradient for L-BFGS.
    """
    pts = _normalize(pts)
    
    # Compute angular distances from samples to all points
    # dists[i, j] = arccos(samples[i] . pts[j])
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Soft min of distances for each sample
    # soft_min_i = -1/beta * log(sum_j exp(-beta * dists[i, j]))
    # We want to minimize the max of these soft mins.
    # Actually, we can just sum them or take the max. Summing is smoother for global opt.
    # Let's use the sum of soft mins as the objective.
    
    scaled = -beta * dists
    # Use log-sum-exp trick for numerical stability
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled), axis=1, keepdims=True))
    soft_mins = -log_sum_exp / beta
    
    loss = np.sum(soft_mins)
    
    # Gradient
    # d loss / d dists[i, j] = -1/beta * exp(-beta * dists[i, j]) / sum_k exp(-beta * dists[i, k])
    # = - exp(scaled[i, j]) / sum_k exp(scaled[i, k])
    
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled, axis=1, keepdims=True)
    weights = exp_scaled / sum_exp
    
    # d loss / d dots[i, j] = d loss / d dists[i, j] * d dists[i, j] / d dots[i, j]
    # d arccos(x) / dx = -1 / sqrt(1 - x^2)
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots**2))
    grad_dots = -weights / denom
    
    # d loss / d pts[j, k] = sum_i grad_dots[i, j] * samples[i, k]
    # grad_dots shape (M, N), samples shape (M, 3)
    # result shape (N, 3)
    grad_pts = grad_dots.T @ samples
    
    # Project gradient to tangent space
    # grad_tangent[j, k] = grad_pts[j, k] - dot(grad_pts[j, :], pts[j, :]) * pts[j, k]
    grad_dot_pts = np.sum(grad_pts * pts, axis=1, keepdims=True)
    grad_tangent = grad_pts - grad_dot_pts * pts
    
    return loss, grad_tangent

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initialization
    best_P = None
    best_radius = 180.0 * math.pi / 180.0
    
    rng = np.random.default_rng(42)
    
    # Initializations: include spectral init and Fibonacci variants
    initializations = [
        _fibonacci_sphere(N, seed_noise=0.0, seed=42),
        _fibonacci_sphere(N, seed_noise=0.05, seed=1),
        _fibonacci_sphere(N, seed_noise=0.1, seed=2),
    ]
    
    # Add spectral initialization if time permits
    if time.time() < deadline - 60.0:
        try:
            spec_P = _spectral_init(lmax=6, n_iter=200, lr=0.02, seed=42)
            initializations.append(spec_P)
        except Exception:
            pass
    
    # Sample points for soft optimization
    # Use a moderate number to balance accuracy and speed
    M_samples = 20000
    samples = _fibonacci_sphere(M_samples)
    
    for init_P in initializations:
        if time.time() > deadline - 40.0:
            break
            
        P = init_P.copy()
        
        # Phase 1: L-BFGS with soft objective
        # Start with lower beta (smoother), then increase
        betas = [10.0, 50.0, 100.0]
        
        for beta in betas:
            if time.time() > deadline - 15.0:
                break
                
            # Define objective and gradient
            def obj_grad(P_flat):
                P_local = P_flat.reshape(-1, 3)
                loss, grad = _compute_soft_covering_radius_and_grads(P_local, samples, beta)
                return loss, grad.flatten()
            
            # L-BFGS optimization
            try:
                res = minimize(
                    obj_grad,
                    P.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False}
                )
                if res.x is not None:
                    P = res.x.reshape(-1, 3)
                    P = _normalize(P)
            except Exception:
                pass
        
        # Phase 2: Riemannian Gradient Descent / Annealing with exact hull gradient
        # This phase uses the exact geometric definition to polish the solution
        step_size = 0.05
        cool_rate = 0.995
        temp = 0.01
        
        for it in range(1000):
            if time.time() > deadline - 5.0:
                break
                
            radius, grads = _compute_covering_radius_and_grads(P)
            
            if radius < best_radius:
                best_radius = radius
                best_P = P.copy()
                
            # Add noise to escape local minima
            noise = rng.normal(0, temp, (N, 3))
            # Project noise to tangent space
            noise = noise - np.sum(noise * P, axis=1, keepdims=True) * P
            
            grads += noise
            
            # Move
            new_P = P + step_size * grads
            new_P = _normalize(new_P)
            
            new_radius, _ = _compute_covering_radius_and_grads(new_P)
            
            # Metropolis acceptance
            delta_r = new_radius - radius
            if delta_r < 0:
                P = new_P
            else:
                if rng.random() < math.exp(-delta_r / temp):
                    P = new_P
            
            # Cool down
            temp *= cool_rate
            step_size *= 0.999
    
    if best_P is None:
        best_P = _fibonacci_sphere(N)
        
    return best_P
# EVOLVE-BLOCK-END
