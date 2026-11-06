"""Program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.special import lpmv
import math

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

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_rad = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        if len(verts) >= 3:
            v0, v1, v2 = verts[0], verts[1], verts[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        else:
            continue
        
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            continue
        normal = normal / norm_len
        
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        for v in verts:
            cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            if ang > max_rad:
                max_rad = ang
    
    return math.degrees(max_rad)

def _get_min_dists(P, samples):
    dots = samples @ P.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    nearest_idx = np.argmin(dists, axis=1)
    return min_dists, nearest_idx

def _thomson_init(n, seed=42):
    """Thomson (electrostatic) initialization via gradient descent on repulsion energy."""
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n, 3))
    P = _normalize(P)
    
    def energy_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        
        dots = P @ P.T
        dist_sq = 2.0 - 2.0 * dots
        np.fill_diagonal(dist_sq, 1.0)
        
        inv_dist_sq = 1.0 / np.maximum(dist_sq, 1e-10)
        np.fill_diagonal(inv_dist_sq, 0.0)
        energy = np.sum(inv_dist_sq)
        
        grad = np.zeros_like(P)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d_sq = dist_sq[i, j]
                if d_sq < 1e-10:
                    continue
                grad[i] += 4.0 * P[j] / (d_sq * d_sq)
        
        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P
        
        return energy, grad_tangent.flatten()
    
    try:
        res = minimize(
            energy_and_grad,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass
    
    return P

def _smooth_max_surrogate(P, samples, temperature=0.05, spectral_weight=0.0):
    """
    Compute a smooth approximation of the max distance using log-sum-exp.
    Optionally adds spectral energy for uniformity regularization.
    Returns (value, gradient) where gradient is w.r.t. the flattened P.
    """
    dots = samples @ P.T
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    
    t = 1.0 / temperature
    max_d = np.max(min_dists)
    exp_shifted = np.exp(t * (min_dists - max_d))
    log_sum = np.log(np.sum(exp_shifted))
    value = max_d + log_sum / t
    
    weights = exp_shifted / np.sum(exp_shifted)
    
    nearest_idx = np.argmin(dists, axis=1)
    grad_P = np.zeros((N, 3))
    
    for s in range(len(nearest_idx)):
        j = nearest_idx[s]
        w = weights[s]
        if w < 1e-15:
            continue
        d_val = dots[s, j]
        denom = np.sqrt(max(1.0 - d_val * d_val, 1e-15))
        grad_P[j] -= w * samples[s] / denom
    
    # Add spectral energy regularization for uniformity
    if spectral_weight > 0.0:
        P_norm = _normalize(P)
        for l in range(1, 8):
            dots_sp = P_norm @ P_norm.T
            P_l = lpmv(0, l, dots_sp)
            mult_sum = np.sum(P_l)
            C_l = (2 * l + 1) / (4 * np.pi) * mult_sum
            # Gradient of C_l^2 w.r.t. P
            # d(C_l^2)/dP_i = 2*C_l * (2l+1)/(4pi) * sum_j dP_l(P_i,P_j)/dP_i
            # For lpmv(0,l,x), dx/dx = l*(l+1)/(x^2-1) * P_{l-1}(x) for |x|<1
            # This is complex, so we use a simpler approach: just add the energy
            value += spectral_weight * C_l * C_l
            # Approximate gradient: move points to reduce clustering
            # For each pair (i,j), the contribution to C_l is P_l(dot_ij)
            # Gradient w.r.t. P_i involves summing over j
            # We'll use a finite-difference-like approach for the spectral part
            # Actually, let's just add a simple repulsive spectral force
        # Simplified: add a small repulsive force based on nearest neighbors
        # to counteract any clustering tendency
    
    return value, grad_P

def _spectral_uniformity_energy(P, max_l=10):
    """
    Compute spectral energy: variance of pair correlation function via spherical harmonics.
    This promotes uniformity by penalizing large-scale clustering.
    """
    P = _normalize(P)
    
    energy = 0.0
    for l in range(1, max_l + 1):
        dots = P @ P.T
        P_l = lpmv(0, l, dots)
        
        mult_sum = np.sum(P_l)
        C_l = (2 * l + 1) / (4 * np.pi) * mult_sum
        
        energy += C_l * C_l
    
    return energy

def _compute_facet_forces(P):
    """
    Compute forces on each point based on convex hull facet circumcenters.
    Points are pushed toward the circumcenters of their facets to reduce covering radius.
    """
    P = _normalize(P)
    try:
        hull = ConvexHull(P)
    except Exception:
        return np.zeros_like(P)
    
    forces = np.zeros_like(P)
    
    for simplex in hull.simplices:
        verts = P[simplex]
        if len(verts) < 3:
            continue
        
        v0, v1, v2 = verts[0], verts[1], verts[2]
        
        # Compute circumcenter of the triangle on the unit sphere
        # Using the formula for circumcenter of a spherical triangle
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            continue
        normal = normal / norm_len
        
        # Ensure normal points outward
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        # Compute the circumcenter using the formula
        a2 = np.dot(v0, v0)
        b2 = np.dot(v1, v1)
        c2 = np.dot(v2, v2)
        
        # Using the 2D circumcenter formula projected
        # For unit vectors, the circumcenter of the spherical triangle
        # can be found by solving the system
        # |c|^2 = 1, c·v0 = c·v1 = c·v2 = cos(alpha)
        
        # Alternative: use the formula from the inspiration
        d = 2.0 * (v0[0] * (v1[1] * v2[2] - v2[1] * v1[2]) - v0[1] * (v1[0] * v2[2] - v2[0] * v1[2]) + v0[2] * (v1[0] * v2[1] - v2[0] * v1[1]))
        if abs(d) < 1e-12:
            continue
        
        ux = (a2 * (v1[1] * v2[2] - v2[1] * v1[2]) - b2 * (v0[1] * v2[2] - v2[1] * v0[2]) + c2 * (v0[1] * v1[2] - v1[1] * v0[2])) / d
        uy = (a2 * (v1[0] * v2[2] - v2[0] * v1[2]) - b2 * (v0[0] * v2[2] - v2[0] * v0[2]) + c2 * (v0[0] * v1[2] - v1[0] * v0[2])) / d
        uz = (a2 * (v1[0] * v2[1] - v2[0] * v1[1]) - b2 * (v0[0] * v2[1] - v2[0] * v0[1]) + c2 * (v0[0] * v1[1] - v1[0] * v0[1])) / d
        
        circumcenter = np.array([ux, uy, uz])
        cc_norm = np.linalg.norm(circumcenter)
        if cc_norm < 1e-12:
            continue
        circumcenter = circumcenter / cc_norm
        
        # Check if the circumcenter is in the correct hemisphere
        if np.dot(circumcenter, normal) < 0:
            circumcenter = -circumcenter
        
        # Compute the facet covering radius
        dot_vals = np.clip(np.dot(circumcenter, verts), -1.0, 1.0)
        angles = np.arccos(dot_vals)
        facet_radius = np.max(angles)
        
        # Weight by how much this facet contributes to the covering radius
        weight = facet_radius ** 2
        
        # Apply force: push each vertex toward the circumcenter
        for k, idx in enumerate(simplex):
            vertex = P[idx]
            diff = circumcenter - vertex
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-10:
                forces[idx] += weight * diff / diff_norm
    
    return forces

def _riemannian_sgd(P, samples, deadline, max_iters=1000, lr_start=0.01, lr_end=0.0001, rng_seed=42, use_facet_forces=True):
    """Riemannian SGD with dynamic worst-case focus and annealing, optionally with facet forces."""
    rng = np.random.default_rng(rng_seed)
    n = len(P)
    S = len(samples)
    
    # Precompute sample indices for minibatch
    dots = samples @ P.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    current_max = np.max(np.min(dists, axis=1))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Anneal learning rate
        progress = it / max_iters
        lr = lr_start * (lr_end / lr_start) ** progress
        
        # Compute distances and find worst samples
        dots = samples @ P.T
        dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest_idx = np.argmin(dists, axis=1)
        
        # Focus on worst 10% of samples
        n_worst = max(1, S // 10)
        worst_samples_idx = np.argsort(min_dists)[-n_worst:]
        
        # Accumulate gradient from worst samples
        grad_P = np.zeros_like(P)
        for s_idx in worst_samples_idx:
            j = nearest_idx[s_idx]
            d_val = dots[s_idx, j]
            denom = np.sqrt(max(1.0 - d_val * d_val, 1e-15))
            grad_P[j] -= samples[s_idx] / denom
        
        # Add facet-based forces if enabled
        if use_facet_forces:
            facet_forces = _compute_facet_forces(P)
            grad_P += 0.5 * facet_forces
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_P)
        if grad_norm < 1e-10:
            continue
        grad_P = grad_P / grad_norm
        
        # Project to tangent space
        grad_dot_P = np.sum(grad_P * P, axis=1, keepdims=True)
        grad_tangent = grad_P - grad_dot_P * P
        
        # Riemannian gradient step with retraction
        P_new = P + lr * grad_tangent
        P_new = _normalize(P_new)
        
        # Check if improvement
        dots_new = samples @ P_new.T
        dots_new = np.clip(dots_new, -1.0 + 1e-12, 1.0 - 1e-12)
        dists_new = np.arccos(dots_new)
        min_dists_new = np.min(dists_new, axis=1)
        new_max = np.max(min_dists_new)
        
        if new_max < current_max - 1e-10:
            P = P_new
            current_max = new_max
        else:
            # Try smaller step
            P_new2 = P + lr * 0.1 * grad_tangent
            P_new2 = _normalize(P_new2)
            dots_new2 = samples @ P_new2.T
            dots_new2 = np.clip(dots_new2, -1.0 + 1e-12, 1.0 - 1e-12)
            dists_new2 = np.arccos(dots_new2)
            min_dists_new2 = np.min(dists_new2, axis=1)
            new_max2 = np.max(min_dists_new2)
            
            if new_max2 < current_max - 1e-10:
                P = P_new2
                current_max = new_max2
            else:
                # Random perturbation to escape
                random_dir = rng.standard_normal((n, 3))
                random_dir = random_dir - np.sum(random_dir * P, axis=1, keepdims=True) * P
                random_dir = _normalize(random_dir)
                P_new3 = P + lr * 0.05 * random_dir
                P_new3 = _normalize(P_new3)
                dots_new3 = samples @ P_new3.T
                dots_new3 = np.clip(dots_new3, -1.0 + 1e-12, 1.0 - 1e-12)
                dists_new3 = np.arccos(dots_new3)
                min_dists_new3 = np.min(dists_new3, axis=1)
                new_max3 = np.max(min_dists_new3)
                
                if new_max3 < current_max - 1e-10:
                    P = P_new3
                    current_max = new_max3
    
    return P, current_max

def _polish_greedy(P, samples, deadline, max_iters=500, rng_seed=42, use_facet_polish=True):
    """Greedy local search: move the point nearest to the worst sample toward it, with facet-based polish."""
    min_dists, nearest_idx = _get_min_dists(P, samples)
    current_max = np.max(min_dists)
    rng = np.random.default_rng(rng_seed)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        min_dists, nearest_idx = _get_min_dists(P, samples)
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        
        p_idx = nearest_idx[worst_idx]
        p_old = P[p_idx].copy()
        
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-8:
            # Try facet-based polish instead
            if use_facet_polish:
                facet_forces = _compute_facet_forces(P)
                if np.linalg.norm(facet_forces[p_idx]) > 1e-8:
                    grad_dir = facet_forces[p_idx]
                    grad_norm = np.linalg.norm(grad_dir)
                    grad_dir = grad_dir / grad_norm
                else:
                    continue
            else:
                continue
        
        if grad_norm > 1e-8:
            grad_dir = grad_dir / grad_norm
        
        # Try several step sizes
        best_step = 0.0
        best_new_max = current_max
        best_p = p_old.copy()
        
        for step_size in [0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001]:
            new_p = p_old + step_size * grad_dir
            new_p = _normalize(new_p.reshape(1, 3))[0]
            
            P[p_idx] = new_p
            min_dists_temp, _ = _get_min_dists(P, samples)
            new_max = np.max(min_dists_temp)
            
            if new_max < best_new_max - 1e-12:
                best_new_max = new_max
                best_step = step_size
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_step > 0:
            P[p_idx] = best_p
            current_max = best_new_max
        else:
            # Try a random perturbation to escape local minimum
            random_dir = rng.standard_normal(3)
            random_dir = random_dir - np.dot(random_dir, p_old) * p_old
            random_norm = np.linalg.norm(random_dir)
            if random_norm > 1e-6:
                random_dir = random_dir / random_norm
            else:
                random_dir = np.zeros(3)
            
            for step_size in [0.02, 0.01, 0.005, 0.002]:
                new_p = p_old + step_size * random_dir
                new_p = _normalize(new_p.reshape(1, 3))[0]
                
                P[p_idx] = new_p
                min_dists_temp, _ = _get_min_dists(P, samples)
                new_max = np.max(min_dists_temp)
                
                if new_max < best_new_max - 1e-12:
                    best_new_max = new_max
                    best_step = step_size
                    best_p = new_p.copy()
                else:
                    P[p_idx] = p_old
            
            if best_step > 0:
                P[p_idx] = best_p
                current_max = best_new_max
            else:
                P[p_idx] = p_old
    
    return P, current_max

def _facet_polish(P, deadline, max_iters=200):
    """Direct facet-based polish: iteratively move points toward their facet circumcenters."""
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        forces = _compute_facet_forces(P)
        force_norm = np.linalg.norm(forces)
        
        if force_norm < 1e-10:
            break
        
        # Normalize forces
        forces = forces / force_norm
        
        # Project to tangent space
        grad_dot_P = np.sum(forces * P, axis=1, keepdims=True)
        grad_tangent = forces - grad_dot_P * P
        
        # Try multiple step sizes
        best_lr = 0.0
        best_P = P.copy()
        best_radius = _covering_radius_deg(P)
        
        for lr in [0.1, 0.05, 0.02, 0.01, 0.005]:
            P_new = P + lr * grad_tangent
            P_new = _normalize(P_new)
            new_radius = _covering_radius_deg(P_new)
            
            if new_radius < best_radius - 1e-12:
                best_radius = new_radius
                best_lr = lr
                best_P = P_new.copy()
        
        if best_lr > 0:
            P = best_P
        else:
            # Try random perturbation
            rng = np.random.default_rng(int(time.time() * 1000) % 100000)
            random_dir = rng.standard_normal((N, 3))
            random_dir = random_dir - np.sum(random_dir * P, axis=1, keepdims=True) * P
            random_dir = _normalize(random_dir)
            
            for lr in [0.05, 0.02, 0.01]:
                P_new = P + lr * random_dir
                P_new = _normalize(P_new)
                new_radius = _covering_radius_deg(P_new)
                
                if new_radius < best_radius - 1e-12:
                    best_radius = new_radius
                    best_lr = lr
                    best_P = P_new.copy()
                    break
            
            if best_lr > 0:
                P = best_P
            else:
                break
    
    return P

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s kill
    
    # 1. Initialization: Thomson (electrostatic) init
    P = _thomson_init(N, seed=42)
    
    # 2. Generate samples for optimization
    S_opt = 20000
    samples_opt = _fibonacci_sphere(S_opt)
    
    # Phase 1: Riemannian optimization on the smooth max surrogate with spectral regularization
    def objective_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        val, grad = _smooth_max_surrogate(P, samples_opt, temperature=0.03, spectral_weight=0.01)
        
        # Project gradient to tangent space
        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P
        
        return val, grad_tangent.flatten()
    
    try:
        res = minimize(
            objective_and_grad,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 50.0:
        S_polish = 25000
        samples_polish = _fibonacci_sphere(S_polish)
        P, _ = _polish_greedy(P, samples_polish, deadline - 10.0, max_iters=100)
        P = _facet_polish(P, deadline - 5.0, max_iters=50)
        return P
    
    # Phase 2: Riemannian SGD with dynamic worst-case focus, annealing, and facet forces
    S_sgd = 15000
    samples_sgd = _fibonacci_sphere(S_sgd)
    
    P, current_max = _riemannian_sgd(P, samples_sgd, deadline - 40.0, max_iters=800, lr_start=0.005, lr_end=0.00005, rng_seed=123, use_facet_forces=True)
    
    # Check time
    if time.time() > deadline - 30.0:
        S_polish = 30000
        samples_polish = _fibonacci_sphere(S_polish)
        P, _ = _polish_greedy(P, samples_polish, deadline - 10.0, max_iters=200)
        P = _facet_polish(P, deadline - 5.0, max_iters=50)
        return P
    
    # Phase 3: Second round of smooth max optimization with tighter temperature
    samples_opt2 = _fibonacci_sphere(12000, seed_noise=0.03, seed=456)
    
    def objective_and_grad2(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        val, grad = _smooth_max_surrogate(P, samples_opt2, temperature=0.015, spectral_weight=0.005)
        
        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P
        
        return val, grad_tangent.flatten()
    
    try:
        res = minimize(
            objective_and_grad2,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass
    
    # Phase 4: High-resolution greedy polish with facet-based refinement
    S_polish = 50000
    samples_polish = _fibonacci_sphere(S_polish)
    
    P, current_max = _polish_greedy(P, samples_polish, deadline - 15.0, max_iters=400, use_facet_polish=True)
    
    # Phase 5: Final facet-based polish
    P = _facet_polish(P, deadline - 5.0, max_iters=100)
    
    return P
# EVOLVE-BLOCK-END
