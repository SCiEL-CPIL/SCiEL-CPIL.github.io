"""Program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
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

def _smooth_max_surrogate(P, samples, temperature=0.05):
    """
    Compute a smooth approximation of the max distance using log-sum-exp.
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
    
    return value, grad_P

def _spectral_uniformity_energy(P, max_l=10):
    """
    Compute spectral energy: variance of pair correlation function via spherical harmonics.
    This promotes uniformity by penalizing large-scale clustering.
    """
    P = _normalize(P)
    
    from scipy.special import lpmv
    
    energy = 0.0
    for l in range(1, max_l + 1):
        dots = P @ P.T
        P_l = lpmv(0, l, dots)
        
        mult_sum = np.sum(P_l)
        C_l = (2 * l + 1) / (4 * np.pi) * mult_sum
        
        energy += C_l * C_l
    
    return energy

def _riemannian_sgd(P, samples, deadline, max_iters=1000, lr_start=0.01, lr_end=0.0001, rng_seed=42):
    """Riemannian SGD with dynamic worst-case focus and annealing - vectorized."""
    rng = np.random.default_rng(rng_seed)
    n = len(P)
    S = len(samples)
    
    current_max = np.max(np.min(np.arccos(np.clip(samples @ P.T, -1, 1)), axis=1))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        progress = it / max_iters
        lr = lr_start * (lr_end / lr_start) ** progress
        
        dots = samples @ P.T
        dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest_idx = np.argmin(dists, axis=1)
        
        n_worst = max(1, S // 10)
        worst_samples_idx = np.argsort(min_dists)[-n_worst:]
        
        # Vectorized gradient accumulation
        worst_samples = samples[worst_samples_idx]
        worst_nearest = nearest_idx[worst_samples_idx]
        worst_dots = dots[worst_samples_idx, worst_nearest]
        denom = np.sqrt(np.maximum(1.0 - worst_dots * worst_dots, 1e-15))
        
        grad_P = np.zeros_like(P)
        np.add.at(grad_P, worst_nearest, -worst_samples / denom[:, np.newaxis])
        
        grad_norm = np.linalg.norm(grad_P)
        if grad_norm < 1e-10:
            continue
        grad_P = grad_P / grad_norm
        
        grad_dot_P = np.sum(grad_P * P, axis=1, keepdims=True)
        grad_tangent = grad_P - grad_dot_P * P
        
        P_new = P + lr * grad_tangent
        P_new = _normalize(P_new)
        
        dots_new = samples @ P_new.T
        dots_new = np.clip(dots_new, -1.0 + 1e-12, 1.0 - 1e-12)
        new_max = np.max(np.arccos(dots_new).min(axis=1))
        
        if new_max < current_max - 1e-10:
            P = P_new
            current_max = new_max
        else:
            P_new2 = P + lr * 0.1 * grad_tangent
            P_new2 = _normalize(P_new2)
            dots_new2 = samples @ P_new2.T
            dots_new2 = np.clip(dots_new2, -1.0 + 1e-12, 1.0 - 1e-12)
            new_max2 = np.max(np.arccos(dots_new2).min(axis=1))
            
            if new_max2 < current_max - 1e-10:
                P = P_new2
                current_max = new_max2
            else:
                random_dir = rng.standard_normal((n, 3))
                random_dir = random_dir - np.sum(random_dir * P, axis=1, keepdims=True) * P
                random_dir = _normalize(random_dir)
                P_new3 = P + lr * 0.05 * random_dir
                P_new3 = _normalize(P_new3)
                dots_new3 = samples @ P_new3.T
                dots_new3 = np.clip(dots_new3, -1.0 + 1e-12, 1.0 - 1e-12)
                new_max3 = np.max(np.arccos(dots_new3).min(axis=1))
                
                if new_max3 < current_max - 1e-10:
                    P = P_new3
                    current_max = new_max3
    
    return P, current_max

def _polish_greedy(P, samples, deadline, max_iters=500, rng_seed=42):
    """Greedy local search: move the point nearest to the worst sample toward it."""
    min_dists, nearest_idx = _get_min_dists(P, samples)
    current_max = np.max(min_dists)
    rng = np.random.default_rng(rng_seed)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        
        p_idx = nearest_idx[worst_idx]
        p_old = P[p_idx].copy()
        
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-8:
            # Try random perturbation
            random_dir = rng.standard_normal(3)
            random_dir = random_dir - np.dot(random_dir, p_old) * p_old
            random_norm = np.linalg.norm(random_dir)
            if random_norm > 1e-6:
                random_dir = random_dir / random_norm
            else:
                continue
        else:
            grad_dir = grad_dir / grad_norm
        
        best_step = 0.0
        best_new_max = current_max
        best_p = p_old.copy()
        
        # Adaptive step sizes based on current gap
        gap = current_max - np.percentile(min_dists, 90)
        step_sizes = [0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for step_size in step_sizes:
            new_p = p_old + step_size * grad_dir
            new_p = _normalize(new_p.reshape(1, 3))[0]
            
            P[p_idx] = new_p
            dots_temp = samples @ P.T
            dots_temp = np.clip(dots_temp, -1.0 + 1e-12, 1.0 - 1e-12)
            dists_temp = np.arccos(dots_temp)
            min_dists_temp = np.min(dists_temp, axis=1)
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
            # Recompute min_dists and nearest_idx for next iteration
            dots = samples @ P.T
            dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
            dists = np.arccos(dots)
            min_dists = np.min(dists, axis=1)
            nearest_idx = np.argmin(dists, axis=1)
        else:
            # Random perturbation to escape local minimum
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
                dots_temp = samples @ P.T
                dots_temp = np.clip(dots_temp, -1.0 + 1e-12, 1.0 - 1e-12)
                dists_temp = np.arccos(dots_temp)
                min_dists_temp = np.min(dists_temp, axis=1)
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
                dots = samples @ P.T
                dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
                dists = np.arccos(dots)
                min_dists = np.min(dists, axis=1)
                nearest_idx = np.argmin(dists, axis=1)
            else:
                P[p_idx] = p_old
    
    return P, current_max

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Initialization: Thomson (electrostatic) init
    P = _thomson_init(N, seed=42)
    
    # 2. Generate samples for optimization
    S_opt = 15000
    samples_opt = _fibonacci_sphere(S_opt)
    
    # Phase 1: Riemannian optimization on the smooth max surrogate
    def objective_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        val, grad = _smooth_max_surrogate(P, samples_opt, temperature=0.03)
        
        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P
        
        return val, grad_tangent.flatten()
    
    try:
        res = minimize(
            objective_and_grad,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 45.0:
        S_polish = 20000
        samples_polish = _fibonacci_sphere(S_polish)
        P, _ = _polish_greedy(P, samples_polish, deadline - 5.0, max_iters=150)
        return P
    
    # Phase 2: Riemannian SGD with dynamic worst-case focus and annealing
    S_sgd = 12000
    samples_sgd = _fibonacci_sphere(S_sgd)
    
    P, current_max = _riemannian_sgd(P, samples_sgd, deadline - 35.0, max_iters=600, lr_start=0.005, lr_end=0.00005, rng_seed=123)
    
    # Check time
    if time.time() > deadline - 25.0:
        S_polish = 25000
        samples_polish = _fibonacci_sphere(S_polish)
        P, _ = _polish_greedy(P, samples_polish, deadline - 5.0, max_iters=200)
        return P
    
    # Phase 3: Second round of smooth max optimization with tighter temperature
    samples_opt2 = _fibonacci_sphere(10000, seed_noise=0.02, seed=456)
    
    def objective_and_grad2(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        val, grad = _smooth_max_surrogate(P, samples_opt2, temperature=0.015)
        
        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P
        
        return val, grad_tangent.flatten()
    
    try:
        res = minimize(
            objective_and_grad2,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass
    
    # Phase 4: High-resolution greedy polish
    S_polish = 40000
    samples_polish = _fibonacci_sphere(S_polish)
    
    P, current_max = _polish_greedy(P, samples_polish, deadline - 5.0, max_iters=500)
    
    return P
# EVOLVE-BLOCK-END
