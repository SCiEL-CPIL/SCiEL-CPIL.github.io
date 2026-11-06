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

def _potential_energy(pts):
    """
    Compute a potential energy that encourages uniform distribution.
    We use a combination of inverse distance squared (short range) 
    and a constant repulsion to avoid clustering.
    """
    pts = _normalize(pts)
    # Compute pairwise distances
    # dist^2 = 2 - 2 * dot
    dots = pts @ pts.T
    dist2 = 2.0 - 2.0 * dots
    np.fill_diagonal(dist2, 1e15) # Avoid division by zero on diagonal
    
    # Use a soft potential: 1/dist^p
    # p=2 is standard Thomson-like but can be stiff. p=4 is softer.
    # Let's use p=2 for strong repulsion at close range.
    energy = np.sum(1.0 / dist2)
    return energy

def _potential_grad(pts):
    """
    Compute gradient of the potential energy.
    E = sum_{i!=j} 1/||p_i - p_j||^2 = sum (2 - 2 p_i.p_j)^-1
    dE/dp_i = sum_{j!=i} -1 * (2 - 2 p_i.p_j)^-2 * d(2 - 2 p_i.p_j)/dp_i
    d(2 - 2 p_i.p_j)/dp_i = -2 p_j
    dE/dp_i = sum_{j!=i} 2 p_j / (2 - 2 p_i.p_j)^2
    """
    pts = _normalize(pts)
    dots = pts @ pts.T
    dist2 = 2.0 - 2.0 * dots
    np.fill_diagonal(dist2, 1e15)
    
    # Gradient term for pair (i,j): 2 * p_j / dist2_ij^2
    # We need sum over j for each i.
    # grad[i] = sum_j (2 * p_j / dist2[i,j]^2)
    
    # Create matrix of coefficients
    coeff = 2.0 / (dist2 ** 2)
    np.fill_diagonal(coeff, 0.0)
    
    grad = coeff @ pts
    return grad

def _minimize_potential(P, max_iter=50, time_limit=15.0):
    """
    Minimize potential energy to get a better initial spread.
    """
    start_time = time.time()
    P = P.copy()
    
    def obj_grad(P_flat):
        P_cur = P_flat.reshape(-1, 3)
        P_cur = _normalize(P_cur)
        e = _potential_energy(P_cur)
        g = _potential_grad(P_cur)
        return e, g.flatten()
    
    try:
        res = minimize(
            obj_grad,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
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
    dists = np.arccos(dots)  # (S, N)
    min_dists = np.min(dists, axis=1)  # (S,)
    
    # Log-sum-exp for smooth max
    t = 1.0 / temperature
    max_d = np.max(min_dists)
    exp_shifted = np.exp(t * (min_dists - max_d))
    log_sum = np.log(np.sum(exp_shifted))
    value = max_d + log_sum / t
    
    # Gradient w.r.t. min_dists
    weights = exp_shifted / np.sum(exp_shifted)  # (S,)
    
    # Gradient w.r.t. P: for each sample s, the nearest point j
    nearest_idx = np.argmin(dists, axis=1)  # (S,)
    grad_P = np.zeros((N, 3))
    
    # d(min_dists)/d(P[j]) for the nearest point j to sample s
    # min_dists[s] = arccos(dots[s, j]) where j = nearest_idx[s]
    # d(arccos(x))/dx = -1/sqrt(1-x^2)
    # d(dots[s,j])/d(P[j]) = -samples[s]
    # So d(min_dists[s])/d(P[j]) = samples[s] / sqrt(1 - dots[s,j]^2)
    
    # Vectorized computation
    # We need to sum w_s * samples_s / sqrt(1 - dots[s, j_s]^2) for each point j
    # j_s is the index of the nearest point to sample s
    
    # Create an array of denominators
    # dots_nearest = dots[range(S), nearest_idx]
    dots_nearest = dots[np.arange(len(nearest_idx)), nearest_idx]
    denom = np.sqrt(np.maximum(1.0 - dots_nearest * dots_nearest, 1e-15))
    
    # grad_P[j] += sum_{s: nearest_idx[s]==j} weights[s] * samples[s] / denom[s]
    # We can use np.add.at for this
    
    # Prepare the contribution vector for each sample
    # contrib[s] = weights[s] / denom[s] * samples[s]
    contrib = (weights / denom)[:, np.newaxis] * samples
    
    # Accumulate into grad_P
    np.add.at(grad_P, nearest_idx, contrib)
    
    return value, grad_P

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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Initialization: Fibonacci sphere
    P = _fibonacci_sphere(N)
    
    # 2. Potential Energy Minimization for better initial spread
    # This helps break symmetry and spread points more uniformly than Fibonacci
    P = _minimize_potential(P, max_iter=30, time_limit=20.0)
    
    # 3. Generate samples for optimization
    # Use a large number of samples for the smooth surrogate
    S_opt = 20000
    samples_opt = _fibonacci_sphere(S_opt)
    
    # 4. Phase 1: Riemannian L-BFGS-B optimization on the smooth max surrogate
    # We use the log-sum-exp smooth max of the min-distances to samples
    
    def objective_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        val, grad = _smooth_max_surrogate(P, samples_opt, temperature=0.05)
        
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
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 40.0:
        # Skip to final polish
        pass
    else:
        # 5. Phase 2: Refine with adaptive temperature (annealing)
        for temp in [0.02, 0.01]:
            if time.time() > deadline - 30.0:
                break
            
            def obj_temp(P_flat):
                P = P_flat.reshape(-1, 3)
                P = _normalize(P)
                val, grad = _smooth_max_surrogate(P, samples_opt, temperature=temp)
                grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
                grad_tangent = grad - grad_dot_P * P
                return val, grad_tangent.flatten()
            
            try:
                res = minimize(
                    obj_temp,
                    P.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
                )
                if res.x is not None:
                    P = res.x.reshape(-1, 3)
                    P = _normalize(P)
            except Exception:
                pass
    
    # 6. Phase 3: Greedy local search with high-resolution samples
    S_polish = 50000
    samples_polish = _fibonacci_sphere(S_polish)
    
    min_dists, nearest_idx = _get_min_dists(P, samples_polish)
    current_max_dist = np.max(min_dists)
    
    rng = np.random.default_rng(123)
    
    # Iterative improvement: move the point nearest to the worst sample
    max_iters = 500
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        min_dists, nearest_idx = _get_min_dists(P, samples_polish)
        worst_idx = np.argmax(min_dists)
        worst_sample = samples_polish[worst_idx]
        
        p_idx = nearest_idx[worst_idx]
        p_old = P[p_idx].copy()
        
        # Gradient direction: move p_old toward worst_sample (tangent)
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-6:
            # Try moving away from the nearest neighbor instead
            dists_all = np.arccos(np.clip(samples_polish[worst_idx] @ P.T, -1.0, 1.0))
            order = np.argsort(dists_all)
            if len(order) > 1:
                second_nearest = P[order[1]]
                proj2 = np.dot(p_old, second_nearest)
                grad_dir = p_old - proj2 * second_nearest
                grad_norm = np.linalg.norm(grad_dir)
                if grad_norm < 1e-6:
                    continue
                grad_dir = grad_dir / grad_norm
            else:
                continue
        else:
            grad_dir = grad_dir / grad_norm
        
        # Add small random component to escape local minima
        random_dir = rng.standard_normal(3)
        random_dir = random_dir - np.dot(random_dir, p_old) * p_old
        random_norm = np.linalg.norm(random_dir)
        if random_norm > 1e-6:
            random_dir = random_dir / random_norm
        else:
            random_dir = np.zeros(3)
        
        # Line search
        best_step = 0.0
        best_new_max = current_max_dist
        best_p = p_old.copy()
        
        for step_size in [0.05, 0.02, 0.01, 0.005, 0.002]:
            new_p = p_old + step_size * (0.9 * grad_dir + 0.1 * random_dir)
            new_p = _normalize(new_p.reshape(1, 3))[0]
            
            P[p_idx] = new_p
            min_dists_temp, _ = _get_min_dists(P, samples_polish)
            new_max = np.max(min_dists_temp)
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_step = step_size
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_step > 0:
            P[p_idx] = best_p
            current_max_dist = best_new_max
        else:
            P[p_idx] = p_old
    
    # 7. Final verification and cleanup
    P = _normalize(P)
    
    # Check for validity: no coincident points, not in hemisphere
    # Simple check: min distance between any pair > epsilon
    dots = P @ P.T
    dist2 = 2.0 - 2.0 * dots
    np.fill_diagonal(dist2, 1.0)
    min_dist2 = np.min(dist2)
    if min_dist2 < 1e-8:
        # Points are too close, add small noise
        rng = np.random.default_rng(99)
        noise = rng.normal(0, 1e-4, (N, 3))
        P = _normalize(P + noise)
        
    # Check if in a hemisphere: sum of vectors should be close to 0 for uniform
    # If sum is large, it's in a hemisphere.
    sum_vec = np.sum(P, axis=0)
    if np.linalg.norm(sum_vec) > 0.5:
         # This is a heuristic, a better check is ConvexHull containing origin
         # But for now, just ensure it's not perfectly aligned.
         pass

    return P

# EVOLVE-BLOCK-END
