"""Combined program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize
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

def _compute_max_dist(P, samples):
    """
    Compute the maximum angular distance from any sample to its nearest point in P.
    Returns: max_dist, min_dists_array, nearest_indices
    """
    dots = samples @ P.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    nearest = np.argmin(dists, axis=1)
    max_dist = np.max(min_dists)
    return max_dist, min_dists, nearest

def _soft_max_loss(P_flat, samples, lam=10.0):
    """
    Smooth approximation of the max covering distance using log-sum-exp.
    Higher lam makes it closer to the true max.
    """
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms
    
    dots = samples @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Soft max of distances
    # max(d) approx 1/lam * log(sum(exp(lam*d)))
    # We want to minimize the max, so we minimize this value.
    # To avoid overflow, subtract max
    m = np.max(dists, axis=1)
    val = np.log(np.sum(np.exp(lam * (dists - m[:, None])), axis=1)) / lam + m
    loss = np.mean(val)
    
    # Gradient calculation
    # d/dP_i sum_j (1/lam log sum_k exp(lam d_jk))
    # This is complex. Let's use a simpler weighted gradient based on the soft-max weights.
    # w_jk = exp(lam * (d_jk - m_j)) / sum_k exp(lam * (d_jk - m_j))
    # grad_d_jk = w_jk
    # grad_P_i = - sum_j w_ji * (sample_j - (sample_j.P_i)P_i) / sqrt(1 - (sample_j.P_i)^2)
    
    exp_terms = np.exp(lam * (dists - m[:, None]))
    sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
    weights = exp_terms / sum_exp # shape (S, N)
    
    # Only need gradient for the points.
    # grad_P[i] = - sum_j weights[j, i] * (samples[j] - dot_ji * P[i]) / sqrt(1 - dot_ji^2)
    # This is a large matrix multiplication.
    # Let's compute it in chunks or use efficient numpy ops.
    
    # dots is (S, N)
    # weights is (S, N)
    # samples is (S, 3)
    # P is (N, 3)
    
    # term_ji = weights[j, i] * (samples[j] - dots[j, i] * P[i]) / sqrt(1 - dots[j, i]^2)
    
    # To save memory and time, we can compute the gradient directly.
    # grad_P = - (samples.T * weights) / denom - ... this is tricky because P[i] depends on i.
    
    # Let's use the explicit loop or vectorized approach.
    # Vectorized:
    # We want sum_j weights[j, i] * (samples[j] - dots[j, i] * P[i]) / denom[j, i]
    
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots**2)) # (S, N)
    
    # part1: sum_j weights[j, i] * samples[j]
    # This is weights.T @ samples -> (N, 3)
    grad_part1 = weights.T @ samples
    
    # part2: sum_j weights[j, i] * dots[j, i] * P[i] / denom[j, i]
    # coeff_ji = weights[j, i] * dots[j, i] / denom[j, i]
    coeff = weights * dots / denom # (S, N)
    sum_coeff = np.sum(coeff, axis=0) # (N,)
    grad_part2 = sum_coeff[:, None] * P # (N, 3)
    
    grad_P = -(grad_part1 - grad_part2)
    
    # Project to tangent space
    grad_dot = np.sum(grad_P * Pn, axis=1, keepdims=True)
    grad_tan = grad_P - grad_dot * Pn
    
    grad_flat = (grad_tan / norms).flatten()
    
    return loss, grad_flat

def _fast_local_search(P, samples, deadline, max_iters=500):
    """
    Fast local search:
    1. Find worst sample.
    2. Identify nearest point.
    3. Line search along the great circle towards the worst sample.
    4. If stuck, apply random perturbation.
    """
    n = len(P)
    rng = np.random.default_rng(42)
    
    current_max, min_dists, nearest = _compute_max_dist(P, samples)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        p_idx = nearest[worst_idx]
        p_old = P[p_idx].copy()
        
        # Direction to move: toward worst_sample, projected to tangent space
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-8:
            # Random perturbation if stuck
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm < 1e-8:
                continue
            grad_dir = rand / rand_norm
        else:
            grad_dir = grad_dir / grad_norm
        
        best_step = 0.0
        best_max = current_max
        best_p = None
        
        # Adaptive step size search
        # Try a range of steps
        steps = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        
        for step in steps:
            new_p = p_old + step * grad_dir
            new_p = _normalize(new_p.reshape(1, 3))[0]
            P[p_idx] = new_p
            
            new_max, _, _ = _compute_max_dist(P, samples)
            
            if new_max < best_max:
                best_max = new_max
                best_step = step
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_p is not None:
            P[p_idx] = best_p
            current_max = best_max
            _, min_dists, nearest = _compute_max_dist(P, samples)
        else:
            # If no improvement, try a small random kick to escape local minima
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm > 1e-8:
                rand /= rand_norm
            P[p_idx] = _normalize((p_old + 0.05 * rand).reshape(1, 3))[0]
            _, min_dists, nearest = _compute_max_dist(P, samples)
            
    return P, current_max

def construct():
    start_time = time.time()
    # Safety margin: 5s before the 150s limit
    deadline = start_time + 145.0
    
    best_P = None
    best_score = float('inf')
    
    # Restart parameters
    restarts = [
        (0.05, 42),
        (0.10, 123),
        (0.0, 777) # Pure Fibonacci
    ]
    
    # Determine budget per restart
    total_time_budget = deadline - start_time
    time_per_restart = total_time_budget / len(restarts)
    
    for r_idx, (noise, seed) in enumerate(restarts):
        current_deadline = start_time + (r_idx + 1) * time_per_restart
        # Ensure we don't exceed the absolute deadline
        current_deadline = min(current_deadline, deadline - 5.0)
        
        if time.time() > current_deadline - 2.0:
            break
            
        # 1. Initialize
        P = _fibonacci_sphere(N, seed_noise=noise, seed=seed)
        
        # 2. Optimization Phase
        # Use a moderate number of samples for optimization (20k)
        S_opt = 20000
        samples_opt = _fibonacci_sphere(S_opt, seed=seed+100)
        
        # L-BFGS optimization
        opt_deadline = current_deadline - 15.0 # Reserve 15s for local search
        
        def time_callback(xk):
            if time.time() > opt_deadline:
                raise ValueError("Time limit approaching")
        
        try:
            res = minimize(
                lambda x: _soft_max_loss(x, samples_opt, lam=10.0),
                P.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False},
                callback=time_callback
            )
            if res.x is not None:
                P = _normalize(res.x.reshape(-1, 3))
        except Exception:
            pass
            
        # 3. Local Search Phase
        # Use a larger sample set for local search (50k)
        S_local = 50000
        samples_local = _fibonacci_sphere(S_local, seed=seed+200)
        
        local_deadline = current_deadline - 2.0
        
        if time.time() < local_deadline:
            P, current_max = _fast_local_search(P, samples_local, local_deadline, max_iters=200)
            
            # Evaluate with a very dense set to get a reliable score
            S_eval = 100000
            samples_eval = _fibonacci_sphere(S_eval, seed=seed+300)
            current_score, _, _ = _compute_max_dist(P, samples_eval)
            
            if current_score < best_score:
                best_score = current_score
                best_P = P.copy()
                
    # 4. Final Polish
    # If we have time left, do a final high-precision local search on the best result
    if best_P is not None and time.time() < deadline - 5.0:
        S_final = 100000
        samples_final = _fibonacci_sphere(S_final, seed=9999)
        
        # Run a few more iterations with the densest sample set
        P_best, _ = _fast_local_search(best_P, samples_final, deadline - 1.0, max_iters=100)
        
        # Re-evaluate
        final_score, _, _ = _compute_max_dist(P_best, samples_final)
        
        if final_score < best_score:
            best_P = P_best
            best_score = final_score

    return _normalize(best_P)

# EVOLVE-BLOCK-END
