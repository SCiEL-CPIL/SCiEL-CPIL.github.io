import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    time_limit = 145.0  # safety margin before 150s kill
    
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z*z)
    points = np.column_stack([r*np.cos(phi), r*np.sin(phi), z])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # 2. Sample points on sphere for covering evaluation
    M = 10000
    rng = np.random.default_rng(12345)
    samples = rng.normal(size=(M, 3))
    samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
    
    beta1 = 30.0  # soft-max sharpness (over samples)
    beta2 = 30.0  # soft-min sharpness (over points)
    w_repulse = 0.1  # repulsion weight
    
    def covering_force(pts):
        """Returns (score, force) where force is the direction to move pts (tangent)."""
        c = samples @ pts.T  # (M, N)
        c = np.clip(c, -1.0 + 1e-12, 1.0 - 1e-12)
        d = np.arccos(c)  # (M, N)
        
        # soft-min over points for each sample
        neg_beta2_d = -beta2 * d
        max_nb = np.max(neg_beta2_d, axis=1, keepdims=True)
        exp_nb = np.exp(neg_beta2_d - max_nb)
        sum_exp_nb = np.sum(exp_nb, axis=1, keepdims=True)
        w = exp_nb / sum_exp_nb  # (M, N)
        logsumexp_nb = max_nb[:,0] + np.log(sum_exp_nb[:,0])
        d_m = -logsumexp_nb / beta2  # (M,)
        
        # soft-max over samples
        beta1_d = beta1 * d_m
        max_b1 = np.max(beta1_d)
        exp_b1 = np.exp(beta1_d - max_b1)
        sum_exp_b1 = np.sum(exp_b1)
        v = exp_b1 / sum_exp_b1  # (M,)
        score = (max_b1 + np.log(sum_exp_b1)) / beta1
        
        # force = sum_m v_m sum_j w_mj samples_m / denom_mj  (direction toward samples)
        denom = np.sqrt(np.clip(1.0 - c*c, 1e-12, None))  # (M, N)
        weights = (v[:, None] * w) / denom  # (M, N)
        force = weights.T @ samples  # (N, 3)
        
        # project to tangent
        dot = np.sum(force * pts, axis=1, keepdims=True)
        force_t = force - dot * pts
        return score, force_t
    
    def repulsion_force(pts):
        diff = pts[:, None, :] - pts[None, :, :]  # (N, N, 3)
        dist_sq = np.sum(diff*diff, axis=2)  # (N, N)
        np.fill_diagonal(dist_sq, 1.0)
        # force_i = sum_j (p_i - p_j) / (dist_sq + 0.1)^2
        w = 1.0 / (dist_sq + 0.1)**2
        np.fill_diagonal(w, 0.0)
        force = np.sum(diff * w[:, :, None], axis=1)  # (N, 3)
        dot = np.sum(force * pts, axis=1, keepdims=True)
        force_t = force - dot * pts
        return force_t
    
    # Optimization
    step = 0.05
    best_points = points.copy()
    best_score = np.inf
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        score, cf = covering_force(points)
        rf = repulsion_force(points)
        
        # normalize forces
        cf_rms = np.sqrt(np.mean(np.sum(cf*cf, axis=1))) + 1e-12
        rf_rms = np.sqrt(np.mean(np.sum(rf*rf, axis=1))) + 1e-12
        cf = cf / cf_rms
        rf = rf / rf_rms
        total_force = cf + w_repulse * rf
        
        if not np.isfinite(score):
            break
        
        if score < best_score:
            best_score = score
            best_points = points.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        new_points = points + step * total_force
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        points = new_points
        
        step *= 0.9995
        if step < 1e-5:
            break
    
    # Final polish with smaller step
    points = best_points.copy()
    step = 0.01
    end_polish = min(time.time() + 5.0, start_time + time_limit)
    while time.time() < end_polish:
        score, cf = covering_force(points)
        rf = repulsion_force(points)
        cf_rms = np.sqrt(np.mean(np.sum(cf*cf, axis=1))) + 1e-12
        rf_rms = np.sqrt(np.mean(np.sum(rf*rf, axis=1))) + 1e-12
        total_force = cf/cf_rms + w_repulse * rf/rf_rms
        if not np.isfinite(score):
            break
        if score < best_score:
            best_score = score
            best_points = points.copy()
        new_points = points + step * total_force
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        points = new_points
        step *= 0.999
    
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
