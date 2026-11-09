import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin: evaluator kills at 150s

    def compute_dists(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        return dists

    def get_unique_dists(dists):
        iu = np.triu_indices(8, k=1)
        return dists[iu]

    def compute_score_log(points):
        """Exact score using true max diameter."""
        dists = compute_dists(points)
        d_pairs = get_unique_dists(dists)
        if np.min(d_pairs) < 1e-12:
            return -np.inf
        log_prod = np.sum(np.log(d_pairs))
        D = np.max(d_pairs)
        if D < 1e-12:
            return -np.inf
        log_D = np.log(D)
        score_log = log_prod - 28.0 * log_D
        return score_log

    def compute_soft_diameter_log(d_pairs, alpha):
        """Smooth approximation of log(max(d_pairs)) using log-sum-exp."""
        # log(sum(exp(alpha * d))) / alpha approximates log(max(d)) as alpha->inf
        # For numerical stability: max_d + log(sum(exp(alpha*(d - max_d)))) / alpha
        max_d = np.max(d_pairs)
        if max_d < 1e-12:
            return np.log(1e-12)
        shifted = alpha * (d_pairs - max_d)
        # Clip to avoid overflow
        shifted = np.clip(shifted, -100, 100)
        log_sum = np.log(np.sum(np.exp(shifted)))
        return max_d + log_sum / alpha

    def compute_score_soft(points, alpha):
        """Score using soft diameter surrogate."""
        dists = compute_dists(points)
        d_pairs = get_unique_dists(dists)
        if np.min(d_pairs) < 1e-12:
            return -np.inf
        log_prod = np.sum(np.log(d_pairs))
        log_D_soft = compute_soft_diameter_log(d_pairs, alpha)
        score_log = log_prod - 28.0 * log_D_soft
        return score_log

    def objective_soft(params, alpha):
        points = params.reshape(8, 2)
        sl = compute_score_soft(points, alpha)
        if np.isinf(sl):
            return 1e10
        return -sl

    def gradient_soft(params, alpha):
        """Analytical gradient of the soft objective."""
        points = params.reshape(8, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        d_pairs = get_unique_dists(dists)
        
        if np.min(d_pairs) < 1e-12:
            return np.zeros_like(params)
        
        # Gradient of log_prod = sum log(d_ij) w.r.t. points
        # d/dp_k log(d_ij) = (p_i - p_j) / d_ij^2 * (delta_{k,i} - delta_{k,j})
        grad_log_prod = np.zeros((8, 2))
        iu = np.triu_indices(8, k=1)
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            d = d_pairs[idx]
            if d < 1e-12:
                continue
            # d/dp_i log(d_ij) = (p_i - p_j) / d^2
            # d/dp_j log(d_ij) = -(p_i - p_j) / d^2
            factor = 1.0 / d
            grad_log_prod[i] += factor * (points[i] - points[j])
            grad_log_prod[j] -= factor * (points[i] - points[j])
        
        # Gradient of -28 * log_D_soft
        # log_D_soft = max_d + log(sum(exp(alpha*(d-max_d))))/alpha
        # Let S = sum(exp(alpha*(d - max_d)))
        # d log_D_soft / d d_k = alpha * exp(alpha*(d_k - max_d)) / S for k != argmax
        # For k = argmax: d log_D_soft / d d_k = 1 - (1/S) * sum_{m != k} exp(alpha*(d_m - max_d))
        # Simplified: weights w_k = exp(alpha*(d_k - max_d)) / S
        # Then d log_D_soft / d d_k = w_k, except for argmax where it's 1 - sum_{m!=argmax} w_m = w_argmax
        # Actually: d log_D_soft / d d_k = w_k for all k (since the max_d term differentiates to 1 for argmax and 0 otherwise,
        # but the log-sum-exp part gives w_k, and they combine to w_k for all k)
        
        max_d = np.max(d_pairs)
        shifted = alpha * (d_pairs - max_d)
        shifted = np.clip(shifted, -100, 100)
        exp_shifted = np.exp(shifted)
        S = np.sum(exp_shifted)
        weights = exp_shifted / S  # These are the softmax weights
        
        # Gradient of log_D_soft w.r.t. each d_pairs[k] is weights[k]
        grad_log_D = np.zeros((8, 2))
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            d = d_pairs[idx]
            if d < 1e-12:
                continue
            w = weights[idx]
            # d d_ij / d p_i = (p_i - p_j) / d
            # d d_ij / d p_j = -(p_i - p_j) / d
            factor = w / d
            grad_log_D[i] += factor * (points[i] - points[j])
            grad_log_D[j] -= factor * (points[i] - points[j])
        
        # Total gradient of objective = -grad_log_prod + 28 * grad_log_D
        grad = -grad_log_prod + 28.0 * grad_log_D
        return grad.flatten()

    def objective_with_grad(params, alpha):
        f = objective_soft(params, alpha)
        g = gradient_soft(params, alpha)
        return f, g

    def polish_exact(points, max_iter=50000):
        """Polish using exact objective with Nelder-Mead."""
        def obj(params):
            p = params.reshape(8, 2)
            sl = compute_score_log(p)
            if np.isinf(sl):
                return 1e10
            return -sl
        res = minimize(obj, points.flatten(), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-14, 'fatol': 1e-14, 'niter': max_iter})
        if res.fun < 1e9:
            return res.x.reshape(8, 2)
        return points

    def polish_bfgs(points, alpha_start=10.0, alpha_end=1000.0, n_steps=20):
        """Polish using BFGS with annealed alpha for soft diameter."""
        current = points.copy()
        for i in range(n_steps):
            alpha = alpha_start * (alpha_end / alpha_start) ** (i / max(n_steps - 1, 1))
            elapsed = time.time() - start_time
            if elapsed > time_limit - 25:
                break
            try:
                res = minimize(objective_with_grad, current.flatten(),
                               method='L-BFGS-B', jac=True,
                               options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
                if res.fun < 1e9:
                    current = res.x.reshape(8, 2)
            except Exception:
                pass
        return current

    best_points = None
    best_score = -np.inf

    # Strategy 1: BFGS with annealed soft diameter from multiple starts
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    inits = []
    
    # Regular octagon
    angles = np.arange(8) * (np.pi / 4)
    oct_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(oct_pts)
    
    # D4 symmetric family: two radii and rotation
    for r1 in [0.5, 0.7, 0.85, 0.95, 1.0, 1.1, 1.3, 1.8]:
        for r2 in [0.5, 0.7, 0.85, 0.95, 1.0, 1.1, 1.3, 1.8]:
            for a in [0.0, np.pi/8, np.pi/16, -np.pi/8]:
                angles1 = a + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                angles2 = a + np.pi/4 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
                pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
                inits.append(np.vstack((pts1, pts2)))
    
    # Random perturbations of regular octagon
    for _ in range(10):
        pert = rng.normal(0, 0.3, size=(8, 2))
        inits.append(oct_pts + pert)
    
    # Random points in unit disk
    for _ in range(15):
        angles_r = rng.uniform(0, 2*np.pi, size=8)
        radii = rng.uniform(0.3, 1.5, size=8)
        inits.append(np.column_stack((radii * np.cos(angles_r), radii * np.sin(angles_r))))
    
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            break
        try:
            # First: BFGS with annealed alpha
            polished = polish_bfgs(init, alpha_start=5.0, alpha_end=500.0, n_steps=15)
            sl = compute_score_log(polished)
            if sl > best_score:
                best_score = sl
                best_points = polished.copy()
        except Exception:
            pass
    
    # Strategy 2: Exact Nelder-Mead polish from best found
    if best_points is not None:
        elapsed = time.time() - start_time
        if elapsed < time_limit - 20:
            try:
                polished = polish_exact(best_points, max_iter=30000)
                sl = compute_score_log(polished)
                if sl > best_score:
                    best_score = sl
                    best_points = polished.copy()
            except Exception:
                pass
    
    # Strategy 3: More BFGS iterations from best with higher alpha
    if best_points is not None:
        elapsed = time.time() - start_time
        if elapsed < time_limit - 15:
            try:
                polished = polish_bfgs(best_points, alpha_start=50.0, alpha_end=2000.0, n_steps=10)
                sl = compute_score_log(polished)
                if sl > best_score:
                    best_score = sl
                    best_points = polished.copy()
            except Exception:
                pass
    
    # Strategy 4: Final exact polish
    if best_points is not None:
        elapsed = time.time() - start_time
        if elapsed < time_limit - 10:
            try:
                polished = polish_exact(best_points, max_iter=20000)
                sl = compute_score_log(polished)
                if sl > best_score:
                    best_score = sl
                    best_points = polished.copy()
            except Exception:
                pass

    if best_points is None:
        angles = np.arange(8) * (np.pi / 4)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    # Normalize to diameter 2
    dists = compute_dists(best_points)
    D = np.max(dists)
    if D > 0:
        best_points = best_points * (2.0 / D)

    # Verify
    dists = compute_dists(best_points)
    iu = np.triu_indices(8, k=1)
    if np.min(dists[iu]) < 1e-6 or np.any(np.abs(best_points) >= 1e6):
        angles = np.arange(8) * (np.pi / 4)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_points = best_points * (2.0 / (2 * np.cos(np.pi/8)))

    return best_points.tolist()
