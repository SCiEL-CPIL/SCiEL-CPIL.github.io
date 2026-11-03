import numpy as np
import time
import sys
from scipy.optimize import minimize
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def _compute_convolution(h):
    """Compute full autoconvolution of h."""
    return np.convolve(h, h, mode='full')

def _find_dominant_lags(h, conv, n_lags=3):
    """Find the indices of the largest convolution peaks."""
    sorted_idx = np.argsort(conv)[::-1]
    return sorted_idx[:n_lags]

def _trust_region_subproblem(h, k_star, delta_trust=0.1, max_iter=50):
    """
    Solve a trust-region subproblem that maximizes ||h||^2 subject to
    conv[k_star] <= M (linear in h for fixed support around k_star).
    
    For fixed k_star, conv[k_star] = sum_j h[j]*h[k_star-j]
    This is quadratic in h, but if we fix the support (indices with h>0),
    the constraint becomes a quadratic form. We use L-BFGS-B with the
    active set of indices contributing to the peak.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    if np.sum(h) < 1e-15:
        return h
    
    h /= np.sum(h)
    conv = _compute_convolution(h)
    M = conv[k_star]
    
    # Identify active set: indices j where h[j] > 0 and h[k_star-j] > 0 (if valid)
    active_mask = np.zeros(N, dtype=bool)
    for j in range(N):
        if h[j] > 1e-10:
            active_mask[j] = True
            j2 = k_star - j
            if 0 <= j2 < N and h[j2] > 1e-10:
                active_mask[j2] = True
    
    if not np.any(active_mask):
        active_mask = h > 1e-10
    
    # Get active indices
    active_idx = np.where(active_mask)[0]
    n_active = len(active_idx)
    
    if n_active < 2:
        return h
    
    # Extract active variables
    h_active = h[active_idx].copy()
    
    # Build the quadratic form for conv[k_star] restricted to active set
    # conv[k_star] = sum_j h[j]*h[k_star-j]
    # For j in active_idx, k_star-j must also be in active_idx or 0
    Q = np.zeros((n_active, n_active))
    for i, j in enumerate(active_idx):
        j2 = k_star - j
        if 0 <= j2 < N:
            # Find index of j2 in active_idx
            pos = np.where(active_idx == j2)[0]
            if len(pos) > 0:
                Q[i, pos[0]] += 1.0
                if pos[0] != i:
                    Q[pos[0], i] += 1.0  # Symmetric
    
    # Objective: maximize ||h_active||^2 subject to Q-form <= M and sum(h_active) = 1 (approximately)
    # Use trust region: ||h_new - h_old||^2 <= delta_trust^2
    
    def objective(x):
        """Negative L2 norm squared (we minimize negative)."""
        return -np.sum(x**2)
    
    def gradient(x):
        return -2.0 * x
    
    def conv_constraint(x):
        """M - x^T Q x >= 0"""
        return M - x @ Q @ x
    
    # Bounds: x >= 0
    bounds = [(0.0, None)] * n_active
    
    # Initial point
    x0 = h_active.copy()
    
    # Run L-BFGS-B with the inequality constraint
    try:
        res = opt.minimize(
            objective,
            x0,
            jac=gradient,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if res.success or res.x is not None:
            h_new = h.copy()
            h_new[active_idx] = np.clip(res.x, 0, None)
            
            # Renormalize
            if np.sum(h_new) > 1e-15:
                h_new /= np.sum(h_new)
                
                # Verify improvement
                s_old = _score_raw(h)
                s_new = _score_raw(h_new)
                if s_new > s_old:
                    return h_new
    except Exception:
        pass
    
    return h

def _soft_max_surrogate(h, temperature=0.01):
    """
    Compute a smooth surrogate for max(conv) using log-sum-exp.
    This allows gradient-based optimization without the non-smoothness of max.
    """
    conv = _compute_convolution(h)
    # Log-sum-exp for numerical stability
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return max_conv
    log_conv = np.log(conv + 1e-30)
    log_max = np.max(log_conv)
    soft_max = max_conv + temperature * (log_max + np.log(np.sum(np.exp((log_conv - log_max) / temperature))))
    return soft_max

def _soft_max_grad(h, temperature=0.01):
    """
    Gradient of the soft-max surrogate with respect to h.
    d(soft_max)/dh_j = sum_k (conv_k / soft_max) * d(conv_k)/dh_j
    where d(conv_k)/dh_j = 2*h[k-j] for valid k-j
    """
    conv = _compute_convolution(h)
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return np.zeros(N)
    
    # Soft-max weights
    log_conv = np.log(conv + 1e-30)
    log_max = np.max(log_conv)
    weights = np.exp((log_conv - log_max) / temperature)
    weights /= np.sum(weights)
    
    # Gradient of soft_max w.r.t. h
    grad = np.zeros(N)
    for j in range(N):
        # d(conv_k)/dh_j = 2*h[k-j] for k-j in [0, N-1]
        for k in range(2*N - 1):
            j2 = k - j
            if 0 <= j2 < N:
                grad[j] += weights[k] * 2.0 * h[j2]
    
    return grad * temperature  # Scale by temperature for soft-max

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Generate diverse initializations ---
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse structured
    np.random.seed(42)
    for trial in range(30):
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Concentrated Gaussians
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(0.5, N/3, 6):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
                
    # 4. Triangular
    for c in np.linspace(3, N-3, 6):
        for w in np.linspace(2, N, 6):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Random smooth
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)
            
    # 6. Structured sparse: pairs symmetric around center
    for k in range(3, 10):
        for offset in range(1, 8):
            h_pair = np.zeros(N)
            center = N // 2
            for i in range(k):
                pos1 = center - offset - i
                pos2 = center + offset + i
                if 0 <= pos1 < N:
                    h_pair[pos1] = 1.0
                if 0 <= pos2 < N:
                    h_pair[pos2] = 1.0
            if np.sum(h_pair) > 0:
                h_pair /= np.sum(h_pair)
                inits.append(h_pair)
                
    # 7. Spike-like (single peak)
    for pos in range(N):
        h_spike = np.zeros(N)
        h_spike[pos] = 1.0
        inits.append(h_spike)
        
    # 8. Two-spike
    for pos1 in range(0, N, 3):
        for pos2 in range(pos1+2, N, 3):
            h_two = np.zeros(N)
            h_two[pos1] = 1.0
            h_two[pos2] = 1.0
            h_two /= np.sum(h_two)
            inits.append(h_two)

    # 9. Sidon-set-like: use a known maximal Sidon set on Z_50 or random search for low-collision sets
    # A Sidon set (B_2 set) has all pairwise sums distinct (mod N), minimizing convolution collisions.
    # We'll try to construct such sets by greedy or random search.
    
    def find_sidon_like(n_points, n_trials=500):
        """Find a set of indices with minimal collision count in pairwise sums mod N."""
        best_set = None
        best_collisions = float('inf')
        
        for _ in range(n_trials):
            idx = np.random.choice(N, n_points, replace=False)
            # Count collisions in pairwise sums
            sums = np.array([(i + j) % N for i in idx for j in idx if i <= j])
            unique_sums = np.unique(sums)
            collisions = len(sums) - len(unique_sums)
            if collisions < best_collisions:
                best_collisions = collisions
                best_set = idx.copy()
        
        return best_set
    
    # Try various sizes of Sidon-like sets
    for k in range(5, 12):
        sidon_set = find_sidon_like(k, n_trials=200)
        if sidon_set is not None:
            h_sidon = np.zeros(N)
            h_sidon[sidon_set] = 1.0
            h_sidon /= np.sum(h_sidon)
            inits.append(h_sidon)

    # 10. Weighted Sidon-like: vary weights on Sidon sets
    for k in range(5, 10):
        sidon_set = find_sidon_like(k, n_trials=100)
        if sidon_set is not None:
            for _ in range(10):
                weights = np.random.rand(k) + 0.1
                weights /= np.sum(weights)
                h_w = np.zeros(N)
                h_w[sidon_set] = weights
                inits.append(h_w)

    # --- Phase 2: Trust-Region Optimization with Active-Set Identification ---
    # For each initialization, solve a trust-region subproblem that maximizes
    # ||h||^2 subject to the dominant convolution peak being bounded.
    # This exploits the fact that for fixed support, the peak constraint is
    # a quadratic form, enabling rapid convergence via L-BFGS-B.
    
    def run_trust_region(h_init, delta_trust=0.1, max_iter=50):
        """
        Run trust-region optimization from h_init.
        Identifies the dominant lag and solves the subproblem.
        """
        if np.sum(h_init) < 1e-15:
            return None
        h = h_init / np.sum(h_init)
        conv = _compute_convolution(h)
        k_star = np.argmax(conv)
        
        h_new = _trust_region_subproblem(h, k_star, delta_trust, max_iter)
        return h_new
    
    # Run trust-region from multiple inits
    for h_init in inits:
        if time.time() > deadline - 40:
            break
        h_candidate = run_trust_region(h_init, delta_trust=0.1, max_iter=50)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # --- Phase 3: Enhanced Block-Coordinate Descent with Trust-Region Polish ---
    # Identify dominant peak and optimize using trust-region subproblems
    # on the active set, followed by block-coordinate refinement.
    
    def enhanced_refine(h_start, max_iter=30, use_trust_region=True):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline - 15:
                break
            improved = False
            iter_count += 1
            
            # Try trust-region subproblem first
            if use_trust_region:
                conv = _compute_convolution(h)
                k_star = np.argmax(conv)
                h_tr = _trust_region_subproblem(h, k_star, delta_trust=0.05, max_iter=30)
                s_tr = _score_raw(h_tr)
                s_curr = _score_raw(h)
                if s_tr > s_curr * (1 + 1e-8):
                    h = h_tr
                    improved = True
                    continue
            
            # Block-coordinate descent as fallback
            conv = _compute_convolution(h)
            k_star = np.argmax(conv)
            
            # For each position i, try to optimize h[i] and h[k_star - i]
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                
                hi = h[i]
                hj = h[j]
                
                t_min = -hi
                t_max = hj
                
                if t_max - t_min < 1e-15:
                    continue
                
                best_t = 0.0
                best_local_score = _score_raw(h)
                
                # Adaptive number of evaluations based on remaining time
                n_evals = 15 if time.time() < deadline - 20 else 8
                for t in np.linspace(t_min * 0.8, t_max * 0.8, n_evals):
                    h_test = h.copy()
                    h_test[i] += t
                    h_test[j] -= t
                    if np.any(h_test < -1e-15):
                        continue
                    h_test = np.clip(h_test, 0, None)
                    if np.sum(h_test) < 1e-15:
                        continue
                    h_test /= np.sum(h_test)
                    s_test = _score_raw(h_test)
                    if s_test > best_local_score:
                        best_local_score = s_test
                        best_t = t
                
                if abs(best_t) > 1e-10:
                    h[i] += best_t
                    h[j] -= best_t
                    h = np.clip(h, 0, None)
                    if np.sum(h) < 1e-15:
                        break
                    h /= np.sum(h)
                    improved = True
        
        return h
    
    # Refine best solution with enhanced refinement
    if time.time() < deadline - 20:
        h_refined = enhanced_refine(best_h, max_iter=40, use_trust_region=True)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # --- Phase 4: Local search with random perturbations and trust-region polish ---
    if time.time() < deadline - 10:
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            h_pert = best_h.copy()
            # Random perturbation with varying intensity
            mask = np.random.rand(N) < 0.3
            intensity = 0.1 + 0.2 * np.random.rand()
            h_pert[mask] *= (1 + intensity * np.random.randn(np.sum(mask)))
            h_pert = np.clip(h_pert, 0, None)
            if np.sum(h_pert) < 1e-15:
                continue
            h_pert /= np.sum(h_pert)
            
            s_pert = _score_raw(h_pert)
            if s_pert > best_score:
                best_score = s_pert
                best_h = h_pert.copy()
                
                # Refine the new best with enhanced refinement
                if time.time() < deadline - 15:
                    h_refined = enhanced_refine(best_h, max_iter=20, use_trust_region=True)
                    s_refined = _score_raw(h_refined)
                    if s_refined > best_score:
                        best_score = s_refined
                        best_h = h_refined.copy()
    
    # --- Phase 5: Soft-Max Surrogate Optimization ---
    # Use a smooth surrogate for the max operation to enable gradient-based
    # optimization without non-smoothness issues.
    
    def _score_soft_max(h, temperature=0.01):
        """Score using soft-max surrogate for max(conv)."""
        h = np.asarray(h, dtype=np.float64)
        l1 = np.sum(h)
        if l1 <= 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        soft_max_val = _soft_max_surrogate(h, temperature)
        if soft_max_val <= 1e-15:
            return 0.0
        return l2_sq / (l1 * DELTA * soft_max_val)
    
    def _grad_score_soft_max(h, temperature=0.01):
        """
        Gradient of the soft-max score.
        score = ||h||^2 / (l1 * DELTA * soft_max)
        d(score)/dh = [2h * l1 * soft_max - ||h||^2 * (1/l1) * d(soft_max)/dh] / (l1 * DELTA * soft_max)^2
        Simplified: we compute numerically or analytically.
        """
        h = np.asarray(h, dtype=np.float64)
        l1 = np.sum(h)
        if l1 <= 1e-15:
            return np.zeros(N)
        
        l2_sq = np.sum(h**2)
        soft_max_val = _soft_max_surrogate(h, temperature)
        if soft_max_val <= 1e-15:
            return np.zeros(N)
        
        # d(soft_max)/dh
        d_soft_max = _soft_max_grad(h, temperature)
        
        # d(score)/dh_j = [2*h_j * (l1 * DELTA * soft_max) - l2_sq * DELTA * d(soft_max)/dh_j] / (l1 * DELTA * soft_max)^2
        #                = [2*h_j / (l1 * soft_max)] - [l2_sq * d(soft_max)/dh_j / (l1^2 * DELTA * soft_max^2)]
        
        denom = (l1 * DELTA * soft_max_val) ** 2
        grad = np.zeros(N)
        for j in range(N):
            num = 2.0 * h[j] * (l1 * DELTA * soft_max_val) - l2_sq * DELTA * d_soft_max[j]
            grad[j] = num / denom
        
        return grad
    
    # Try soft-max surrogate optimization from the best solution
    if time.time() < deadline - 15:
        h_curr = best_h.copy()
        h_curr = np.clip(h_curr, 0, None)
        if np.sum(h_curr) < 1e-15:
            h_curr = np.ones(N) / N
        h_curr /= np.sum(h_curr)
        
        # Use L-BFGS-B to maximize the soft-max score
        def neg_score_soft_max(x):
            return -_score_soft_max(x, temperature=0.01)
        
        def grad_neg_score_soft_max(x):
            return -_grad_score_soft_max(x, temperature=0.01)
        
        bounds = [(0.0, None)] * N
        
        try:
            res = opt.minimize(
                neg_score_soft_max,
                h_curr,
                jac=grad_neg_score_soft_max,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.x is not None:
                h_candidate = np.clip(res.x, 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final polish: one more round of enhanced refinement
    if time.time() < deadline - 5:
        h_final = enhanced_refine(best_h, max_iter=20, use_trust_region=True)
        s_final = _score_raw(h_final)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()
    
    return list(best_h)
