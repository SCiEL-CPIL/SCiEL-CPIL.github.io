import numpy as np
import time
import sys
from scipy.optimize import minimize

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

def _soft_max(vals, beta):
    """Numerically stable soft max."""
    vals = np.asarray(vals, dtype=np.float64)
    m = np.max(vals)
    exp_vals = np.exp(beta * (vals - m))
    return exp_vals / np.sum(exp_vals)

def _log_soft_max(vals, beta):
    """Log of soft max, numerically stable."""
    vals = np.asarray(vals, dtype=np.float64)
    m = np.max(vals)
    log_sum = np.log(np.sum(np.exp(beta * (vals - m))))
    return m + log_sum / beta

def _autoconv_max_smooth(h, beta):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    conv = np.convolve(h, h, mode='full')
    # Use log-sum-exp for smooth max
    m = np.max(conv)
    if m <= 1e-15:
        return 1e-15
    # Soft max: (1/beta) * log(sum(exp(beta * conv)))
    # For numerical stability, shift by max
    shifted = beta * (conv - m)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m + log_sum / beta

def _log_ratio_and_grad(theta, beta):
    """Compute log(ratio) and its gradient with respect to theta.
    
    ratio = ||h||_2^2 / (||h||_1 * DELTA * ||h*h||_inf)
    log(ratio) = log(||h||_2^2) - log(||h||_1) - log(DELTA) - log(||h*h||_inf)
    
    h = softmax(beta * theta)
    """
    h = _soft_max(theta, beta)
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    
    if l1 <= 1e-15 or l2_sq <= 1e-15:
        return -1e10, np.zeros_like(theta)
    
    # Smooth max of autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv_smooth = _autoconv_max_smooth(h, beta)
    
    if max_conv_smooth <= 1e-15:
        return -1e10, np.zeros_like(theta)
    
    log_ratio = np.log(l2_sq) - np.log(l1) - np.log(DELTA) - np.log(max_conv_smooth)
    
    # Gradient computation using chain rule
    # d(log_ratio)/d(theta) = d(log_ratio)/d(h) * d(h)/d(theta)
    
    # d(h)/d(theta) = beta * h * (I - h^T 1)  (Jacobian of softmax)
    # For efficiency, we compute d(log_ratio)/d(h) first, then multiply
    
    # d(log(||h||_2^2))/d(h) = 2h / ||h||_2^2
    grad_l2 = 2.0 * h / l2_sq
    
    # d(-log(||h||_1))/d(h) = -1/h / ||h||_1  (but h >= 0, so -1/(h*l1))
    # Actually d(-log(sum(h)))/d(h_i) = -1/sum(h) = -1/l1
    grad_l1 = -1.0 / l1
    
    # d(-log(max_conv_smooth))/d(h)
    # max_conv_smooth = m + (1/beta) * log(sum(exp(beta*(conv-m))))
    # d(max_conv_smooth)/d(conv_k) = exp(beta*(conv_k-m)) / sum(exp(beta*(conv-m)))
    # d(conv_k)/d(h_j) = h_{k-j} + h_{k-(N-1-j)} for appropriate indices
    # This is complex, so use finite difference approximation for the max term
    
    # For the max term, use numerical gradient
    eps = 1e-7
    grad_max = np.zeros(N)
    for j in range(N):
        h_plus = h.copy()
        h_plus[j] += eps
        h_plus /= np.sum(h_plus)
        conv_plus = np.convolve(h_plus, h_plus, mode='full')
        max_plus = _autoconv_max_smooth(h_plus, beta)
        
        h_minus = h.copy()
        h_minus[j] = max(0, h_minus[j] - eps)
        h_minus /= np.sum(h_minus)
        conv_minus = np.convolve(h_minus, h_minus, mode='full')
        max_minus = _autoconv_max_smooth(h_minus, beta)
        
        grad_max[j] = (max_plus - max_minus) / (2 * eps)
    
    # d(-log(max_conv_smooth))/d(h) = -grad_max / max_conv_smooth
    grad_max_term = -grad_max / max_conv_smooth
    
    # Total gradient with respect to h
    grad_h = grad_l2 + grad_l1 + grad_max_term
    
    # Chain rule: d/d(theta) = grad_h @ d(h)/d(theta)
    # d(h)/d(theta) = beta * diag(h) - beta * outer(h, h)
    # So d(log_ratio)/d(theta) = beta * (diag(h) - outer(h,h)) @ grad_h
    # = beta * (h * grad_h - h * sum(h * grad_h))
    
    h_grad_h = h * grad_h
    sum_h_grad_h = np.sum(h_grad_h)
    grad_theta = beta * (h_grad_h - h * sum_h_grad_h)
    
    return log_ratio, grad_theta

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    np.random.seed(42)
    
    # --- Phase 1: Generate diverse initializations in theta space ---
    # theta is the pre-softmax parameter; h = softmax(beta * theta)
    # For uniform h, theta should be all equal (or any constant)
    
    inits_theta = []
    beta_inits = []
    
    # 1. Uniform (theta = 0)
    inits_theta.append(np.zeros(N))
    beta_inits.append(10.0)
    
    # 2. Random sparse-like: large positive values at few positions
    for trial in range(30):
        k = np.random.randint(3, 15)
        theta = np.random.randn(N) * 0.5
        idx = np.random.choice(N, k, replace=False)
        theta[idx] += np.random.uniform(2, 5)
        inits_theta.append(theta)
        beta_inits.append(np.random.uniform(5, 20))
    
    # 3. Gaussian-like: peak at random center
    for _ in range(20):
        center = np.random.uniform(5, N-5)
        width = np.random.uniform(1, N/3)
        x_idx = np.arange(N)
        theta = -0.5 * ((x_idx - center) / width)**2 * np.random.uniform(1, 3)
        inits_theta.append(theta)
        beta_inits.append(np.random.uniform(5, 15))
    
    # 4. Two-peak
    for _ in range(15):
        c1 = np.random.uniform(3, N/2 - 2)
        c2 = np.random.uniform(N/2 + 2, N - 3)
        w1 = np.random.uniform(1, 5)
        w2 = np.random.uniform(1, 5)
        x_idx = np.arange(N)
        theta = -0.5 * ((x_idx - c1) / w1)**2 - 0.5 * ((x_idx - c2) / w2)**2
        theta *= np.random.uniform(1, 3)
        inits_theta.append(theta)
        beta_inits.append(np.random.uniform(5, 15))
    
    # 5. Sidon-like: try to create low-collision patterns
    for _ in range(15):
        k = np.random.randint(5, 12)
        # Greedy Sidon-like selection
        theta = np.random.randn(N) * 0.3
        selected = []
        used_sums = set()
        for _ in range(k):
            candidates = [i for i in range(N) if i not in selected]
            if not candidates:
                break
            # Pick candidate that minimizes collision with existing
            best_cand = None
            best_collisions = float('inf')
            for cand in candidates:
                collisions = 0
                for s in selected:
                    sm = (cand + s) % N
                    if sm in used_sums:
                        collisions += 1
                if collisions < best_collisions:
                    best_collisions = collisions
                    best_cand = cand
            if best_cand is not None:
                selected.append(best_cand)
                for s in selected:
                    used_sums.add((best_cand + s) % N)
        
        theta = np.random.randn(N) * 0.3
        for i in selected:
            theta[i] += np.random.uniform(2, 4)
        inits_theta.append(theta)
        beta_inits.append(np.random.uniform(8, 20))
    
    # --- Phase 2: Gradient-based optimization in theta space ---
    def run_theta_opt(theta_init, beta_init, maxiter=200, ftol=1e-10):
        """Optimize theta to maximize log(ratio) using L-BFGS-B."""
        # Wrap to handle beta as fixed per run
        beta = beta_init
        
        def neg_log_ratio(theta):
            val, _ = _log_ratio_and_grad(theta, beta)
            return -val
        
        def neg_log_ratio_grad(theta):
            _, grad = _log_ratio_and_grad(theta, beta)
            return -grad
        
        try:
            res = minimize(
                neg_log_ratio,
                theta_init,
                jac=neg_log_ratio_grad,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            
            if res.success or res.fun < 0:  # Accept even if not fully converged
                theta_opt = res.x
                h = _soft_max(theta_opt, beta)
                h = np.clip(h, 0, None)
                if np.sum(h) > 1e-15:
                    h /= np.sum(h)
                    return h
        except Exception:
            pass
        return None
    
    # Run from multiple inits
    for i, (theta_init, beta_init) in enumerate(zip(inits_theta, beta_inits)):
        if time.time() > deadline - 30:
            break
        
        # Try multiple beta values for each init
        for beta_try in [beta_init * 0.5, beta_init, beta_init * 2.0]:
            if time.time() > deadline - 20:
                break
            
            h_candidate = run_theta_opt(theta_init, beta_try, maxiter=150, ftol=1e-10)
            if h_candidate is not None:
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
    
    # --- Phase 3: Annealing refinement ---
    # Start from best_h, convert to theta, then anneal beta
    
    def h_to_theta(h, beta):
        """Convert h to theta such that softmax(beta * theta) ≈ h."""
        # h = softmax(beta * theta) => log(h) = beta * theta - log(sum(exp(beta*theta)))
        # For simplicity, use theta = log(h) / beta (approximate, ignoring normalization)
        h_safe = np.maximum(h, 1e-15)
        theta = np.log(h_safe) / beta
        # Center it
        theta -= np.mean(theta)
        return theta
    
    if time.time() < deadline - 15:
        # Convert best_h to theta
        beta_start = 10.0
        theta_curr = h_to_theta(best_h, beta_start)
        
        # Anneal: start with low beta (smooth), increase to high beta (sharp)
        for beta_step in [5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 50.0]:
            if time.time() > deadline - 10:
                break
            
            h_candidate = run_theta_opt(theta_curr, beta_step, maxiter=100, ftol=1e-10)
            if h_candidate is not None:
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
                    theta_curr = h_to_theta(best_h, beta_step)
                else:
                    # Keep theta from the attempt for next beta
                    # Even if score didn't improve, the theta might be better for higher beta
                    pass
    
    # --- Phase 4: Local refinement with coordinate descent ---
    def local_refine(h_start, max_iter=30):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                
                hi = h[i]
                hj = h[j]
                
                t_min = -hi * 0.9
                t_max = hj * 0.9
                
                if t_max - t_min < 1e-15:
                    continue
                
                best_t = 0.0
                best_local = _score_raw(h)
                
                for t in np.linspace(t_min, t_max, 8):
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
                    if s_test > best_local:
                        best_local = s_test
                        best_t = t
                
                if abs(best_t) > 1e-10:
                    h[i] += best_t
                    h[j] -= best_t
                    h = np.clip(h, 0, None)
                    if np.sum(h) < 1e-15:
                        break
                    h /= np.sum(h)
                    improved = True
            
            if not improved:
                break
        
        return h
    
    if time.time() < deadline - 5:
        h_refined = local_refine(best_h, max_iter=20)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    return list(best_h)
