"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400
TIME_LIMIT = 120.0  # seconds
SAFETY_MARGIN = 5.0  # seconds


def _compute_upper_bound(h):
    """Compute the upper bound on C3 for given heights h."""
    if np.allclose(h, 0):
        return np.inf
    conv = np.convolve(h, h, mode='full')
    max_abs_conv = np.max(np.abs(conv))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h < 1e-15:
        return np.inf
    return max_abs_conv / (sum_abs_h ** 2)


def _smooth_max_logsumexp(abs_conv, T):
    """Smooth maximum using log-sum-exp trick."""
    max_val = np.max(abs_conv)
    if max_val < 1e-15:
        return 1e-15
    shifted = (abs_conv - max_val) / T
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return T * (log_sum + max_val / T)


def _objective_from_steps(steps, T=1.0):
    """
    Given step increments (random walk steps), compute smooth objective.
    Steps are the increments of a random walk; heights are cumulative sums.
    We normalize to keep sum_abs_h ~ 1 for stability.
    """
    # Cumulative sum gives the "state" trajectory
    cumsum = np.cumsum(steps)
    # Heights are differences between consecutive cumulative states
    # h[i] = cumsum[i] - cumsum[i-1] = steps[i], but we want a low-variance profile
    # Instead, let heights be the cumulative sum itself (smoothed signal)
    # and normalize
    h = cumsum
    # Normalize so that sum_abs_h is reasonable
    s = np.sum(np.abs(h))
    if s < 1e-15:
        return 1e10, h
    h_norm = h / s
    conv = np.convolve(h_norm, h_norm, mode='full')
    abs_conv = np.abs(conv)
    log_max = _smooth_max_logsumexp(abs_conv, T)
    # Objective: log_max / (sum_abs_h_norm^2) but sum_abs_h_norm = 1
    # So objective is just log_max
    return log_max, h_norm


def _objective_grad_from_steps(steps, T=1.0, eps=1e-6):
    """Numerical gradient of objective w.r.t. steps."""
    n = len(steps)
    grad = np.zeros(n)
    for i in range(n):
        sp = steps.copy()
        sp[i] += eps
        fm, _ = _objective_from_steps(sp, T)
        sm = steps.copy()
        sm[i] -= eps
        fm2, _ = _objective_from_steps(sm, T)
        grad[i] = (fm - fm2) / (2 * eps)
    return grad


def _projected_gradient_descent(steps0, lr, n_iter, T=1.0, deadline=None):
    """
    Projected gradient descent on the random-walk step parameterization.
    Projection: clip steps to a bounded range to keep heights reasonable.
    """
    steps = steps0.copy().astype(float)
    best_obj = np.inf
    best_steps = steps.copy()
    
    # Adaptive learning rate
    current_lr = lr
    patience = 0
    max_patience = 50
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        obj, h = _objective_from_steps(steps, T)
        if obj < best_obj:
            best_obj = obj
            best_steps = steps.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                current_lr *= 0.9
                patience = 0
                if current_lr < 1e-10:
                    break
        
        # Compute gradient (use numerical for simplicity, could be analytic)
        if it % 5 == 0 or it < 10:
            grad = _objective_grad_from_steps(steps, T)
            # Normalize gradient
            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-15:
                grad = grad / gnorm
        
        # Update with projected gradient
        steps = steps - current_lr * grad
        # Project: clip to bounded range
        steps = np.clip(steps, -1.0, 1.0)
        
        # Occasional restarts to escape local minima
        if it % 200 == 199:
            # Perturb
            steps += 0.1 * np.random.randn(len(steps))
            steps = np.clip(steps, -1.0, 1.0)
    
    return best_steps, best_obj


def construct():
    """Construct the signed step function heights using random-walk parameterization."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    np.random.seed(42)
    
    best_h = None
    best_value = np.inf
    
    # Multiple initializations for the random walk steps
    n_initializations = 8
    initializations = []
    
    # 1. Small random steps
    initializations.append(0.01 * np.random.randn(N))
    # 2. Larger random steps
    initializations.append(0.1 * np.random.randn(N))
    # 3. Biased random walk (positive drift)
    initializations.append(0.05 * np.random.randn(N) + 0.01)
    # 4. Biased random walk (negative drift)
    initializations.append(0.05 * np.random.randn(N) - 0.01)
    # 5. Smooth low-frequency pattern
    t = np.linspace(0, 4*np.pi, N)
    initializations.append(0.05 * np.sin(t) + 0.01 * np.random.randn(N))
    # 6. Block pattern
    block = 20
    init_block = np.zeros(N)
    for i in range(0, N, block*2):
        init_block[i:i+block] = 0.05
        init_block[i+block:i+2*block] = -0.05
    initializations.append(init_block + 0.01 * np.random.randn(N))
    # 7. Very small random
    initializations.append(0.005 * np.random.randn(N))
    # 8. Mixed frequency
    t2 = np.linspace(0, 2*np.pi, N)
    initializations.append(0.03 * np.sin(3*t2) + 0.02 * np.sin(7*t2) + 0.01 * np.random.randn(N))
    
    for init_idx, steps0 in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Determine iterations based on remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Estimate time per iteration (rough)
        # Use fewer iterations if time is tight
        n_iter = min(500, max(50, int(remaining * 50)))
        
        # Learning rate schedule
        lr = 0.1
        
        try:
            best_steps, best_obj = _projected_gradient_descent(
                steps0, lr=lr, n_iter=n_iter, T=1.0, deadline=deadline
            )
            
            # Convert best steps to heights
            h_candidate = np.cumsum(best_steps)
            s = np.sum(np.abs(h_candidate))
            if s < 1e-15:
                continue
            h_candidate = h_candidate / s
            
            value = _compute_upper_bound(h_candidate)
            
            if value < best_value:
                best_value = value
                best_h = h_candidate.copy()
                
        except Exception:
            continue
    
    if best_h is None:
        # Fallback: return a simple pattern
        h = np.zeros(N)
        for i in range(N):
            if i % 2 == 0:
                h[i] = 1.0
            else:
                h[i] = -1.0
        return h
    
    return best_h
# EVOLVE-BLOCK-END
