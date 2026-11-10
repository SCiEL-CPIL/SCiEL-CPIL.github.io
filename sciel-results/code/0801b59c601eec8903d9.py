# EVOLVE-BLOCK-START
import numpy as np
import time


def _shannon_entropy(probs):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _project_entropy(P, slope):
    """Compute entropy of projection along given slope.
    P[y, x] where y is row, x is column.
    For slope r, value = x + r*y
    """
    n = P.shape[0]
    ys = np.arange(n)[:, None]  # (n, 1)
    xs = np.arange(n)[None, :]  # (1, n)
    values = xs + slope * ys  # (n, n)
    
    # Get unique values
    unique_vals = np.unique(values)
    ent = 0.0
    for v in unique_vals:
        mask = (values == v)
        prob = np.sum(P[mask])
        if prob > 1e-15:
            ent -= prob * np.log2(prob)
    return ent


def _compute_ratio(P):
    """Compute D/M ratio where D = H_{-1} and M = max(H_0, H_1, H_2)."""
    D = _project_entropy(P, -1.0)
    H0 = _project_entropy(P, 0.0)
    H1 = _project_entropy(P, 1.0)
    H2 = _project_entropy(P, 2.0)
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M


def _soft_max(vals, temperature=0.1):
    """Smooth approximation of max using log-sum-exp."""
    vals = np.array(vals)
    m = np.max(vals)
    if m < 1e-15:
        return 0.0
    return m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))


def _objective(params):
    """Objective to maximize: log(D) - log(soft_max(H0, H1, H2))."""
    n = 10
    # params are log-weights, shape (n, n)
    log_w = params.reshape(n, n)
    # Normalize to get P
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)
    total = np.sum(w)
    if total < 1e-15:
        return -1e10
    P = w / total
    
    D = _project_entropy(P, -1.0)
    H0 = _project_entropy(P, 0.0)
    H1 = _project_entropy(P, 1.0)
    H2 = _project_entropy(P, 2.0)
    
    if D < 1e-15:
        return -1e10
    
    # Use soft max for smoothness
    M_soft = _soft_max([H0, H1, H2], temperature=0.05)
    if M_soft < 1e-15:
        return -1e10
    
    return np.log(D) - np.log(M_soft)


def _random_init(n, seed=42):
    """Random initial parameter vector (log-weights)."""
    rng = np.random.RandomState(seed)
    # Start near uniform with some random perturbation
    base = np.zeros((n, n))
    perturbation = rng.randn(n, n) * 0.5
    return (base + perturbation).flatten()


def _gradient_ascent(params, step_size=0.1, max_iter=500, n=10, deadline=None):
    """Simple gradient ascent with numerical gradients."""
    current_params = params.copy()
    current_obj = _objective(current_params)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute numerical gradient
        grad = np.zeros_like(current_params)
        eps = 1e-5
        for j in range(len(current_params)):
            old_val = current_params[j]
            current_params[j] = old_val + eps
            f_plus = _objective(current_params)
            current_params[j] = old_val - eps
            f_minus = _objective(current_params)
            current_params[j] = old_val
            grad[j] = (f_plus - f_minus) / (2 * eps)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        grad = grad / grad_norm
        
        # Adaptive step size
        new_params = current_params + step_size * grad
        new_obj = _objective(new_params)
        
        if new_obj > current_obj:
            current_params = new_params
            current_obj = new_obj
        else:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-8:
                break
    
    return current_params, current_obj


def _refine_with_local_search(params, n=10, max_iter=200, deadline=None):
    """Local refinement with smaller steps."""
    current_params = params.copy()
    current_obj = _objective(current_params)
    step_size = 0.05
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = np.zeros_like(current_params)
        eps = 1e-6
        for j in range(len(current_params)):
            old_val = current_params[j]
            current_params[j] = old_val + eps
            f_plus = _objective(current_params)
            current_params[j] = old_val - eps
            f_minus = _objective(current_params)
            current_params[j] = old_val
            grad[j] = (f_plus - f_minus) / (2 * eps)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        grad = grad / grad_norm
        
        new_params = current_params + step_size * grad
        new_obj = _objective(new_params)
        
        if new_obj > current_obj:
            current_params = new_params
            current_obj = new_obj
        else:
            step_size *= 0.7
            if step_size < 1e-9:
                break
    
    return current_params, current_obj


def _params_to_P(params, n=10):
    """Convert log-weight parameters to normalized probability matrix."""
    log_w = params.reshape(n, n)
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)
    total = np.sum(w)
    if total < 1e-15:
        return np.ones((n, n)) / (n * n)
    P = w / total
    # Clip and renormalize
    P = np.clip(P, 0, None)
    s = np.sum(P)
    if s < 1e-15:
        return np.ones((n, n)) / (n * n)
    P = P / s
    return P


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization on a 10x10 grid.
    
    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s limit
    
    # Try multiple random initializations
    best_params = None
    best_obj = -np.inf
    best_P = np.ones((n, n)) / (n * n)
    
    # Initialize with uniform (baseline)
    uniform_params = np.zeros((n, n)).flatten()
    P_uniform = _params_to_P(uniform_params, n)
    obj_uniform = _objective(uniform_params)
    best_params = uniform_params.copy()
    best_obj = obj_uniform
    best_P = P_uniform.copy()
    
    # Try several random initializations
    rng = np.random.RandomState(12345)
    for trial in range(8):
        if time.time() > deadline - 120:
            break
        
        # Random perturbation around uniform
        scale = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0][trial % 8]
        params = (rng.randn(n, n) * scale).flatten()
        
        # Phase 1: Coarse gradient ascent
        params, obj = _gradient_ascent(params, step_size=0.3, max_iter=300, n=n, deadline=deadline - 60)
        
        if obj > best_obj:
            best_obj = obj
            best_params = params.copy()
            best_P = _params_to_P(params, n)
        
        # Phase 2: Fine refinement
        if time.time() < deadline - 30:
            params, obj = _refine_with_local_search(params, n=n, max_iter=300, deadline=deadline - 10)
            if obj > best_obj:
                best_obj = obj
                best_params = params.copy()
                best_P = _params_to_P(params, n)
    
    # Also try some structured initializations
    # Diagonal concentration
    for diag_offset in range(-3, 4):
        if time.time() > deadline - 60:
            break
        params = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x - y == diag_offset:
                    params[y, x] = 1.0
        params = params.flatten()
        params, obj = _gradient_ascent(params, step_size=0.2, max_iter=200, n=n, deadline=deadline - 30)
        if obj > best_obj:
            best_obj = obj
            best_params = params.copy()
            best_P = _params_to_P(params, n)
    
    # Final: ensure validity
    P = best_P
    P = np.clip(P, 0, None)
    s = np.sum(P)
    if s < 1e-9:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / s
    
    # Validate
    if P.shape != (n, n):
        P = np.ones((n, n)) / (n * n)
    if not np.all(np.isfinite(P)):
        P = np.ones((n, n)) / (n * n)
    if np.any(P < -1e-9):
        P = np.clip(P, 0, None)
        s = np.sum(P)
        if s < 1e-9:
            P = np.ones((n, n)) / (n * n)
        else:
            P = P / s
    
    return P


# EVOLVE-BLOCK-END
