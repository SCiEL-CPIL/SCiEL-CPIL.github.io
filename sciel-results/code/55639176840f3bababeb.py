# EVOLVE-BLOCK-START
import numpy as np
import time


def _shannon_entropy(probs):
    """Compute Shannon entropy in bits for a probability vector."""
    probs = np.array(probs)
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _project_entropy(P, r, n=10):
    """Compute entropy of projection x + r*y for distribution P[y,x]."""
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    vals = X + r * Y
    min_v = int(vals.min())
    max_v = int(vals.max())
    num_vals = max_v - min_v + 1
    counts = np.zeros(num_vals)
    for v in range(num_vals):
        mask = (vals == min_v + v)
        counts[v] = P[mask].sum()
    return _shannon_entropy(counts)


def _compute_ratio(P):
    """Compute D/M ratio."""
    n = 10
    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        return 0.0
    P = P / total
    
    D = _project_entropy(P, -1, n)
    H0 = _project_entropy(P, 0, n)
    H1 = _project_entropy(P, 1, n)
    H2 = _project_entropy(P, 2, n)
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
    return D / M


def _compute_gradient(P, n=10, eps=1e-7):
    """Compute gradient of D/M w.r.t. P using finite differences."""
    grad = np.zeros_like(P)
    ratio = _compute_ratio(P)
    for i in range(n):
        for j in range(n):
            P_plus = P.copy()
            P_plus[i, j] += eps
            r_plus = _compute_ratio(P_plus)
            grad[i, j] = (r_plus - ratio) / eps
    return grad


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    n = 10
    # Initialize with uniform
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P.copy()
    best_ratio = _compute_ratio(P)
    
    # Try some structured initializations
    # Diagonal x+y=c
    for c in range(2*n-1):
        P_init = np.zeros((n, n))
        for x in range(n):
            y = c - x
            if 0 <= y < n:
                P_init[y, x] = 1.0
        if P_init.sum() > 0:
            P_init /= P_init.sum()
            r = _compute_ratio(P_init)
            if r > best_ratio:
                best_ratio = r
                best_P = P_init.copy()
    
    # Diagonal x+2y=c
    for c in range(2*n+1):
        P_init = np.zeros((n, n))
        for x in range(n):
            y = (c - x) / 2.0
            if y == int(y) and 0 <= int(y) < n:
                P_init[int(y), x] = 1.0
        if P_init.sum() > 0:
            P_init /= P_init.sum()
            r = _compute_ratio(P_init)
            if r > best_ratio:
                best_ratio = r
                best_P = P_init.copy()
    
    # Anti-diagonal x-y=c
    for c in range(-(n-1), n):
        P_init = np.zeros((n, n))
        for x in range(n):
            y = x - c
            if 0 <= y < n:
                P_init[y, x] = 1.0
        if P_init.sum() > 0:
            P_init /= P_init.sum()
            r = _compute_ratio(P_init)
            if r > best_ratio:
                best_ratio = r
                best_P = P_init.copy()
    
    # Gradient ascent from best initial point
    P = best_P.copy()
    step_size = 0.1
    patience = 0
    max_patience = 50
    
    while time.time() < deadline:
        grad = _compute_gradient(P, n)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad = grad / grad_norm
        
        # Try step
        P_trial = P + step_size * grad
        P_trial = np.clip(P_trial, 0.0, None)
        total = P_trial.sum()
        if total <= 1e-9:
            step_size *= 0.5
            patience += 1
            if patience > max_patience:
                break
            continue
        P_trial = P_trial / total
        
        r_trial = _compute_ratio(P_trial)
        
        if r_trial > best_ratio:
            best_ratio = r_trial
            best_P = P_trial.copy()
            P = P_trial.copy()
            patience = 0
            step_size = min(step_size * 1.1, 1.0)
        else:
            step_size *= 0.5
            patience += 1
            if patience > max_patience:
                break
    
    # Final validation
    P_out = np.clip(best_P, 0.0, None)
    total = P_out.sum()
    if total <= 1e-9:
        P_out = np.ones((n, n)) / (n * n)
    else:
        P_out = P_out / total
    
    return P_out


# EVOLVE-BLOCK-END
