# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _shannon_entropy(probs):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def _compute_ratio(P):
    """Compute D/M ratio for a 10x10 distribution P[y, x]."""
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    
    P_flat = P.flatten()
    
    diff_idx = (X - Y + 9).flatten()
    x_idx = X.flatten()
    xy_idx = (X + Y).flatten()
    x2y_idx = (X + 2 * Y).flatten()
    
    p_diff = np.bincount(diff_idx, weights=P_flat, minlength=19)
    p_x = np.bincount(x_idx, weights=P_flat, minlength=10)
    p_xy = np.bincount(xy_idx, weights=P_flat, minlength=19)
    p_x2y = np.bincount(x2y_idx, weights=P_flat, minlength=28)
    
    D = _shannon_entropy(p_diff)
    H0 = _shannon_entropy(p_x)
    H1 = _shannon_entropy(p_xy)
    H2 = _shannon_entropy(p_x2y)
    M = max(H0, H1, H2)
    
    if M < 1e-9:
        return 0.0
    return D / M


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 570
    
    # Precompute index arrays
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    diff_idx = (X - Y + 9).flatten()
    x_idx = X.flatten()
    xy_idx = (X + Y).flatten()
    x2y_idx = (X + 2 * Y).flatten()
    
    def _entropy_from_counts(counts):
        total = counts.sum()
        if total < 1e-15:
            return 0.0
        probs = counts / total
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    def _objective(params):
        log_w = params.reshape(10, 10)
        log_w_max = np.max(log_w)
        exp_w = np.exp(log_w - log_w_max)
        P = exp_w / np.sum(exp_w)
        
        P_flat = P.flatten()
        p_diff = np.bincount(diff_idx, weights=P_flat, minlength=19)
        p_x = np.bincount(x_idx, weights=P_flat, minlength=10)
        p_xy = np.bincount(xy_idx, weights=P_flat, minlength=19)
        p_x2y = np.bincount(x2y_idx, weights=P_flat, minlength=28)
        
        D = _entropy_from_counts(p_diff)
        H0 = _entropy_from_counts(p_x)
        H1 = _entropy_from_counts(p_xy)
        H2 = _entropy_from_counts(p_x2y)
        M = max(H0, H1, H2)
        
        if M < 1e-9:
            return 0.0
        return -(D / M)
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = _compute_ratio(best_P)
    
    np.random.seed(42)
    max_restarts = 25
    
    for restart in range(max_restarts):
        if time.time() > deadline - 30:
            break
        
        # Diverse initializations
        if restart == 0:
            init = np.zeros((n, n))
        elif restart == 1:
            # Uniform in log space
            init = np.zeros((n, n))
        elif restart == 2:
            # Concentrate on anti-diagonal (x+y = const)
            init = np.zeros((n, n))
            for i in range(n):
                init[i, i] = 1.0
        elif restart == 3:
            # Concentrate on main diagonal (x-y = const)
            init = np.zeros((n, n))
            for i in range(n):
                init[i, i] = 1.0
        elif restart == 4:
            # x+2y = 0 only: x=0, y=0
            init = np.zeros((n, n))
            init[0, 0] = 1.0
        elif restart == 5:
            # Spread along lines of constant x+2y
            init = np.zeros((n, n))
            for yv in range(n):
                for xv in range(n):
                    val = xv + 2 * yv
                    if val % 5 == 0:
                        init[yv, xv] = 1.0
        elif restart == 6:
            # Random sparse
            init = np.zeros((n, n))
            indices = np.random.choice(100, size=20, replace=False)
            init.flat[indices] = np.random.randn(20)
        elif restart == 7:
            # Gaussian-like
            init = np.random.randn(n, n) * 1.5
        elif restart == 8:
            # Concentrate x on few values
            init = np.zeros((n, n))
            for xv in [0, 5, 9]:
                init[:, xv] = 1.0
        elif restart == 9:
            # Concentrate y on few values  
            init = np.zeros((n, n))
            for yv in [0, 5, 9]:
                init[yv, :] = 1.0
        else:
            init = np.random.randn(n, n) * (1.0 + 0.5 * (restart % 5))
        
        result = minimize(
            _objective,
            init.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if result.fun < -best_score:
            P_candidate = np.exp(result.x - np.max(result.x)).reshape(n, n)
            P_candidate /= P_candidate.sum()
            score = _compute_ratio(P_candidate)
            if score > best_score:
                best_score = score
                best_P = P_candidate.copy()
        
        # Polish with Nelder-Mead if we have a good candidate
        if time.time() > deadline - 60:
            break
        
        if best_score > 0.85:
            x0 = np.log(best_P + 1e-15)
            result2 = minimize(
                _objective,
                x0.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'maxfev': 5000}
            )
            if result2.fun < -best_score:
                P_candidate = np.exp(result2.x - np.max(result2.x)).reshape(n, n)
                P_candidate /= P_candidate.sum()
                score = _compute_ratio(P_candidate)
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.copy()
    
    # Final validation
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total < 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P /= total
    
    return best_P


# EVOLVE-BLOCK-END
