# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _shannon_entropy(p):
    """Shannon entropy in bits, ignoring zero probabilities."""
    p = np.asarray(p, dtype=np.float64)
    mask = p > 0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def _project_entropy(P, r):
    """Entropy of the projection x + r*y."""
    n = P.shape[0]
    xs = np.arange(n)
    ys = np.arange(n)
    vals = xs[None, :] + r * ys[:, None]  # shape (n, n)
    unique_vals = np.unique(vals)
    total = P.sum()
    if total <= 0:
        return 0.0
    probs = []
    for v in unique_vals:
        mask = (vals == v)
        probs.append(P[mask].sum())
    probs = np.array(probs)
    mask = probs > 0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(probs[mask] * np.log2(probs[mask])))


def _objective_neg(p_vec):
    """Negative of D - soft_max(H_0, H_1, H_2) for minimization."""
    P = p_vec.reshape(10, 10)
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 1e10
    P = P / s
    D = _project_entropy(P, -1.0)
    H0 = _project_entropy(P, 0.0)
    H1 = _project_entropy(P, 1.0)
    H2 = _project_entropy(P, 2.0)
    # soft-max with temperature
    temps = np.array([H0, H1, H2])
    t = 0.5  # temperature for soft-max
    soft_max = t * np.log(np.sum(np.exp(temps / t)))
    return -(D - soft_max)


def _coordinate_descent(P, max_iters=200, tol=1e-10):
    """Coordinate descent refinement."""
    P = P.copy()
    n = 10
    best_val = _objective_neg(P.ravel())
    for it in range(max_iters):
        improved = False
        for y in range(n):
            for x in range(n):
                old_val = P[y, x]
                # Try a few perturbations
                for delta in [0.01, -0.01, 0.05, -0.05, 0.001, -0.001]:
                    new_val = old_val + delta
                    if new_val < 0:
                        continue
                    P[y, x] = new_val
                    s = P.sum()
                    if s <= 0:
                        P[y, x] = old_val
                        continue
                    P_norm = P / s
                    val = _objective_neg(P_norm.ravel())
                    if val < best_val - tol:
                        best_val = val
                        improved = True
                    else:
                        P[y, x] = old_val
        if not improved:
            break
    P = np.clip(P, 0, None)
    s = P.sum()
    if s > 0:
        P = P / s
    return P


def _make_initial_patterns():
    """Create diverse initial patterns."""
    n = 10
    patterns = []
    
    # Uniform
    patterns.append(np.ones((n, n), dtype=np.float64) / (n * n))
    
    # Anti-diagonal emphasis (high H_{-1})
    P = np.zeros((n, n), dtype=np.float64)
    for y in range(n):
        for x in range(n):
            diff = x - y
            P[y, x] = 1.0  # uniform on anti-diagonals
    P /= P.sum()
    patterns.append(P)
    
    # Emphasize extreme anti-diagonals
    P = np.zeros((n, n), dtype=np.float64)
    for y in range(n):
        for x in range(n):
            diff = abs(x - y)
            P[y, x] = 1.0 + 0.5 * diff
    P /= P.sum()
    patterns.append(P)
    
    # Corner emphasis
    P = np.zeros((n, n), dtype=np.float64)
    for y in range(n):
        for x in range(n):
            d = min(x, y, n - 1 - x, n - 1 - y)
            P[y, x] = 1.0 / (1.0 + d)
    P /= P.sum()
    patterns.append(P)
    
    # Random
    np.random.seed(42)
    P = np.random.rand(n, n) + 0.1
    P /= P.sum()
    patterns.append(P)
    
    # Another random
    np.random.seed(123)
    P = np.random.rand(n, n) + 0.1
    P /= P.sum()
    patterns.append(P)
    
    # Diagonal emphasis
    P = np.zeros((n, n), dtype=np.float64)
    for y in range(n):
        for x in range(n):
            P[y, x] = 1.0 if x == y else 0.1
    P /= P.sum()
    patterns.append(P)
    
    # Mixed: emphasize both diagonals
    P = np.zeros((n, n), dtype=np.float64)
    for y in range(n):
        for x in range(n):
            if x == y or x + y == n - 1:
                P[y, x] = 1.0
            else:
                P[y, x] = 0.05
    P /= P.sum()
    patterns.append(P)
    
    return patterns


def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # 600s budget, leave margin
    
    best_P = None
    best_score = -np.inf
    
    for P_init in _make_initial_patterns():
        if time.time() > deadline - 30:
            break
        
        # SLSQP optimization
        p0 = P_init.ravel()
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            {'type': 'ineq', 'fun': lambda x: np.min(x)}
        ]
        bounds = [(0, 1)] * 100
        
        try:
            res = minimize(
                _objective_neg,
                p0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if time.time() > deadline - 30:
                break
            P_opt = np.clip(res.x, 0, None)
            s = P_opt.sum()
            if s > 0:
                P_opt = P_opt / s
        except Exception:
            P_opt = P_init
        
        # Coordinate descent polish
        if time.time() < deadline - 10:
            P_cd = _coordinate_descent(P_opt, max_iters=100)
        else:
            P_cd = P_opt
        
        # Evaluate
        P_cd = np.clip(P_cd, 0, None)
        s = P_cd.sum()
        if s <= 0:
            continue
        P_cd = P_cd / s
        
        D = _project_entropy(P_cd, -1.0)
        H0 = _project_entropy(P_cd, 0.0)
        H1 = _project_entropy(P_cd, 1.0)
        H2 = _project_entropy(P_cd, 2.0)
        M = max(H0, H1, H2)
        
        if M > 1e-9:
            score = D / M
            if score > best_score:
                best_score = score
                best_P = P_cd.copy()
    
    if best_P is None:
        # Fallback to uniform
        best_P = np.ones((10, 10), dtype=np.float64) / 100.0
    
    # Final validation
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P = best_P / s
    
    return best_P

# EVOLVE-BLOCK-END
