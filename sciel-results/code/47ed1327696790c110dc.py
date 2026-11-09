import numpy as np
from scipy.optimize import minimize
import time

def _shannon_entropy(probs):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    probs = np.asarray(probs)
    nonzero = probs[probs > 1e-15]
    if len(nonzero) == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log2(nonzero)))

def _compute_entropies(P):
    """Compute D and M from distribution P[y, x]."""
    n = P.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)  # X[y,x] = x, Y[y,x] = y
    
    # x - y
    diff = X - Y
    # x
    xvals = X
    # x + y
    sum1 = X + Y
    # x + 2y
    sum2 = X + 2 * Y
    
    def proj_entropy(vals):
        # Pool cells with equal value
        unique_vals = np.unique(vals)
        probs = np.array([P[vals == v].sum() for v in unique_vals])
        return _shannon_entropy(probs)
    
    D = proj_entropy(diff)
    H0 = proj_entropy(xvals)
    H1 = proj_entropy(sum1)
    H2 = proj_entropy(sum2)
    M = max(H0, H1, H2)
    
    return D, M

def _objective(params):
    """Objective to minimize: -D/M (we maximize D/M)."""
    P = np.exp(params)
    P = P / P.sum()
    D, M = _compute_entropies(P)
    if M <= 1e-9:
        return 1e6
    return -D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    deadline = time.time() + 600.0
    safety_margin = 30.0
    
    # Start with uniform
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    D, M = _compute_entropies(P)
    best_ratio = D / M if M > 1e-9 else 0.0
    best_P = P.copy()
    
    # Try some structured initializations
    # Strategy 1: Concentrate on cells that maximize x-y spread
    # Try putting mass on a "staircase" or diagonal patterns
    
    # Let's try: for each possible "base" pattern, see if it helps
    # Pattern: P[y,x] proportional to some function
    
    # Try optimization from multiple starting points
    start_points = []
    
    # Uniform
    start_points.append(np.log(P.flatten()))
    
    # Try concentrated on y=0 row
    P1 = np.zeros((n, n))
    P1[0, :] = 1.0 / n
    start_points.append(np.log(np.maximum(P1.flatten(), 1e-15)))
    
    # Try concentrated on x=y diagonal
    P2 = np.zeros((n, n))
    np.fill_diagonal(P2, 1.0 / n)
    start_points.append(np.log(np.maximum(P2.flatten(), 1e-15)))
    
    # Try anti-diagonal x + y = 9
    P3 = np.zeros((n, n))
    for i in range(n):
        P3[i, n - 1 - i] = 1.0 / n
    start_points.append(np.log(np.maximum(P3.flatten(), 1e-15)))
    
    # Random sparse patterns
    rng = np.random.default_rng(42)
    for trial in range(20):
        P_rand = np.zeros((n, n))
        # Pick a random subset of cells
        num_cells = rng.integers(5, 20)
        cells = rng.choice(n * n, size=num_cells, replace=False)
        P_rand.flat[cells] = 1.0 / num_cells
        start_points.append(np.log(np.maximum(P_rand.flatten(), 1e-15)))
    
    # Run optimization from each starting point
    for start_idx, start in enumerate(start_points):
        if time.time() > deadline - safety_margin:
            break
        
        try:
            result = minimize(
                _objective,
                start,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                P_opt = np.exp(result.x)
                P_opt = P_opt / P_opt.sum()
                D_opt, M_opt = _compute_entropies(P_opt)
                if M_opt > 1e-9:
                    ratio = D_opt / M_opt
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_P = P_opt.copy()
        except Exception:
            pass
    
    # Additional local search with random perturbations
    if time.time() < deadline - safety_margin:
        current_P = best_P.copy()
        current_ratio = best_ratio
        
        for iteration in range(5000):
            if time.time() > deadline - safety_margin:
                break
            
            # Random perturbation
            P_new = current_P + rng.normal(0, 0.01, size=(n, n))
            P_new = np.maximum(P_new, 0)
            if P_new.sum() <= 1e-9:
                continue
            P_new = P_new / P_new.sum()
            
            D_new, M_new = _compute_entropies(P_new)
            if M_new > 1e-9:
                ratio_new = D_new / M_new
                if ratio_new > current_ratio:
                    current_P = P_new.copy()
                    current_ratio = ratio_new
                    if ratio_new > best_ratio:
                        best_ratio = ratio_new
                        best_P = P_new.copy()
    
    # Final validation and clipping
    best_P = np.maximum(best_P, 0)
    total = best_P.sum()
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / total
    
    return best_P
