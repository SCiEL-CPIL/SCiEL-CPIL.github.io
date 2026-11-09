# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximize the arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    D = H_{-1} (entropy of x - y), M = max(H_0, H_1, H_2).
    Uses a parametric ansatz P(x,y) ∝ exp(-α x² - β y² - γ xy) optimized via gradient ascent.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, we'll use 590s to be safe

    # Precompute grid indices
    # P[y, x], so row index is y, col index is x
    # x ranges 0..9, y ranges 0..9
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    # X[y, x] = x, Y[y, x] = y

    # Center the grid to allow negative values in the exponent for symmetry
    Xc = X - (n - 1) / 2.0
    Yc = Y - (n - 1) / 2.0

    def compute_ratio_from_logP(logP):
        """Compute D/M given log probabilities."""
        P = np.exp(logP - np.max(logP))  # stabilize
        P = P / P.sum()
        
        # Clip tiny negatives just in case (shouldn't happen with exp)
        P = np.maximum(P, 0)
        total = P.sum()
        if total <= 1e-9:
            return -1.0
        P = P / total

        # Slopes: -1, 0, 1, 2
        # Value for slope r: x + r*y
        vals = {
            -1: X - Y,
            0: X,
            1: X + Y,
            2: X + 2 * Y
        }

        def entropy(vals_r, P):
            # Pool probabilities by value
            unique_vals = np.unique(vals_r)
            probs = np.array([P[vals_r == v].sum() for v in unique_vals])
            probs = probs[probs > 1e-12]
            if len(probs) == 0:
                return 0.0
            return -np.sum(probs * np.log2(probs))

        D = entropy(vals[-1], P)
        H0 = entropy(vals[0], P)
        H1 = entropy(vals[1], P)
        H2 = entropy(vals[2], P)
        M = max(H0, H1, H2)
        
        if M < 1e-9:
            return -1.0
        
        return D / M

    def objective(params):
        """Negative ratio for minimization."""
        alpha, beta, gamma = params
        logP = -alpha * Xc**2 - beta * Yc**2 - gamma * Xc * Yc
        # Normalize
        logP -= np.max(logP)
        P = np.exp(logP)
        P /= P.sum()
        ratio = compute_ratio_from_logP(logP)
        if ratio < 0:
            return 1e6
        return -ratio

    # Initial guess: uniform distribution corresponds to alpha=beta=gamma=0
    # But 0,0,0 is a saddle point or flat area. Let's try small perturbations.
    # We want to maximize D (x-y entropy) and minimize M (max of x, x+y, x+2y entropies).
    # Uniform: D = log2(19) ≈ 4.25, H0=H1=H2=log2(10)≈3.32. Ratio ≈ 1.28.
    # Wait, the seed score is 1.0. Let's check the seed ratio.
    # Seed is uniform.
    # D = entropy of x-y. x,y in 0..9. x-y in -9..9. 19 values.
    # Counts: 1,2,...,9,10,9,...,2,1. Sum=100.
    # H = - sum p log2 p.
    # M = max(H0, H1, H2).
    # H0: x in 0..9, 10 values, uniform. H0 = log2(10) ≈ 3.3219.
    # H1: x+y in 0..18. 19 values. Same distribution as x-y. H1 ≈ 4.2479.
    # H2: x+2y in 0..27. 28 values.
    # So M = H1 ≈ 4.2479.
    # Ratio = D/M ≈ 4.2479 / 4.2479 = 1.0.
    # Ah, so the seed ratio is exactly 1.0 (or very close).
    # We need to increase D or decrease M.
    # Decreasing M is hard if D also decreases.
    # Increasing D: make x-y more uniform.
    # The distribution of x-y for uniform grid is triangular (peaked at 0).
    # To make it more uniform, we need to weight the corners more?
    # If we weight corners (0,0), (0,9), (9,0), (9,9), then x-y takes values 0, -9, 9, 0.
    # This concentrates D. Bad.
    # We want to spread out x-y.
    # The values x-y = k have max count 10 (at k=0). Min count 1 (at k=±9).
    # To equalize, we need to boost the low-probability cells (corners) relative to center.
    # So P should be higher at corners, lower at center.
    # This suggests alpha, beta > 0 (Gaussian decay from center) is WRONG.
    # We want alpha, beta < 0? i.e. exp(+|x|^2)? That's unbounded.
    # The ansatz exp(-alpha x^2) with alpha < 0 grows away from center.
    # Let's try alpha, beta, gamma in a range that allows "anti-Gaussian" shapes.
    # Or just use a polynomial ansatz?
    # Let's stick to the quadratic but allow negative coefficients.
    
    # Let's define the search space.
    # alpha, beta, gamma in [-1, 1] might be too small or large.
    # Let's use a grid search or multi-start optimization.
    
    best_P = None
    best_score = -1.0

    # Try a few initial points
    inits = [
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.0],
        [-0.1, -0.1, 0.0],
        [0.2, 0.2, 0.1],
        [-0.2, -0.2, -0.1],
        [0.5, 0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, -0.5],
    ]

    for init in inits:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success or res.fun < 1e5:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    alpha, beta, gamma = res.x
                    logP = -alpha * Xc**2 - beta * Yc**2 - gamma * Xc * Yc
                    logP -= np.max(logP)
                    P = np.exp(logP)
                    P /= P.sum()
                    best_P = P
        except Exception:
            pass

    # If no optimization succeeded or score is low, try a manual heuristic
    # Boost corners to increase D (spread x-y)
    # P[y, x] = 1 + eps * (something)
    # Let's try a specific heuristic: P proportional to 1 + c * (x-4.5)^2 + c * (y-4.5)^2
    # This boosts corners.
    # Let's evaluate this heuristic directly.
    
    if best_P is None or best_score < 1.0:
        # Try heuristic: boost corners
        # P[y, x] ∝ 1 + k * ((x-4.5)^2 + (y-4.5)^2)
        # This is equivalent to alpha=beta=-k, gamma=0 in our ansatz (approximately, if we don't center)
        # Actually, our ansatz is centered.
        # Let's just use the best from optimization. If that failed, return uniform?
        # No, let's try one more thing: random search in the 3D space.
        
        np.random.seed(42)
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            params = np.random.uniform(-0.5, 0.5, 3)
            score = -objective(params)
            if score > best_score:
                best_score = score
                alpha, beta, gamma = params
                logP = -alpha * Xc**2 - beta * Yc**2 - gamma * Xc * Yc
                logP -= np.max(logP)
                P = np.exp(logP)
                P /= P.sum()
                best_P = P

    if best_P is None:
        # Fallback to uniform
        best_P = np.ones((n, n)) / (n * n)

    # Final validation and cleanup
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P /= total

    return best_P

# EVOLVE-BLOCK-END
