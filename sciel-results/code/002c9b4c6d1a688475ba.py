# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_P_fast(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] using sorted pairwise sums."""
    n = len(support)
    w = weights / weights.sum()
    S = support[:, None] + support[None, :]
    W = np.outer(w, w)
    S_flat = S.ravel()
    W_flat = W.ravel()
    order = np.argsort(S_flat, kind='mergesort')
    S_sorted = S_flat[order]
    W_cdf = np.cumsum(W_sorted)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            thr = 2.0 * support[l] - support[k]
            idx = np.searchsorted(S_sorted, thr, side='left')
            if idx > 0:
                P += w[k] * w[l] * W_cdf[idx - 1]
    return P


def _compute_P_vectorized(support, weights):
    """Vectorized P computation for moderate n."""
    n = len(support)
    w = weights / weights.sum()
    S = support[:, None] + support[None, :]
    W = np.outer(w, w)
    # thresholds[k,l] = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]
    # For each (k,l), compute sum of W[i,j] where S[i,j] < thresholds[k,l]
    # Use broadcasting: S_flat[None, :, :] < thresholds[:, :, None]
    S_flat = S.ravel()[None, None, :]
    thr_expanded = thresholds[:, :, None]
    mask = S_flat < thr_expanded  # shape (n, n, n*n)
    P = np.sum(w[:, None] * w[None, :] * np.sum(W.ravel()[None, None, :] * mask, axis=2))
    return P


def _two_point_P(p, ratio):
    """
    For a two-point distribution: mass p at 0, mass (1-p) at R.
    P[X1+X2+X3 < 2*X4]
    
    X4 = 0 (prob p): need X1+X2+X3 < 0 -> impossible (all >= 0), contributes 0.
    X4 = R (prob 1-p): need X1+X2+X3 < 2R.
      Each Xi is 0 with prob p, R with prob 1-p.
      X1+X2+X3 = m*R where m = number of Xi equal to R, m ~ Binomial(3, 1-p).
      Need m*R < 2R -> m < 2 -> m = 0 or 1.
      P(m<=1) = C(3,0)p^3 + C(3,1)p^2(1-p) = p^3 + 3p^2(1-p) = p^2(3-2p)
    
    So P = (1-p) * p^2 * (3 - 2p)
    """
    return (1.0 - p) * p**2 * (3.0 - 2.0 * p)


def _optimize_two_point():
    """Find optimal p for two-point distribution."""
    # Maximize f(p) = (1-p)*p^2*(3-2p) = 3p^2 - 6p^3 + 2p^4
    # f'(p) = 6p - 18p^2 + 8p^3 = 2p(3 - 9p + 4p^2) = 2p(4p^2 - 9p + 3)
    # 4p^2 - 9p + 3 = 0 -> p = (9 ± sqrt(81-48))/8 = (9 ± sqrt(33))/8
    # p = (9 - sqrt(33))/8 ≈ (9-5.7446)/8 ≈ 0.4069
    # or p = (9 + sqrt(33))/8 ≈ 1.968 (invalid)
    p_opt = (9.0 - np.sqrt(33.0)) / 8.0
    P_opt = _two_point_P(p_opt, 1.0)
    return p_opt, P_opt


def _optimize_multi_point(n_points, max_time=120.0):
    """
    Optimize a multi-point distribution using coordinate descent on
    (positions, weights) with the two-point optimum as anchor.
    
    Strategy: Start with the two-point optimal, then add intermediate points
    and optimize all parameters jointly.
    """
    deadline = time.time() + max_time
    rng = np.random.default_rng(12345)
    
    # Anchor: two-point optimum
    p_opt, P_two = _optimize_two_point()
    
    # Build initial multi-point support:
    # Points at 0, small values, and large value
    # Use geometric spacing between 0 and large value
    best_P = P_two
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([p_opt, 1.0 - p_opt])
    
    for n in range(3, min(n_points + 1, 8)):
        if time.time() > deadline - 30:
            break
        
        # Multiple initializations
        for init_type in range(3):
            if time.time() > deadline - 30:
                break
            
            if init_type == 0:
                # Geometric spacing from small to large
                ratio = 3.0
                support = np.array([0.0] + [ratio**i for i in range(1, n-1)] + [ratio**(n-1)])
            elif init_type == 1:
                # Log-spaced
                support = np.logspace(-1, np.log10(10.0), n)
                support[0] = 0.0
            else:
                # Random geometric
                ratio = rng.uniform(2.0, 5.0)
                support = np.array([0.0] + [ratio**i for i in range(1, n-1)] + [ratio**(n-1)])
            
            # Initial weights: based on two-point, split mass among intermediate
            weights = np.ones(n) / n
            weights[0] = p_opt * 0.5
            weights[-1] = (1.0 - p_opt)
            mid_mass = p_opt * 0.5
            if n > 2:
                weights[1:-1] = mid_mass / (n - 2)
            
            # Coordinate descent optimization
            for iteration in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Optimize weights (positions fixed)
                best_w_change = False
                for i in range(n):
                    old_w = weights.copy()
                    # Try perturbing weight i
                    for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                        new_w = old_w.copy()
                        new_w[i] += delta
                        new_w = np.maximum(new_w, 1e-10)
                        new_w = new_w / new_w.sum()
                        P_new = _compute_P_fast(support, new_w)
                        if P_new > best_P + 1e-12:
                            weights = new_w
                            best_P = P_new
                            best_w_change = True
                            break
                    if best_w_change:
                        break
                
                # Optimize positions (weights fixed) - only for intermediate points
                for i in range(1, n - 1):
                    old_s = support.copy()
                    improved = False
                    for scale in [0.5, 0.8, 1.2, 1.5]:
                        new_s = old_s.copy()
                        new_s[i] = old_s[i] * scale
                        # Ensure sorted and non-negative
                        if new_s[i] < 0 or not np.all(np.isfinite(new_s)):
                            continue
                        P_new = _compute_P_fast(new_s, weights)
                        if P_new > best_P + 1e-12:
                            support = new_s
                            best_P = P_new
                            improved = True
                            break
                    if improved:
                        break
                
                # Also try adjusting the largest point
                for scale in [0.8, 1.2, 1.5]:
                    old_s = support.copy()
                    new_s = old_s.copy()
                    new_s[-1] = old_s[-1] * scale
                    if new_s[-1] < 0 or not np.all(np.isfinite(new_s)):
                        continue
                    P_new = _compute_P_fast(new_s, weights)
                    if P_new > best_P + 1e-12:
                        support = new_s
                        best_P = P_new
                        break
                
                # Check convergence
                if not best_w_change:
                    # Try a random restart in weights
                    if iteration % 20 == 10:
                        weights = np.abs(rng.normal(0, 1, n))
                        weights[0] = max(weights[0], 0.1)
                        weights[-1] = max(weights[-1], 0.1)
                        weights = weights / weights.sum()
            
            # Update best
            if best_P > 0:
                best_support = support.copy()
                best_weights = weights.copy()
    
    return best_support, best_weights, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use analytical two-point optimum as anchor, then add intermediate
    points and optimize positions and weights via coordinate descent.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin: 600s limit
    
    # Start with the analytical two-point optimum
    p_opt, P_two = _optimize_two_point()
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([p_opt, 1.0 - p_opt])
    best_P = P_two
    
    # Try multi-point optimizations with increasing complexity
    for n_points in [3, 4, 5, 6, 7]:
        remaining = deadline - time.time()
        if remaining < 60:
            break
        
        time_budget = min(remaining - 30, 150.0)
        support, weights, P = _optimize_multi_point(n_points, max_time=time_budget)
        
        if P > best_P:
            best_P = P
            best_support = support
            best_weights = weights
    
    # Final refinement: try fine-tuning the best solution found
    remaining = deadline - time.time()
    if remaining > 60 and len(best_support) >= 3:
        # Local search around the best solution
        rng = np.random.default_rng(999)
        for _ in range(min(50, int(remaining / 3))):
            if time.time() > deadline - 30:
                break
            
            s_trial = best_support.copy()
            w_trial = best_weights.copy()
            
            # Perturb a random position
            i = rng.integers(1, len(s_trial) - 1)
            s_trial[i] *= rng.uniform(0.7, 1.3)
            s_trial = np.maximum(s_trial, 0.0)
            
            # Perturb a random weight
            j = rng.integers(0, len(w_trial))
            w_trial[j] *= rng.uniform(0.8, 1.2)
            w_trial = np.maximum(w_trial, 1e-10)
            w_trial = w_trial / w_trial.sum()
            
            P_trial = _compute_P_fast(s_trial, w_trial)
            if P_trial > best_P:
                best_P = P_trial
                best_support = s_trial
                best_weights = w_trial
    
    # Validate
    if len(best_support) != len(best_weights):
        return np.array([0.0, 1.0]), np.array([p_opt, 1.0 - p_opt])
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
        return np.array([0.0, 1.0]), np.array([p_opt, 1.0 - p_opt])
    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        return np.array([0.0, 1.0]), np.array([p_opt, 1.0 - p_opt])
    if best_weights.sum() <= 1e-9:
        return np.array([0.0, 1.0]), np.array([p_opt, 1.0 - p_opt])
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END
