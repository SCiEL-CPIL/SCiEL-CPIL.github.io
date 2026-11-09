# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] efficiently using convolution."""
    n = len(support)
    w = weights.astype(np.float64)
    s = support.astype(np.float64)
    
    # Compute the distribution of S = X1 + X2 + X3 using convolution
    # S takes values s[i]+s[j]+s[k] with probability w[i]*w[j]*w[k]
    # Build the 3D sum distribution
    # For efficiency, build pairwise sums first
    pair_sums = s[:, None] + s[None, :]  # n x n
    pair_weights = w[:, None] * w[None, :]  # n x n
    
    # Now S = pair_sum + X3
    # For each possible X4 value s[k], threshold is 2*s[k]
    # P = sum_k w[k] * P[S < 2*s[k]]
    # P[S < 2*s[k]] = sum over (i,j,l) with s[i]+s[j]+s[l] < 2*s[k] of w[i]*w[j]*w[l]
    
    total_P = 0.0
    for k in range(n):
        threshold = 2.0 * s[k]
        # P[S < threshold] where S = X1+X2+X3
        p_sum = 0.0
        for i in range(n):
            for j in range(n):
                base = s[i] + s[j]
                if base >= threshold:
                    continue
                # Need s[l] < threshold - base
                # Find all l where s[l] < threshold - base
                rem = threshold - base
                # Since s is sorted, find the count
                idx = np.searchsorted(s, rem, side='left')
                if idx > 0:
                    p_sum += w[i] * w[j] * w[:idx].sum()
        total_P += w[k] * p_sum
    return total_P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    best_P = 0.0
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([0.5, 0.5])
    
    # 2-point baseline: {0, L} with mass p on 0
    # P = (1-p) * (p^3 + 3*p^2*(1-p))
    def two_point_P(p):
        return (1.0 - p) * (p**3 + 3.0 * p**2 * (1.0 - p))
    
    # Optimize 2-point
    from scipy.optimize import minimize_scalar
    res2 = minimize_scalar(lambda p: -two_point_P(p), bounds=(0.01, 0.99), method='bounded')
    p2 = res2.x
    P2 = two_point_P(p2)
    if P2 > best_P:
        best_P = P2
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([p2, 1.0 - p2])
    
    # Multi-start Nelder-Mead for n-point distributions
    from scipy.optimize import minimize
    
    def objective_n(n, initial_params):
        """
        Parameters: [s1, s2, ..., s_{n-1}, w1, w2, ..., w_{n-1}]
        Support is [0, s1, s2, ..., s_{n-1}, 1] (sorted)
        Weights: w0 = 1 - sum(w1..w_{n-1})
        """
        s_inner = initial_params[:n-2]
        w_inner = initial_params[n-2:]
        
        # Ensure valid
        if np.any(s_inner < 0) or np.any(s_inner > 1):
            return 1e6
        if np.any(w_inner < 0):
            return 1e6
        w0 = 1.0 - np.sum(w_inner)
        if w0 < 0:
            return 1e6
        
        support = np.concatenate([[0.0], s_inner, [1.0]])
        weights = np.concatenate([[w0], w_inner])
        
        # Sort support and reorder weights
        order = np.argsort(support)
        support = support[order]
        weights = weights[order]
        
        P = _compute_P(support, weights)
        return -P
    
    def run_nelder_mead(n, x0, max_iter=200):
        """Run Nelder-Mead for n-point distribution."""
        if time.time() > deadline - 30:
            return None, None, None
        
        def obj(params):
            if time.time() > deadline - 30:
                return 1e6
            s_inner = params[:n-2]
            w_inner = params[n-2:]
            
            # Clamp to valid range
            s_inner = np.clip(s_inner, 0.0, 1.0)
            w_inner = np.clip(w_inner, 0.0, 1.0)
            
            w0 = 1.0 - np.sum(w_inner)
            if w0 < 0:
                # Renormalize
                total = np.sum(w_inner)
                if total > 1e-12:
                    w_inner = w_inner / total * 0.99
                    w0 = 1.0 - np.sum(w_inner)
                else:
                    return 1e6
                if w0 < 0:
                    return 1e6
            
            support = np.concatenate([[0.0], s_inner, [1.0]])
            weights = np.concatenate([[w0], w_inner])
            
            order = np.argsort(support)
            support = support[order]
            weights = weights[order]
            
            P = _compute_P(support, weights)
            return -P
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
        except Exception:
            return None, None, None
        
        if res.fun < -1e6:  # invalid
            return None, None, None
        
        P = -res.fun
        params = res.x
        s_inner = np.clip(params[:n-2], 0.0, 1.0)
        w_inner = np.clip(params[n-2:], 0.0, 1.0)
        w0 = 1.0 - np.sum(w_inner)
        if w0 < 0:
            return None, None, None
        
        support = np.concatenate([[0.0], s_inner, [1.0]])
        weights = np.concatenate([[w0], w_inner])
        order = np.argsort(support)
        support = support[order]
        weights = weights[order]
        
        return P, support, weights
    
    # Try 3-point distributions with multiple starts
    np.random.seed(42)
    
    for n in [3, 4, 5]:
        if time.time() > deadline - 60:
            break
        
        n_starts = max(5, int(300 / n))  # more starts for smaller n
        
        for start_idx in range(n_starts):
            if time.time() > deadline - 30:
                break
            
            # Generate random initial point
            s_inner = np.sort(np.random.uniform(0.05, 0.95, n-2))
            w_inner = np.random.uniform(0.05, 0.4, n-2)
            w0 = 1.0 - np.sum(w_inner)
            if w0 < 0.01:
                w_inner = w_inner * 0.5
                w0 = 1.0 - np.sum(w_inner)
            
            x0 = np.concatenate([s_inner, w_inner])
            
            P, support, weights = run_nelder_mead(n, x0, max_iter=150)
            
            if P is not None and P > best_P:
                best_P = P
                best_support = support.copy()
                best_weights = weights.copy()
    
    # Also try structured starts: evenly spaced support
    for n in [3, 4, 5, 6]:
        if time.time() > deadline - 30:
            break
        
        s_inner = np.linspace(0.1, 0.9, n-2)
        # Try a few weight configurations
        for w_config in [
            np.full(n-2, 1.0/n),
            np.linspace(0.1, 0.3, n-2),
            np.linspace(0.3, 0.1, n-2),
            np.random.uniform(0.1, 0.3, n-2),
        ]:
            if time.time() > deadline - 15:
                break
            w_inner = w_config
            w0 = 1.0 - np.sum(w_inner)
            if w0 < 0:
                w_inner = w_inner / np.sum(w_inner) * 0.9
                w0 = 1.0 - np.sum(w_inner)
            if w0 < 0:
                continue
            
            x0 = np.concatenate([s_inner, w_inner])
            P, support, weights = run_nelder_mead(n, x0, max_iter=100)
            
            if P is not None and P > best_P:
                best_P = P
                best_support = support.copy()
                best_weights = weights.copy()
    
    # Local refinement: try small perturbations
    if time.time() < deadline - 20:
        n = len(best_support)
        support = best_support.copy()
        weights = best_weights.copy()
        
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            
            # Perturb weights
            pert_w = weights.copy()
            idx = np.random.randint(0, n)
            pert_w[idx] += 0.02 * np.random.randn()
            pert_w = np.maximum(pert_w, 0.0)
            if pert_w.sum() > 1e-9:
                pert_w /= pert_w.sum()
                P = _compute_P(support, pert_w)
                if P > best_P:
                    best_P = P
                    best_weights = pert_w.copy()
            
            # Perturb support
            pert_s = support.copy()
            idx = np.random.randint(1, n-1)  # don't move 0 or 1
            pert_s[idx] += 0.02 * np.random.randn()
            pert_s = np.sort(pert_s)
            if pert_s[0] >= 0 and pert_s[-1] <= 1.0:
                P = _compute_P(pert_s, best_weights)
                if P > best_P:
                    best_P = P
                    best_support = pert_s.copy()
    
    support = best_support.astype(np.float64)
    weights = best_weights.astype(np.float64)
    
    # Ensure validity
    assert len(support) == len(weights)
    assert len(support) >= 1
    assert np.all(np.isfinite(support))
    assert np.all(np.isfinite(weights))
    assert np.all(support >= -1e-9)
    assert np.all(weights >= -1e-9)
    assert np.sum(weights) > 1e-9
    
    return support, weights

# EVOLVE-BLOCK-END
