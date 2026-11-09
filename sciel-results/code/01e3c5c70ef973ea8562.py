import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where D is entropy of X-Y and M is max entropy of X, X+Y, X+2Y.
    Uses anti-diagonal weight parameterization + full simplex refinement.
    """
    n = 10
    start_time = time.time()
    timeout = 600
    
    # Precompute projection indices
    x_coords, y_coords = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    # P[y, x]
    v_diff = (x_coords - y_coords).flatten()
    v_x = x_coords.flatten()
    v_sum = (x_coords + y_coords).flatten()
    v_2y = (x_coords + 2 * y_coords).flatten()
    
    # Precompute unique values and bin indices
    diff_vals, diff_inv = np.unique(v_diff, return_inverse=True)
    x_vals, x_inv = np.unique(v_x, return_inverse=True)
    sum_vals, sum_inv = np.unique(v_sum, return_inverse=True)
    twoy_vals, twoy_inv = np.unique(v_2y, return_inverse=True)
    
    def _clip_and_normalize(P):
        P = np.clip(P, 0.0, None)
        total = P.sum()
        if total <= 1e-9:
            return np.ones((10, 10)) / 100.0
        return P / total
    
    def _compute_ratio(P):
        """Compute D/M ratio for a given 10x10 distribution P[y, x]."""
        P_flat = P.flatten()
        
        def _entropy_from_bins(inv_idx, n_bins, probs):
            grouped = np.zeros(n_bins)
            np.add.at(grouped, inv_idx, probs)
            mask = grouped > 1e-15
            if not np.any(mask):
                return 0.0
            gp = grouped[mask]
            return -np.sum(gp * np.log2(gp))
        
        D = _entropy_from_bins(diff_inv, len(diff_vals), P_flat)
        H0 = _entropy_from_bins(x_inv, len(x_vals), P_flat)
        H1 = _entropy_from_bins(sum_inv, len(sum_vals), P_flat)
        H2 = _entropy_from_bins(twoy_inv, len(twoy_vals), P_flat)
        
        M = max(H0, H1, H2)
        if M <= 1e-9:
            return 0.0
        return D / M
    
    def _anti_diag_weights_to_P(weights):
        """Convert anti-diagonal weights (19 values for x-y from -9 to 9) to a 10x10 P."""
        P = np.zeros((10, 10))
        # Anti-diagonals: for d = x - y, cells where x - y = d
        # d ranges from -9 to 9
        for d in range(-9, 10):
            w = weights[d + 9]
            if w <= 0:
                continue
            # Cells with x - y = d: x = y + d, y from max(0, -d) to min(9, 9-d)
            y_min = max(0, -d)
            y_max = min(9, 9 - d)
            count = y_max - y_min + 1
            if count > 0:
                for y in range(y_min, y_max + 1):
                    x = y + d
                    P[y, x] = w / count
        total = P.sum()
        if total > 1e-9:
            P /= total
        else:
            P = np.ones((10, 10)) / 100.0
        return P
    
    def _neg_ratio_from_weights(weights):
        """Objective: minimize negative ratio given anti-diagonal weights."""
        if time.time() - start_time > timeout - 30:
            return 1e10
        P = _anti_diag_weights_to_P(weights)
        return -_compute_ratio(P)
    
    # Phase 1: Optimize anti-diagonal weights
    # 19 weights, one for each anti-diagonal
    bounds = [(0.0, 1.0)] * 19
    
    best_score = -np.inf
    best_P = None
    
    # Try differential evolution for global search
    if time.time() - start_time < timeout - 120:
        try:
            de_res = differential_evolution(
                _neg_ratio_from_weights,
                bounds=bounds,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True
            )
            if de_res.fun < 1e9:
                P_de = _anti_diag_weights_to_P(de_res.x)
                score_de = _compute_ratio(P_de)
                if score_de > best_score:
                    best_score = score_de
                    best_P = P_de
        except Exception:
            pass
    
    # Phase 2: Full simplex refinement from best anti-diagonal solution
    if best_P is None:
        # Fallback: uniform
        best_P = np.ones((10, 10)) / 100.0
    
    def _neg_ratio_full(z):
        """Objective for full 100-param optimization in log-space."""
        if time.time() - start_time > timeout - 10:
            return 1e10
        P_unnorm = np.exp(z - np.max(z))
        P_flat = P_unnorm / np.sum(P_unnorm)
        P = P_flat.reshape((10, 10))
        return -_compute_ratio(P)
    
    # Refine from current best
    if best_P is not None and time.time() - start_time < timeout - 60:
        z0 = np.log(best_P.flatten() + 1e-15)
        try:
            res = minimize(_neg_ratio_full, z0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-10})
            if res.fun < 1e9:
                z_opt = res.x
                P_unnorm = np.exp(z_opt - np.max(z_opt))
                P_opt = P_unnorm / np.sum(P_unnorm)
                score_opt = _compute_ratio(P_opt.reshape((10, 10)))
                if score_opt > best_score:
                    best_score = score_opt
                    best_P = P_opt.reshape((10, 10))
        except Exception:
            pass
    
    # Phase 3: Try a few more targeted initializations
    if time.time() - start_time < timeout - 30:
        # Concentrate on a few anti-diagonals
        for trial in range(3):
            if time.time() - start_time > timeout - 20:
                break
            # Random sparse anti-diagonal weights
            w = np.zeros(19)
            idx = np.random.choice(19, size=3, replace=False)
            w[idx] = 1.0
            w /= w.sum()
            
            P_init = _anti_diag_weights_to_P(w)
            z0 = np.log(P_init.flatten() + 1e-15)
            try:
                res = minimize(_neg_ratio_full, z0, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-10})
                if res.fun < 1e9:
                    z_opt = res.x
                    P_unnorm = np.exp(z_opt - np.max(z_opt))
                    P_opt = P_unnorm / np.sum(P_unnorm)
                    score_opt = _compute_ratio(P_opt.reshape((10, 10)))
                    if score_opt > best_score:
                        best_score = score_opt
                        best_P = P_opt.reshape((10, 10))
            except Exception:
                pass
    
    # Final validation
    if best_P is None:
        best_P = np.ones((10, 10)) / 100.0
    
    best_P = _clip_and_normalize(best_P)
    return best_P
