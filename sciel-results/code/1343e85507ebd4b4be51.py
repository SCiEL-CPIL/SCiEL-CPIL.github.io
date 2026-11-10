import numpy as np
import scipy.optimize as optimize
import time
from itertools import product

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0

    def calc_prob(support, weights):
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pairwise sums and weights for Z = X1 + X2
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        p_prefix = np.cumsum(p_sorted)
        
        # Thresholds for X3, X4
        s_k_grid = support[:, None]
        s_l_grid = support[None, :]
        thresholds = 2.0 * s_l_grid - s_k_grid
        
        # Searchsorted on sorted s_sums
        idxs = np.searchsorted(s_sorted, thresholds, side='left')
        
        prob_z_less = np.zeros_like(idxs, dtype=np.float64)
        mask = idxs > 0
        if np.any(mask):
            prob_z_less[mask] = p_prefix[np.maximum(idxs[mask] - 1, 0)]
            
        p_k_grid = p[:, None]
        p_l_grid = p[None, :]
        
        total_prob = np.sum(p_k_grid * p_l_grid * prob_z_less)
        
        return float(total_prob)

    def calc_prob_and_grad(support, weights):
        """
        Computes P and gradient dP/dw.
        support: (N,)
        weights: (N,)
        Returns: P, grad (N,)
        """
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0, np.zeros(n)
        p = weights / w_sum
        
        # 1. Compute C_kl = P(Z < 2s_l - s_k)
        s_sums = support[:, None] + support[None, :]
        s_sums_flat = s_sums.flatten()
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        
        s_k_grid = support[:, None]
        s_l_grid = support[None, :]
        thresholds = 2.0 * s_l_grid - s_k_grid
        
        idxs = np.searchsorted(s_sorted, thresholds, side='left')
        
        # p_prefix for Z
        p_prods = p[:, None] * p[None, :]
        p_prods_flat = p_prods.flatten()
        p_sorted = p_prods_flat[sort_idx]
        p_prefix = np.cumsum(p_sorted)
        
        prob_z_less = np.zeros_like(idxs, dtype=np.float64)
        mask = idxs > 0
        if np.any(mask):
            prob_z_less[mask] = p_prefix[np.maximum(idxs[mask] - 1, 0)]
            
        C = prob_z_less # C[k,l]
        
        P = np.sum(p[:, None] * p[None, :] * C)
        
        # 2. Compute Gradient
        # Term 1: sum_l w_l C_ml + sum_k w_k C_km
        # C is not symmetric in k,l generally?
        # C_kl = P(Z < 2s_l - s_k).
        # Term 1_m = sum_l p_l C_ml + sum_k p_k C_km
        term1 = np.sum(p[None, :] * C, axis=1) + np.sum(p[:, None] * C, axis=0)
        
        # Term 2: G_m = 2 * sum_j p_j H_mj
        # H_mj = P(2X4 - X3 > s_m + s_j)
        # Let Y = 2X4 - X3.
        # Values of Y: y_kl = 2s_l - s_k.
        # Prob of Y: p_y_kl = p_k p_l.
        
        y_vals = 2.0 * s_l_grid - s_k_grid # (N, N)
        y_probs = p[:, None] * p[None, :] # (N, N)
        
        y_flat = y_vals.flatten()
        y_probs_flat = y_probs.flatten()
        
        sort_y_idx = np.argsort(y_flat)
        y_sorted = y_flat[sort_y_idx]
        y_probs_sorted = y_probs_flat[sort_y_idx]
        
        # We need P(Y > t) for t = s_m + s_j.
        # Cumulative sum from right
        y_csum_right = np.cumsum(y_probs_sorted[::-1])[::-1]
        
        # Thresholds for H
        s_m_grid = support[:, None]
        s_j_grid = support[None, :]
        h_thresholds = s_m_grid + s_j_grid # (N, N)
        
        # Find index where y_sorted > h_threshold
        # searchsorted gives first index where y_sorted >= val
        # We want strictly greater.
        # If we use side='right', it gives first index where y_sorted > val? No.
        # side='left' gives first index where y_sorted >= val.
        # So indices from idx to end are >= val.
        # We need > val.
        # If there are duplicates, we need to be careful.
        # Using side='right' gives first index where y_sorted > val? 
        # Actually, np.searchsorted(a, v, side='right') returns index i such that a[i-1] <= v < a[i].
        # So a[i:] are > v.
        idxs_y = np.searchsorted(y_sorted, h_thresholds, side='right')
        
        H = np.zeros_like(idxs_y, dtype=np.float64)
        mask_y = idxs_y < n*n
        if np.any(mask_y):
            H[mask_y] = y_csum_right[np.maximum(idxs_y[mask_y], 0)]
            # Note: y_csum_right[i] is sum of y_probs_sorted[i:]
            # So if idxs_y is the start of >, then sum from idxs_y is correct.
            
        # G_m = 2 * sum_j p_j H_mj
        G = 2.0 * (H @ p)
        
        grad = term1 + G
        
        # Chain rule: we optimized p, but we need grad w.r.t w?
        # p = w / sum(w).
        # If we treat w as unconstrained and project, it's easier to work with p on simplex directly.
        # The gradient computed above is dP/dp.
        # We will do gradient ascent on p (simplex constrained).
        
        return float(P), grad

    def project_simplex(v):
        """Project v onto the probability simplex."""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    def optimize_large_grid(N, max_iters, lr_init=0.1):
        """
        Gradient ascent on a fixed grid of size N.
        """
        # Support points: 0, 1/N, ..., 1
        support = np.linspace(0.0, 1.0, N)
        
        # Initial weights: uniform
        p = np.ones(N) / N
        
        best_p = p.copy()
        best_score = calc_prob(support, best_p)
        
        lr = lr_init
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            score, grad = calc_prob_and_grad(support, p)
            
            if score > best_score:
                best_score = score
                best_p = p.copy()
                
            # Gradient step
            # grad is dP/dp
            p_new = p + lr * grad
            
            # Project to simplex
            p_new = project_simplex(p_new)
            
            # Check if improvement
            score_new = calc_prob(support, p_new)
            
            if score_new > score + 1e-10:
                p = p_new
                # Maybe decay lr slightly or keep
            else:
                # Backtrack
                lr *= 0.5
                if lr < 1e-6:
                    break
            
        return support, best_p, best_score

    def polish_local(support, weights, max_iters=200, step_init=0.01):
        """Local refinement with adaptive step size and multi-directional search."""
        n = len(support)
        best_score = calc_prob(support, weights)
        best_sup = support.copy()
        best_w = weights.copy()
        step = step_init
        stall_count = 0
        for it in range(max_iters):
            improved = False
            # Try multiple step sizes for each coordinate
            for i in range(n):
                # Try perturbing support point i
                if i > 0 and i < n - 1:
                    for mult in [1.0, 0.5, 2.0]:
                        for direction in [-1, 1]:
                            new_sup = best_sup.copy()
                            new_sup[i] += direction * step * mult
                            if new_sup[i] < 1e-8 or new_sup[i] > 1 - 1e-8:
                                continue
                            if i > 0 and new_sup[i] <= new_sup[i-1] + 1e-12:
                                continue
                            if i < n-1 and new_sup[i] >= new_sup[i+1] - 1e-12:
                                continue
                            s = calc_prob(new_sup, best_w)
                            if s > best_score + 1e-14:
                                best_score = s
                                best_sup = new_sup.copy()
                                improved = True
                # Try perturbing weight i
                for mult in [1.0, 0.5, 2.0]:
                    for direction in [-1, 1]:
                        new_w = best_w.copy()
                        new_w[i] += direction * step * mult
                        if new_w[i] < 1e-8:
                            continue
                        w_sum = np.sum(new_w)
                        if w_sum <= 1e-8:
                            continue
                        new_w = new_w / w_sum
                        s = calc_prob(best_sup, new_w)
                        if s > best_score + 1e-14:
                            best_score = s
                            best_w = new_w.copy()
                            improved = True
            if improved:
                stall_count = 0
            else:
                stall_count += 1
                step *= 0.5
                if step < 1e-10:
                    break
                if stall_count > 15:
                    break
        return best_sup, best_w, best_score

    def polish_high_precision(support, weights, max_iters=500, step_init=1e-4):
        """High-precision polish with very small steps for final refinement."""
        n = len(support)
        best_score = calc_prob(support, weights)
        best_sup = support.copy()
        best_w = weights.copy()
        step = step_init
        for it in range(max_iters):
            improved = False
            for i in range(n):
                if i > 0 and i < n - 1:
                    for direction in [-1, 1]:
                        new_sup = best_sup.copy()
                        new_sup[i] += direction * step
                        if new_sup[i] < 1e-10 or new_sup[i] > 1 - 1e-10:
                            continue
                        if i > 0 and new_sup[i] <= new_sup[i-1] + 1e-14:
                            continue
                        if i < n-1 and new_sup[i] >= new_sup[i+1] - 1e-14:
                            continue
                        s = calc_prob(new_sup, best_w)
                        if s > best_score + 1e-16:
                            best_score = s
                            best_sup = new_sup.copy()
                            improved = True
                for direction in [-1, 1]:
                    new_w = best_w.copy()
                    new_w[i] += direction * step
                    if new_w[i] < 1e-10:
                        continue
                    w_sum = np.sum(new_w)
                    if w_sum <= 1e-10:
                        continue
                    new_w = new_w / w_sum
                    s = calc_prob(best_sup, new_w)
                    if s > best_score + 1e-16:
                        best_score = s
                        best_w = new_w.copy()
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-12:
                    break
        return best_sup, best_w, best_score

    best_score = -1.0
    best_support = None
    best_weights = None

    # 1. Try Large Grid Gradient Ascent
    # This explores a dense space, likely finding a better global structure than sparse points
    if time.time() < deadline:
        N_grid = 64
        sup_g, w_g, sc_g = optimize_large_grid(N_grid, max_iters=1000, lr_init=0.05)
        if sc_g > best_score:
            best_score = sc_g
            best_support = sup_g.copy()
            best_weights = w_g.copy()
            
        # Polish the grid result? 
        # The support is fixed in grid optimization. 
        # We can try to remove points with near-zero weight and polish the rest.
        if best_weights is not None and time.time() < deadline - 30:
            # Filter out small weights
            thresh = 1e-4
            mask = best_weights > thresh
            if np.sum(mask) > 1:
                sup_reduced = best_support[mask]
                w_reduced = best_weights[mask]
                w_reduced = w_reduced / np.sum(w_reduced)
                
                # Polish reduced set
                sup_pol, wgt_pol, sc_pol = polish_local(sup_reduced, w_reduced, max_iters=100)
                if sc_pol > best_score:
                    best_score = sc_pol
                    best_support = sup_pol.copy()
                    best_weights = wgt_pol.copy()
                    
                # High precision polish
                if time.time() < deadline - 10:
                    sup_hp, wgt_hp, sc_hp = polish_high_precision(best_support, best_weights, max_iters=200, step_init=1e-5)
                    if sc_hp > best_score:
                        best_score = sc_hp
                        best_support = sup_hp.copy()
                        best_weights = wgt_hp.copy()

    # 2. Fallback / Alternative: Small N Optimization (keep from original)
    # Only run if grid didn't take too much time or if we want to compare
    if time.time() < deadline - 30:
        def objective_n4(params):
            # Support [0, a1, a2, 1]
            p0, p1, p2, a1, a2 = params
            if (p0 < 1e-9 or p1 < 1e-9 or p2 < 1e-9 or 
                p0 + p1 + p2 > 1 - 1e-9 or 
                a1 < 1e-9 or a2 < 1e-9 or a1 > 1 - 1e-9 or a2 > 1 - 1e-9):
                return -0.0
            support = np.array([0.0, a1, a2, 1.0])
            weights = np.array([p0, p1, p2, 1.0 - p0 - p1 - p2])
            return -calc_prob(support, weights)

        bounds4 = [(1e-6, 0.5), (1e-6, 0.5), (1e-6, 0.5), (1e-6, 0.45), (0.45, 0.999)]
        res4 = optimize.differential_evolution(objective_n4, bounds=bounds4, seed=42, maxiter=200, tol=1e-8, polish=True)
        score4 = -res4.fun
        p0, p1, p2, a1, a2 = res4.x
        if (p0 > 1e-9 and p1 > 1e-9 and p2 > 1e-9 and 
            p0+p1+p2 < 1-1e-9 and 
            a1 > 1e-9 and a2 > 1e-9 and a1 < 1-1e-9 and a2 < 1-1e-9 and a1 < a2):
            support4 = np.array([0.0, a1, a2, 1.0])
            weights4 = np.array([p0, p1, p2, 1.0 - p0 - p1 - p2])
            s = calc_prob(support4, weights4)
            if s > best_score:
                best_score = s
                best_support = support4.copy()
                best_weights = weights4.copy()
            if time.time() < deadline - 10:
                sup_pol, wgt_pol, sc_pol = polish_local(support4, weights4, max_iters=50)
                if sc_pol > best_score:
                    best_score = sc_pol
                    best_support = sup_pol.copy()
                    best_weights = wgt_pol.copy()

    # If no valid result found, fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    # Ensure non-negative and normalized
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    sum_w = np.sum(best_weights)
    if sum_w > 1e-9:
        best_weights = best_weights / sum_w
    else:
        best_weights = np.ones(len(best_support)) / len(best_support)

    # Final polish with remaining time
    remaining = deadline - time.time()
    if remaining > 5 and len(best_support) <= 20: # Only polish if small enough
        sup_hp, wgt_hp, sc_hp = polish_high_precision(best_support, best_weights, max_iters=min(100, int(remaining*10)), step_init=1e-5)
        if sc_hp > best_score:
            best_support = sup_hp
            best_weights = wgt_hp
            best_score = sc_hp
        best_weights = np.maximum(best_weights, 0)
        sw = np.sum(best_weights)
        if sw > 1e-9:
            best_weights = best_weights / sw

    return best_support, best_weights
