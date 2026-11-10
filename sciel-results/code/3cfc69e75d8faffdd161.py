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
        # Condition: X1 + X2 < 2*X4 - X3
        # Let s_k be value of X3, s_l be value of X4.
        # Threshold is 2*s_l - s_k.
        # We sum p_k * p_l * P(Z < 2*s_l - s_k)
        
        s_k_grid = support[:, None]
        s_l_grid = support[None, :]
        thresholds = 2.0 * s_l_grid - s_k_grid
        
        # Searchsorted on sorted s_sums
        # Number of elements < threshold
        idxs = np.searchsorted(s_sorted, thresholds, side='left')
        
        prob_z_less = np.zeros_like(idxs, dtype=np.float64)
        mask = idxs > 0
        if np.any(mask):
            prob_z_less[mask] = p_prefix[np.maximum(idxs[mask] - 1, 0)]
            
        p_k_grid = p[:, None]
        p_l_grid = p[None, :]
        
        total_prob = np.sum(p_k_grid * p_l_grid * prob_z_less)
        
        return float(total_prob)

    def objective_n2(params):
        # Support fixed at [0, 1]
        # Params: [p_0]
        # p_1 = 1 - p_0
        p0 = params[0]
        if p0 < 1e-9 or p0 > 1 - 1e-9:
            return -0.0
        support = np.array([0.0, 1.0])
        weights = np.array([p0, 1.0 - p0])
        return -calc_prob(support, weights)

    def objective_n3(params):
        # Support [0, a, 1]
        # Params: [p_0, p_a, a]
        # p_1 = 1 - p_0 - p_a
        p0, pa, a = params
        if p0 < 1e-9 or pa < 1e-9 or p0 + pa > 1 - 1e-9 or a < 1e-9 or a > 1 - 1e-9:
            return -0.0
        support = np.array([0.0, a, 1.0])
        weights = np.array([p0, pa, 1.0 - p0 - pa])
        return -calc_prob(support, weights)

    def objective_n4(params):
        # Support [0, a1, a2, 1]
        # Params: [p_0, p_a1, p_a2, a1, a2]
        # p_1 = 1 - p_0 - p_a1 - p_a2
        p0, p1, p2, a1, a2 = params
        if (p0 < 1e-9 or p1 < 1e-9 or p2 < 1e-9 or 
            p0 + p1 + p2 > 1 - 1e-9 or 
            a1 < 1e-9 or a2 < 1e-9 or a1 > 1 - 1e-9 or a2 > 1 - 1e-9):
            return -0.0
        support = np.array([0.0, a1, a2, 1.0])
        weights = np.array([p0, p1, p2, 1.0 - p0 - p1 - p2])
        return -calc_prob(support, weights)

    def objective_n5(params):
        # Support [0, a1, a2, a3, 1]
        # Params: [p_0, p_a1, p_a2, p_a3, a1, a2, a3]
        # p_1 = 1 - p_0 - p_a1 - p_a2 - p_a3
        p0, p1, p2, p3, a1, a2, a3 = params
        if (p0 < 1e-9 or p1 < 1e-9 or p2 < 1e-9 or p3 < 1e-9 or 
            p0 + p1 + p2 + p3 > 1 - 1e-9 or 
            a1 < 1e-9 or a2 < 1e-9 or a3 < 1e-9 or 
            a1 > 1 - 1e-9 or a2 > 1 - 1e-9 or a3 > 1 - 1e-9):
            return -0.0
        support = np.array([0.0, a1, a2, a3, 1.0])
        weights = np.array([p0, p1, p2, p3, 1.0 - p0 - p1 - p2 - p3])
        return -calc_prob(support, weights)

    def objective_n6(params):
        # Support [0, a1, a2, a3, a4, 1]
        # Params: [p_0, p_a1, p_a2, p_a3, p_a4, a1, a2, a3, a4]
        # p_1 = 1 - sum(p_0..p_a4)
        p0, p1, p2, p3, p4, a1, a2, a3, a4 = params
        if (p0 < 1e-9 or p1 < 1e-9 or p2 < 1e-9 or p3 < 1e-9 or p4 < 1e-9 or 
            p0 + p1 + p2 + p3 + p4 > 1 - 1e-9 or 
            a1 < 1e-9 or a2 < 1e-9 or a3 < 1e-9 or a4 < 1e-9 or 
            a1 > 1 - 1e-9 or a2 > 1 - 1e-9 or a3 > 1 - 1e-9 or a4 > 1 - 1e-9):
            return -0.0
        support = np.array([0.0, a1, a2, a3, a4, 1.0])
        weights = np.array([p0, p1, p2, p3, p4, 1.0 - p0 - p1 - p2 - p3 - p4])
        return -calc_prob(support, weights)

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

    # Optimize n=2
    if time.time() < deadline:
        res2 = optimize.differential_evolution(objective_n2, bounds=[(1e-6, 1-1e-6)], seed=42, maxiter=1000, tol=1e-8, polish=True)
        score2 = -res2.fun
        support2 = np.array([0.0, 1.0])
        weights2 = np.array([res2.x[0], 1.0 - res2.x[0]])
        if score2 > best_score:
            best_score = score2
            best_support = support2.copy()
            best_weights = weights2.copy()

    # Optimize n=3
    if time.time() < deadline:
        bounds3 = [(1e-6, 0.9), (1e-6, 0.9), (1e-6, 0.9)]
        res3 = optimize.differential_evolution(objective_n3, bounds=bounds3, seed=42, maxiter=1000, tol=1e-8, polish=True)
        score3 = -res3.fun
        p0, pa, a = res3.x
        if p0 > 1e-9 and pa > 1e-9 and p0+pa < 1-1e-9 and a > 1e-9 and a < 1-1e-9:
            support3 = np.array([0.0, a, 1.0])
            weights3 = np.array([p0, pa, 1.0 - p0 - pa])
            s = calc_prob(support3, weights3)
            if s > best_score:
                best_score = s
                best_support = support3.copy()
                best_weights = weights3.copy()

    # Optimize n=4
    if time.time() < deadline:
        bounds4 = [(1e-6, 0.5), (1e-6, 0.5), (1e-6, 0.5), (1e-6, 0.45), (0.45, 0.999)]
        res4 = optimize.differential_evolution(objective_n4, bounds=bounds4, seed=42, maxiter=1000, tol=1e-8, polish=True)
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
            # Polish
            if time.time() < deadline - 30:
                sup_pol, wgt_pol, sc_pol = polish_local(support4, weights4, max_iters=150)
                if sc_pol > best_score:
                    best_score = sc_pol
                    best_support = sup_pol.copy()
                    best_weights = wgt_pol.copy()
                # High-precision polish
                if time.time() < deadline - 15:
                    sup_hp, wgt_hp, sc_hp = polish_high_precision(best_support, best_weights, max_iters=200, step_init=1e-4)
                    if sc_hp > best_score:
                        best_score = sc_hp
                        best_support = sup_hp.copy()
                        best_weights = wgt_hp.copy()

    # Optimize n=5
    if time.time() < deadline - 60:
        bounds5 = [(1e-6, 0.4), (1e-6, 0.4), (1e-6, 0.4), (1e-6, 0.4), 
                   (0.01, 0.35), (0.15, 0.65), (0.45, 0.95)]
        res5 = optimize.differential_evolution(objective_n5, bounds=bounds5, seed=42, maxiter=800, tol=1e-8, polish=True)
        score5 = -res5.fun
        p0, p1, p2, p3, a1, a2, a3 = res5.x
        if (p0 > 1e-9 and p1 > 1e-9 and p2 > 1e-9 and p3 > 1e-9 and 
            p0+p1+p2+p3 < 1-1e-9 and 
            a1 > 1e-9 and a2 > 1e-9 and a3 > 1e-9 and 
            a1 < 1-1e-9 and a2 < 1-1e-9 and a3 < 1-1e-9 and a1 < a2 < a3):
            support5 = np.array([0.0, a1, a2, a3, 1.0])
            weights5 = np.array([p0, p1, p2, p3, 1.0 - p0 - p1 - p2 - p3])
            s = calc_prob(support5, weights5)
            if s > best_score:
                best_score = s
                best_support = support5.copy()
                best_weights = weights5.copy()
            # Polish
            if time.time() < deadline - 30:
                sup_pol, wgt_pol, sc_pol = polish_local(support5, weights5, max_iters=150)
                if sc_pol > best_score:
                    best_score = sc_pol
                    best_support = sup_pol.copy()
                    best_weights = wgt_pol.copy()
                # High-precision polish
                if time.time() < deadline - 15:
                    sup_hp, wgt_hp, sc_hp = polish_high_precision(best_support, best_weights, max_iters=200, step_init=1e-4)
                    if sc_hp > best_score:
                        best_score = sc_hp
                        best_support = sup_hp.copy()
                        best_weights = wgt_hp.copy()

    # Optimize n=6
    if time.time() < deadline - 120:
        bounds6 = [(1e-6, 0.3), (1e-6, 0.3), (1e-6, 0.3), (1e-6, 0.3), (1e-6, 0.3),
                   (0.01, 0.25), (0.2, 0.45), (0.45, 0.7), (0.7, 0.98)]
        res6 = optimize.differential_evolution(objective_n6, bounds=bounds6, seed=42, maxiter=600, tol=1e-8, polish=True)
        score6 = -res6.fun
        p0, p1, p2, p3, p4, a1, a2, a3, a4 = res6.x
        if (p0 > 1e-9 and p1 > 1e-9 and p2 > 1e-9 and p3 > 1e-9 and p4 > 1e-9 and 
            p0+p1+p2+p3+p4 < 1-1e-9 and 
            a1 > 1e-9 and a2 > 1e-9 and a3 > 1e-9 and a4 > 1e-9 and 
            a1 < 1-1e-9 and a2 < 1-1e-9 and a3 < 1-1e-9 and a4 < 1-1e-9 and 
            a1 < a2 < a3 < a4):
            support6 = np.array([0.0, a1, a2, a3, a4, 1.0])
            weights6 = np.array([p0, p1, p2, p3, p4, 1.0 - p0 - p1 - p2 - p3 - p4])
            s = calc_prob(support6, weights6)
            if s > best_score:
                best_score = s
                best_support = support6.copy()
                best_weights = weights6.copy()
            # Polish
            if time.time() < deadline - 20:
                sup_pol, wgt_pol, sc_pol = polish_local(support6, weights6, max_iters=120)
                if sc_pol > best_score:
                    best_score = sc_pol
                    best_support = sup_pol.copy()
                    best_weights = wgt_pol.copy()
                # High-precision polish
                if time.time() < deadline - 10:
                    sup_hp, wgt_hp, sc_hp = polish_high_precision(best_support, best_weights, max_iters=300, step_init=1e-4)
                    if sc_hp > best_score:
                        best_score = sc_hp
                        best_support = sup_hp.copy()
                        best_weights = wgt_hp.copy()

    # If no valid result found (should not happen), fallback
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
    if remaining > 10:
        # First do a medium-precision polish
        max_polish1 = min(150, int(remaining * 3))
        if max_polish1 > 0:
            sup_final, wgt_final, sc_final = polish_local(best_support, best_weights, max_iters=max_polish1, step_init=0.005)
            if sc_final > best_score:
                best_support = sup_final
                best_weights = wgt_final
                best_score = sc_final
            # Re-normalize
            best_weights = np.maximum(best_weights, 0)
            sw = np.sum(best_weights)
            if sw > 1e-9:
                best_weights = best_weights / sw
        
        # Then do a high-precision polish with remaining time
        remaining2 = deadline - time.time()
        if remaining2 > 5:
            max_polish2 = min(500, int(remaining2 * 8))
            if max_polish2 > 0:
                sup_hp, wgt_hp, sc_hp = polish_high_precision(best_support, best_weights, max_iters=max_polish2, step_init=1e-5)
                if sc_hp > best_score:
                    best_support = sup_hp
                    best_weights = wgt_hp
                    best_score = sc_hp
                # Final re-normalization
                best_weights = np.maximum(best_weights, 0)
                sw = np.sum(best_weights)
                if sw > 1e-9:
                    best_weights = best_weights / sw

    return best_support, best_weights
