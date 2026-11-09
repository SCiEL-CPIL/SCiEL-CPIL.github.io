import numpy as np
from scipy.optimize import differential_evolution
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 580s safety margin before 630s kill

    def calculate_P(support, weights):
        n = len(support)
        w = weights / weights.sum()
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                inner = 0.0
                for i in range(n):
                    for j in range(n):
                        if support[i] + support[j] < threshold:
                            inner += w[i] * w[j]
                P += w[k] * w[l] * inner
        return P

    def calculate_P_fast(support, weights):
        """Vectorized computation of P using numpy."""
        n = len(support)
        w = weights / weights.sum()
        # Build pairwise sums
        pair_sums = support[:, None] + support[None, :]
        # For each (k, l) pair, compute the inner probability
        # P = sum_{k,l} w_k * w_l * sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j
        # We can precompute the pairwise sum matrix
        # For each (k,l), threshold = 2*s_l - s_k
        # inner(k,l) = sum_{i,j} w_i*w_j * 1[s_i+s_j < threshold]
        # This is a 2D cumulative sum approach
        # Sort support and use cumulative sums
        # Actually for small n, just do it directly but vectorized
        # Create threshold matrix: T[k,l] = 2*s_l - s_k
        T = 2.0 * support[None, :] - support[:, None]  # shape (n, n), T[k,l]
        # For each (k,l), we need sum of w_i*w_j where pair_sums[i,j] < T[k,l]
        # pair_sums has shape (n,n), T has shape (n,n)
        # We need for each (k,l): sum over (i,j) of w_i*w_j * 1[pair_sums[i,j] < T[k,l]]
        # This is a 4D operation: for each k,l,i,j
        # Let's do it in chunks to save memory
        # Actually for n up to ~10, n^4 = 10000 which is fine
        if n <= 20:
            # Full 4D approach
            # pair_sums[i,j], T[k,l], w_i, w_j
            # inner[k,l] = sum_{i,j} w[i]*w[j] * (pair_sums[i,j] < T[k,l])
            # Shape: (n,n,n,n)
            ps = pair_sums[None, None, :, :]  # (1,1,n,n)
            th = T[:, :, None, None]  # (n,n,1,1)
            wi = w[None, None, :, None]  # (1,1,n,1)
            wj = w[None, None, None, :]  # (1,1,1,n)
            mask = ps < th  # (n,n,n,n)
            inner = np.sum(mask * wi * wj, axis=(2, 3))  # (n,n)
            P = np.sum(w[:, None] * w[None, :] * inner)
            return P
        else:
            # Fallback to loop
            return calculate_P(support, weights)

    # 2-point baseline
    p_opt = (1.0 + np.sqrt(33.0)) / 16.0
    support2 = np.array([0.0, 1.0])
    weights2 = np.array([1.0 - p_opt, p_opt])
    best_P = calculate_P_fast(support2, weights2)
    best_support = support2.copy()
    best_weights = weights2.copy()

    # 3-point: {0, t, 1} with weights (w0, w1, w2)
    def opt3(params):
        t, w1, w2 = params
        w0 = 1.0 - w1 - w2
        if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or t < -1e-9 or t > 1 + 1e-9:
            return 1.0
        s = np.array([0.0, t, 1.0])
        w = np.array([w0, w1, w2])
        return -calculate_P_fast(s, w)

    # Use differential evolution for 3-point
    bounds3 = [(1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0)]
    res3 = differential_evolution(opt3, bounds3, maxiter=300, tol=1e-9,
                                   popsize=20, seed=42, polish=True,
                                   init='sobol')
    if -res3.fun > best_P:
        best_P = -res3.fun
        t, w1, w2 = res3.x
        w0 = 1.0 - w1 - w2
        if w0 > 0 and w1 > 0 and w2 > 0:
            best_support = np.array([0.0, t, 1.0])
            best_weights = np.array([w0, w1, w2])

    # 4-point: {0, a, b, 1} with weights (w0, w1, w2, w3)
    def opt4(params):
        a, b, w1, w2, w3 = params
        w0 = 1.0 - w1 - w2 - w3
        if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or w3 < -1e-9:
            return 1.0
        if a < -1e-9 or b < -1e-9 or a >= b - 1e-9 or b > 1 + 1e-9:
            return 1.0
        s = np.array([0.0, a, b, 1.0])
        w = np.array([w0, w1, w2, w3])
        return -calculate_P_fast(s, w)

    # Use differential evolution for 4-point
    bounds4 = [(1e-6, 0.999), (0.001, 1.0), (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0)]
    res4 = differential_evolution(opt4, bounds4, maxiter=300, tol=1e-9,
                                   popsize=20, seed=42, polish=True,
                                   init='sobol')
    if -res4.fun > best_P:
        best_P = -res4.fun
        a, b, w1, w2, w3 = res4.x
        w0 = 1.0 - w1 - w2 - w3
        if w0 > 0 and w1 > 0 and w2 > 0 and w3 > 0 and a < b:
            best_support = np.array([0.0, a, b, 1.0])
            best_weights = np.array([w0, w1, w2, w3])

    # 5-point: {0, a, b, c, 1} with weights (w0, w1, w2, w3, w4)
    if time.time() < deadline:
        def opt5(params):
            a, b, c, w1, w2, w3, w4 = params
            w0 = 1.0 - w1 - w2 - w3 - w4
            if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or w3 < -1e-9 or w4 < -1e-9:
                return 1.0
            if a < -1e-9 or b < -1e-9 or c < -1e-9 or a >= b - 1e-9 or b >= c - 1e-9 or c > 1 + 1e-9:
                return 1.0
            s = np.array([0.0, a, b, c, 1.0])
            w = np.array([w0, w1, w2, w3, w4])
            return -calculate_P_fast(s, w)

        bounds5 = [(1e-6, 0.999), (0.001, 0.999), (0.002, 1.0),
                   (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0)]
        res5 = differential_evolution(opt5, bounds5, maxiter=200, tol=1e-9,
                                       popsize=15, seed=42, polish=True,
                                       init='sobol')
        if -res5.fun > best_P:
            best_P = -res5.fun
            a, b, c, w1, w2, w3, w4 = res5.x
            w0 = 1.0 - w1 - w2 - w3 - w4
            if w0 > 0 and w1 > 0 and w2 > 0 and w3 > 0 and w4 > 0 and a < b < c:
                best_support = np.array([0.0, a, b, c, 1.0])
                best_weights = np.array([w0, w1, w2, w3, w4])

    # 6-point: {0, a, b, c, d, 1} with weights (w0, w1, w2, w3, w4, w5)
    if time.time() < deadline:
        def opt6(params):
            a, b, c, d, w1, w2, w3, w4, w5 = params
            w0 = 1.0 - w1 - w2 - w3 - w4 - w5
            if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or w3 < -1e-9 or w4 < -1e-9 or w5 < -1e-9:
                return 1.0
            if a < -1e-9 or b < -1e-9 or c < -1e-9 or d < -1e-9:
                return 1.0
            if a >= b - 1e-9 or b >= c - 1e-9 or c >= d - 1e-9 or d > 1 + 1e-9:
                return 1.0
            s = np.array([0.0, a, b, c, d, 1.0])
            w = np.array([w0, w1, w2, w3, w4, w5])
            return -calculate_P_fast(s, w)

        bounds6 = [(1e-6, 0.999), (0.001, 0.999), (0.002, 0.999), (0.003, 1.0),
                   (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0)]
        res6 = differential_evolution(opt6, bounds6, maxiter=150, tol=1e-9,
                                       popsize=12, seed=42, polish=True,
                                       init='sobol')
        if -res6.fun > best_P:
            best_P = -res6.fun
            a, b, c, d, w1, w2, w3, w4, w5 = res6.x
            w0 = 1.0 - w1 - w2 - w3 - w4 - w5
            if w0 > 0 and w1 > 0 and w2 > 0 and w3 > 0 and w4 > 0 and w5 > 0 and a < b < c < d:
                best_support = np.array([0.0, a, b, c, d, 1.0])
                best_weights = np.array([w0, w1, w2, w3, w4, w5])

    # Local refinement with Nelder-Mead on best solution found
    if time.time() < deadline:
        # Refine the best solution found so far
        n_best = len(best_support)
        from scipy.optimize import minimize
        
        if n_best >= 3:
            # Generic refinement for n-point solutions
            # Parameters: interior support points (n-2 of them) + interior weights (n-1 of them)
            # w0 is determined by sum-to-1 constraint
            
            def make_refine_func(n_pts):
                def refine(params):
                    # params: [s1, s2, ..., s_{n-2}, w1, w2, ..., w_{n-1}]
                    n_int_s = n_pts - 2
                    n_int_w = n_pts - 1
                    s_int = params[:n_int_s]
                    w_int = params[n_int_s:]
                    
                    # Check constraints
                    if np.any(s_int < -1e-9) or np.any(s_int > 1 + 1e-9):
                        return 1.0
                    if np.any(w_int < -1e-9):
                        return 1.0
                    w0 = 1.0 - np.sum(w_int)
                    if w0 < -1e-9:
                        return 1.0
                    # Check sorted support
                    s = np.concatenate([[0.0], s_int, [1.0]])
                    if np.any(np.diff(s) < -1e-9):
                        return 1.0
                    w = np.concatenate([[w0], w_int])
                    return -calculate_P_fast(s, w)
                return refine
            
            # Initial guess: interior support points and weights from current best
            s_init = best_support[1:-1].tolist()
            w_init = best_weights[1:].tolist()
            init_params = s_init + w_init
            
            refine_func = make_refine_func(n_best)
            res_ref = minimize(refine_func, init_params,
                               method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-13,
                                        'adaptive': True})
            if -res_ref.fun > best_P:
                n_int_s = n_best - 2
                n_int_w = n_best - 1
                s_int = res_ref.x[:n_int_s]
                w_int = res_ref.x[n_int_s:]
                w0 = 1.0 - np.sum(w_int)
                s = np.concatenate([[0.0], s_int, [1.0]])
                w = np.concatenate([[w0], w_int])
                # Validate
                if np.all(s >= -1e-9) and np.all(w >= -1e-9) and np.sum(w) > 1e-9:
                    if np.all(np.diff(s) >= -1e-9):
                        best_P = -res_ref.fun
                        best_support = s
                        best_weights = w

    # Final high-precision polish: run Nelder-Mead again with tighter tolerances
    if time.time() < deadline - 30:
        n_final = len(best_support)
        if n_final >= 3:
            from scipy.optimize import minimize
            
            def make_final_refine(n_pts):
                def final_refine(params):
                    n_int_s = n_pts - 2
                    n_int_w = n_pts - 1
                    s_int = params[:n_int_s]
                    w_int = params[n_int_s:]
                    
                    if np.any(s_int < -1e-9) or np.any(s_int > 1 + 1e-9):
                        return 1.0
                    if np.any(w_int < -1e-9):
                        return 1.0
                    w0 = 1.0 - np.sum(w_int)
                    if w0 < -1e-9:
                        return 1.0
                    s = np.concatenate([[0.0], s_int, [1.0]])
                    if np.any(np.diff(s) < -1e-9):
                        return 1.0
                    w = np.concatenate([[w0], w_int])
                    return -calculate_P_fast(s, w)
                return final_refine
            
            s_init = best_support[1:-1].tolist()
            w_init = best_weights[1:].tolist()
            init_params = s_init + w_init
            
            final_refine_func = make_final_refine(n_final)
            res_final = minimize(final_refine_func, init_params,
                                 method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                                          'adaptive': True})
            if -res_final.fun > best_P:
                n_int_s = n_final - 2
                n_int_w = n_final - 1
                s_int = res_final.x[:n_int_s]
                w_int = res_final.x[n_int_s:]
                w0 = 1.0 - np.sum(w_int)
                s = np.concatenate([[0.0], s_int, [1.0]])
                w = np.concatenate([[w0], w_int])
                if np.all(s >= -1e-9) and np.all(w >= -1e-9) and np.sum(w) > 1e-9:
                    if np.all(np.diff(s) >= -1e-9):
                        best_P = -res_final.fun
                        best_support = s
                        best_weights = w

    support = best_support.astype(np.float64)
    weights = best_weights.astype(np.float64)
    support = np.maximum(support, 0.0)
    weights = np.maximum(weights, 0.0)

    # Ensure valid output
    if len(support) != len(weights) or len(support) < 1:
        # Fallback to 2-point
        p_opt = (1.0 + np.sqrt(33.0)) / 16.0
        support = np.array([0.0, 1.0])
        weights = np.array([1.0 - p_opt, p_opt])
    if np.any(np.isnan(support)) or np.any(np.isinf(support)) or np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        p_opt = (1.0 + np.sqrt(33.0)) / 16.0
        support = np.array([0.0, 1.0])
        weights = np.array([1.0 - p_opt, p_opt])
    if np.any(support < -1e-9):
        support = np.maximum(support, 0.0)
    if np.any(weights < -1e-9):
        weights = np.maximum(weights, 0.0)
    if np.sum(weights) <= 1e-9:
        p_opt = (1.0 + np.sqrt(33.0)) / 16.0
        support = np.array([0.0, 1.0])
        weights = np.array([1.0 - p_opt, p_opt])

    return support, weights
