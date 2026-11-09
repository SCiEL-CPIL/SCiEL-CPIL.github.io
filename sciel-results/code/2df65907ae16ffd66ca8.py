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

    # Local refinement with Nelder-Mead on best solution found
    if time.time() < deadline:
        # Refine the best solution found so far
        n_best = len(best_support)
        if n_best == 3:
            # Refine 3-point
            def refine3(params):
                t, w1, w2 = params
                w0 = 1.0 - w1 - w2
                if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or t < -1e-9 or t > 1 + 1e-9:
                    return 1.0
                s = np.array([0.0, t, 1.0])
                w = np.array([w0, w1, w2])
                return -calculate_P_fast(s, w)
            from scipy.optimize import minimize
            res_ref = minimize(refine3, best_weights[1:].tolist() if len(best_weights) > 1 else [0.3, 0.3],
                               method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if -res_ref.fun > best_P:
                t, w1, w2 = res_ref.x
                w0 = 1.0 - w1 - w2
                if w0 > 0 and w1 > 0 and w2 > 0:
                    best_P = -res_ref.fun
                    best_support = np.array([0.0, t, 1.0])
                    best_weights = np.array([w0, w1, w2])
        elif n_best == 4:
            def refine4(params):
                a, b, w1, w2, w3 = params
                w0 = 1.0 - w1 - w2 - w3
                if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or w3 < -1e-9:
                    return 1.0
                if a < -1e-9 or b < -1e-9 or a >= b - 1e-9 or b > 1 + 1e-9:
                    return 1.0
                s = np.array([0.0, a, b, 1.0])
                w = np.array([w0, w1, w2, w3])
                return -calculate_P_fast(s, w)
            from scipy.optimize import minimize
            res_ref = minimize(refine4, best_support[1:3].tolist() + best_weights[1:4].tolist(),
                               method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if -res_ref.fun > best_P:
                a, b, w1, w2, w3 = res_ref.x
                w0 = 1.0 - w1 - w2 - w3
                if w0 > 0 and w1 > 0 and w2 > 0 and w3 > 0 and a < b:
                    best_P = -res_ref.fun
                    best_support = np.array([0.0, a, b, 1.0])
                    best_weights = np.array([w0, w1, w2, w3])

    support = best_support.astype(np.float64)
    weights = best_weights.astype(np.float64)
    support = np.maximum(support, 0.0)
    weights = np.maximum(weights, 0.0)

    return support, weights
