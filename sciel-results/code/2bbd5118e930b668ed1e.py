# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] exactly.
    X1..X4 i.i.d. from discrete measure with support `support` and weights `weights` (normalized).
    P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i w_j)
    """
    n = len(support)
    s = support
    w = weights

    # Pairwise sums: s_i + s_j for all i,j
    # We need for each (k, l) the sum of w_i*w_j over (i,j) such that s_i+s_j < 2*s_l - s_k
    # Let T_kl = 2*s_l - s_k
    # For fixed l, we have n values of T_kl (as k varies).
    # For fixed k, we have n values of T_kl (as l varies).

    # Efficient approach:
    # Precompute all pairwise sums S_ij = s_i + s_j with weights W_ij = w_i * w_j
    # Then for each (k,l), we need sum of W_ij where S_ij < 2*s_l - s_k
    # This is a 2D problem.

    # Alternative: For each pair (i,j), it contributes w_i*w_j to P for all (k,l)
    # such that s_i+s_j < 2*s_l - s_k, i.e., s_k + s_i + s_j < 2*s_l.
    # So P = sum_{i,j} w_i*w_j * (sum_{k,l: s_k + (s_i+s_j) < 2*s_l} w_k * w_l)
    # Let U_ij = s_i + s_j. Then for each (i,j), we need:
    #   Q_ij = sum_{k,l: s_k < 2*s_l - U_ij} w_k * w_l
    # Then P = sum_{i,j} w_i*w_j * Q_ij

    # Even better: Precompute for each possible threshold T, the value
    #   F(T) = sum_{k,l: s_k + U < 2*s_l} ... this doesn't simplify directly.

    # Let's think differently. P = E[1{X1+X2+X3 < 2*X4}]
    # = sum_{x1,x2,x3,x4} w(x1)w(x2)w(x3)w(x4) 1{x1+x2+x3 < 2*x4}

    # For fixed x4, we need sum_{x1,x2,x3: x1+x2+x3 < 2*x4} w(x1)w(x2)w(x3)
    # Let G(t) = sum_{x1,x2,x3: x1+x2+x3 < t} w(x1)w(x2)w(x3)
    # Then P = sum_{x4} w(x4) * G(2*x4)

    # To compute G(t) for t = 2*x4 for each x4 in support:
    # G(t) = sum_{i,j,k: s_i+s_j+s_k < t} w_i*w_j*w_k

    # We can precompute the convolution of the distribution with itself twice.
    # Let A be the distribution. A*A is the distribution of X1+X2.
    # Then G(t) = sum_{m} (A*A)(m) * A_cumsum(t - m) ... this is getting complex.

    # Direct computation for small n:
    # n is at most a few hundred. n^4 is too slow for n=100 (10^8).
    # But n^3 is feasible for n=100 (10^6).

    # P = sum_{i,j,k,l} w_i*w_j*w_k*w_l * 1{s_i+s_j+s_k < 2*s_l}
    # = sum_l w_l * sum_{i,j,k: s_i+s_j+s_k < 2*s_l} w_i*w_j*w_k

    # For each l, we need H(2*s_l) where H(t) = sum_{i,j,k: s_i+s_j+s_k < t} w_i*w_j*w_k
    # H(t) can be computed by iterating over all triples (i,j,k) for each l,
    # but that's n^4 total. For n=50, that's 6.25M, which is fine.
    # For n=100, that's 100M, which might be slow in Python but OK in NumPy.

    # Let's vectorize:
    # Create arrays of all triple sums:
    # s_i + s_j + s_k for all i,j,k
    # This is an n^3 array. For n=100, that's 1M elements. OK.
    # Then for each l, we need sum of w_i*w_j*w_k where triple_sum < 2*s_l.

    # Actually, let's use the formula from the task:
    # P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)

    # Precompute pairwise sums:
    # pair_sums[i,j] = s_i + s_j
    # pair_weights[i,j] = w_i * w_j

    # For each (k,l), threshold = 2*s_l - s_k
    # We need sum of pair_weights[i,j] where pair_sums[i,j] < threshold

    # Let's sort pair_sums and use cumulative sums for pair_weights.
    # But the threshold varies with (k,l).

    # Alternative: For each l, and for each k, compute the inner sum.
    # That's n^2 thresholds, each requiring a sum over n^2 pairs. Total n^4.
    # For n=50: 6.25M. For n=100: 100M. Might be too slow in pure Python loops.

    # Let's use NumPy broadcasting for the inner sum:
    # For fixed l, thresholds_k = 2*s_l - s_k (array of length n)
    # For each k, inner_sum_k = sum_{i,j} w_i*w_j * (s_i+s_j < thresholds_k)
    # We can vectorize over k:
    #   pair_sums < thresholds_k[:, None, None]  # shape (n, n, n)
    #   inner_sums = np.sum(pair_weights[None, :, :] * (pair_sums[None, :, :] < thresholds_k[:, None, None]), axis=(1,2))
    # Then P = sum_k w_k * inner_sums_k * w_l ... wait, we also sum over l.

    # Let's do it fully vectorized:
    # pair_sums: (n, n)
    # pair_weights: (n, n)
    # thresholds[k, l] = 2*s_l - s_k  (shape (n, n))
    # For each (k,l), inner = sum_{i,j} pair_weights[i,j] * (pair_sums[i,j] < thresholds[k,l])
    # This is a (n, n, n, n) comparison. Too large for n=100 (100M booleans = 100MB).
    # For n=50, 6.25M booleans = 6.25MB. OK.

    # Let's do it in a loop over l to keep memory bounded:
    P_val = 0.0
    for l in range(n):
        thresholds_k = 2.0 * s[l] - s  # shape (n,)
        # For each k, inner_k = sum_{i,j} pair_weights[i,j] * (pair_sums[i,j] < thresholds_k[k])
        # Vectorize over k:
        # comparison shape: (n, n, n) -> pair_sums[None, :, :] < thresholds_k[:, None, None]
        if n <= 200:
            comp = pair_sums[None, :, :] < thresholds_k[:, None, None]  # (n, n, n)
            inner_sums = np.sum(pair_weights[None, :, :] * comp, axis=(1, 2))  # (n,)
            P_val += np.sum(w * inner_sums) * w[l]
        else:
            for k in range(n):
                inner = np.sum(pair_weights * (pair_sums < thresholds_k[k]))
                P_val += w[k] * inner * w[l]

    return P_val


def _neg_P(x):
    """Objective to minimize: -P. x = [support..., weights...]"""
    n = len(x) // 2
    support = x[:n]
    weights = x[n:]
    # Ensure non-negativity for support
    support = np.maximum(support, 0.0)
    # Weights should be positive; clip to small positive
    weights = np.maximum(weights, 1e-10)
    # Normalize
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 1.0
    weights = weights / w_sum
    try:
        P = _compute_P(support, weights)
    except Exception:
        return 1.0
    return -P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    deadline = time.time() + 600  # 600s limit, leave safety margin

    best_P = -1.0
    best_support = None
    best_weights = None

    # Start with a reasonable initial guess: uniform on {0, 1, ..., n-1}
    n = 32
    support0 = np.arange(n, dtype=np.float64)
    weights0 = np.ones(n, dtype=np.float64) / n

    P0 = _compute_P(support0, weights0)
    if P0 > best_P:
        best_P = P0
        best_support = support0.copy()
        best_weights = weights0.copy()

    # Try a few different n values
    for n_try in [16, 32, 48, 64]:
        if time.time() > deadline - 60:
            break
        s_init = np.arange(n_try, dtype=np.float64)
        w_init = np.ones(n_try, dtype=np.float64) / n_try
        P_try = _compute_P(s_init, w_init)
        if P_try > best_P:
            best_P = P_try
            best_support = s_init.copy()
            best_weights = w_init.copy()

    # Try some structured distributions
    # 1. Point mass at 0: P = P[0+0+0 < 0] = 0
    # 2. Two-point: {0, a} with weights {p, 1-p}
    #    X1+X2+X3 < 2*X4
    #    If X4=0: need X1+X2+X3 < 0, impossible (all >=0). Contribution: (1-p)*0 = 0
    #    If X4=a: need X1+X2+X3 < 2a.
    #    X1+X2+X3 can be 0, a, 2a, 3a with probabilities:
    #    P(sum=0) = p^3
    #    P(sum=a) = 3p^2(1-p)
    #    P(sum=2a) = 3p(1-p)^2
    #    P(sum=3a) = (1-p)^3
    #    Condition sum < 2a: sum=0 or sum=a
    #    P(sum < 2a) = p^3 + 3p^2(1-p) = p^2(p + 3 - 3p) = p^2(3 - 2p)
    #    Contribution from X4=a: (1-p) * p^2(3-2p)
    #    Total P = (1-p) * p^2 * (3 - 2p)
    #    Maximize f(p) = (1-p)p^2(3-2p) = p^2(3-2p)(1-p) = p^2(3 - 5p + 2p^2) = 3p^2 - 5p^3 + 2p^4
    #    f'(p) = 6p - 15p^2 + 8p^3 = p(6 - 15p + 8p^2)
    #    8p^2 - 15p + 6 = 0 -> p = (15 ± sqrt(225 - 192))/16 = (15 ± sqrt(33))/16
    #    sqrt(33) ≈ 5.7446
    #    p1 = (15 - 5.7446)/16 ≈ 9.2554/16 ≈ 0.5785
    #    p2 = (15 + 5.7446)/16 ≈ 20.7446/16 ≈ 1.2965 (invalid)
    #    So p ≈ 0.5785
    #    f(0.5785) = 0.5785^2 * (3 - 2*0.5785) * (1 - 0.5785)
    #    = 0.3347 * (3 - 1.157) * 0.4215
    #    = 0.3347 * 1.843 * 0.4215 ≈ 0.260

    p_opt = (15 - np.sqrt(33)) / 16
    a_val = 1.0
    support_2pt = np.array([0.0, a_val])
    weights_2pt = np.array([p_opt, 1 - p_opt])
    P_2pt = _compute_P(support_2pt, weights_2pt)
    if P_2pt > best_P:
        best_P = P_2pt
        best_support = support_2pt.copy()
        best_weights = weights_2pt.copy()

    # 3. Three-point: {0, a, 2a}
    # Let's try uniform on {0, 1, 2}
    support_3pt = np.array([0.0, 1.0, 2.0])
    weights_3pt = np.ones(3) / 3
    P_3pt = _compute_P(support_3pt, weights_3pt)
    if P_3pt > best_P:
        best_P = P_3pt
        best_support = support_3pt.copy()
        best_weights = weights_3pt.copy()

    # Try to optimize the two-point distribution more carefully
    # and also try three-point with optimization

    # Use scipy if available
    try:
        from scipy.optimize import minimize
        has_scipy = True
    except ImportError:
        has_scipy = False

    if has_scipy:
        # Optimize two-point
        def _two_point_P(params):
            p, a = params
            p = np.clip(p, 0.01, 0.99)
            a = max(a, 1e-6)
            s = np.array([0.0, a])
            w = np.array([p, 1 - p])
            return _compute_P(s, w)

        # Optimize three-point
        def _three_point_P(params):
            p1, p2, a = params
            p1 = np.clip(p1, 0.01, 0.98)
            p2 = np.clip(p2, 0.01, 0.98)
            p3 = 1 - p1 - p2
            if p3 <= 0.01:
                return -1.0
            a = max(a, 1e-6)
            s = np.array([0.0, a, 2*a])
            w = np.array([p1, p2, p3])
            return _compute_P(s, w)

        # Try optimizing two-point
        res2 = minimize(lambda x: -_two_point_P(x), [0.58, 1.0], method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res2.success or res2.fun < -best_P:
            p_opt2, a_opt2 = res2.x
            s_opt2 = np.array([0.0, a_opt2])
            w_opt2 = np.array([p_opt2, 1 - p_opt2])
            P_opt2 = _compute_P(s_opt2, w_opt2)
            if P_opt2 > best_P:
                best_P = P_opt2
                best_support = s_opt2.copy()
                best_weights = w_opt2.copy()

        # Try optimizing three-point
        if time.time() < deadline - 120:
            res3 = minimize(lambda x: -_three_point_P(x), [0.3, 0.4, 1.0], method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res3.success or res3.fun < -best_P:
                p1_opt, p2_opt, a_opt = res3.x
                p3_opt = 1 - p1_opt - p2_opt
                if p3_opt > 0.01:
                    s_opt3 = np.array([0.0, a_opt, 2*a_opt])
                    w_opt3 = np.array([p1_opt, p2_opt, p3_opt])
                    P_opt3 = _compute_P(s_opt3, w_opt3)
                    if P_opt3 > best_P:
                        best_P = P_opt3
                        best_support = s_opt3.copy()
                        best_weights = w_opt3.copy()

    # Try a geometric-like distribution
    # P(X=k) = (1-q)q^k for k=0,1,2,...
    # Truncate at some max_k
    for max_k in [10, 20, 50]:
        if time.time() > deadline - 60:
            break
        for q in [0.3, 0.4, 0.5, 0.6, 0.7]:
            s = np.arange(max_k + 1, dtype=np.float64)
            w = (1 - q) * (q ** s)
            w = w / np.sum(w)
            P_try = _compute_P(s, w)
            if P_try > best_P:
                best_P = P_try
                best_support = s.copy()
                best_weights = w.copy()

    # Final safety check
    if best_support is None:
        best_support = np.arange(32, dtype=np.float64)
        best_weights = np.ones(32, dtype=np.float64) / 32

    return best_support, best_weights


# EVOLVE-BLOCK-END
