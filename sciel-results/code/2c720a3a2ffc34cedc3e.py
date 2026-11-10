# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximize D/M where D = H_{-1} (entropy of x-y) and M = max(H_0, H_1, H_2).
    Uses gradient-based optimization with precomputed pooling matrices.
    """
    n = 10
    size = n * n  # 100

    # Precompute pooling matrices for slopes -1, 0, 1, 2
    # For slope r, value = x + r*y
    # P[y, x] corresponds to index y*n + x

    def build_pooling_matrix(r):
        """Build pooling matrix A such that A @ p_flat gives marginal distribution of x + r*y."""
        # Determine range of x + r*y for x,y in 0..9
        # x in [0,9], y in [0,9]
        # values = x + r*y
        if r == -1:
            min_val = 0 + (-1) * 9  # x=0, y=9 -> -9
            max_val = 9 + (-1) * 0  # x=9, y=0 -> 9
        elif r == 0:
            min_val = 0
            max_val = 9
        elif r == 1:
            min_val = 0
            max_val = 18
        elif r == 2:
            min_val = 0
            max_val = 27
        else:
            min_val = 0
            max_val = 9 + abs(r) * 9

        num_bins = int(max_val - min_val + 1)
        A = np.zeros((num_bins, size), dtype=np.float64)

        for y in range(n):
            for x in range(n):
                val = x + r * y
                idx = y * n + x
                bin_idx = int(val - min_val)
                if 0 <= bin_idx < num_bins:
                    A[bin_idx, idx] = 1.0
        return A

    A_minus1 = build_pooling_matrix(-1)
    A_0 = build_pooling_matrix(0)
    A_1 = build_pooling_matrix(1)
    A_2 = build_pooling_matrix(2)

    def shannon_entropy(p):
        """Compute Shannon entropy in bits, ignoring zero probabilities."""
        p = np.asarray(p)
        p = p[p > 1e-15]
        if len(p) == 0:
            return 0.0
        return float(-np.sum(p * np.log2(p)))

    def objective_and_grad(p_vec):
        """
        Given a probability vector p (length 100), compute:
        - D = H_{-1}
        - M = max(H_0, H_1, H_2)
        - value = D / M
        - gradient of value w.r.t. p_vec (approximate via finite differences or analytic)
        Returns (negative value, gradient of negative value)
        """
        p_vec = np.asarray(p_vec)
        # Clip to valid range
        p_vec = np.maximum(p_vec, 1e-15)
        p_vec = p_vec / np.sum(p_vec)

        # Compute marginals
        m_minus1 = A_minus1 @ p_vec
        m_0 = A_0 @ p_vec
        m_1 = A_1 @ p_vec
        m_2 = A_2 @ p_vec

        D = shannon_entropy(m_minus1)
        H0 = shannon_entropy(m_0)
        H1 = shannon_entropy(m_1)
        H2 = shannon_entropy(m_2)
        M = max(H0, H1, H2)

        if M < 1e-9:
            return (1e10, np.zeros(size))

        value = D / M

        # Gradient via finite differences for robustness
        eps = 1e-5
        grad = np.zeros(size)
        for i in range(size):
            p_plus = p_vec.copy()
            p_plus[i] += eps
            p_plus = np.maximum(p_plus, 1e-15)
            p_plus = p_plus / np.sum(p_plus)

            mp_minus1 = A_minus1 @ p_plus
            mp_0 = A_0 @ p_plus
            mp_1 = A_1 @ p_plus
            mp_2 = A_2 @ p_plus

            Dp = shannon_entropy(mp_minus1)
            Hp0 = shannon_entropy(mp_0)
            Hp1 = shannon_entropy(mp_1)
            Hp2 = shannon_entropy(mp_2)
            Mp = max(Hp0, Hp1, Hp2)

            if Mp < 1e-9:
                grad[i] = 0
            else:
                value_plus = Dp / Mp
                grad[i] = (value_plus - value) / eps

        # We want to maximize value, so minimize -value
        return (-value, -grad)

    # Initial guess: uniform distribution
    p0 = np.ones(size) / size

    # Bounds: each entry in [1e-8, 1]
    bounds = Bounds(1e-8, 1.0)

    # Constraint: sum(p) = 1
    A_eq = np.ones((1, size))
    b_eq = [1.0]
    constraint = LinearConstraint(A_eq, 1.0, 1.0)

    # Run optimization
    try:
        result = minimize(
            lambda p: objective_and_grad(p)[0],
            p0,
            jac=lambda p: objective_and_grad(p)[1],
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraint,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        p_opt = result.x
    except Exception:
        p_opt = p0

    # Reshape and normalize
    P = p_opt.reshape((n, n))
    P = np.maximum(P, 0.0)
    total = np.sum(P)
    if total > 1e-9:
        P = P / total
    else:
        P = np.ones((n, n)) / size

    return P


# EVOLVE-BLOCK-END
