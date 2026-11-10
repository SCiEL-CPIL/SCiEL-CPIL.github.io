# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximize D/M where D = H_{-1} (entropy of x-y) and M = max(H_0, H_1, H_2).
    Uses analytic gradient with precomputed pooling matrices for fast optimization.
    """
    n = 10
    size = n * n  # 100

    # Precompute pooling matrices for slopes -1, 0, 1, 2
    def build_pooling_matrix(r):
        """Build pooling matrix A such that A @ p_flat gives marginal distribution of x + r*y."""
        if r == -1:
            min_val = -9
            max_val = 9
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
        p = np.asarray(p, dtype=np.float64)
        mask = p > 1e-15
        if not np.any(mask):
            return 0.0
        pm = p[mask]
        return float(-np.sum(pm * np.log2(pm)))

    def compute_value_and_grad(p_vec):
        """
        Given a probability vector p (length 100), compute value = D/M and its analytic gradient.
        Returns (value, grad) where grad is d(value)/d(p_vec).
        """
        p_vec = np.asarray(p_vec, dtype=np.float64)
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
            return (0.0, np.zeros(size))

        value = D / M

        # Analytic gradient of Shannon entropy H(m) = -sum(m_i * log2(m_i))
        # dH/dm_i = -log2(m_i) - 1/ln(2) for m_i > 0
        # dm_i/dp_j = A[i,j]
        # So dH/dp_j = sum_i (dH/dm_i) * A[i,j] = -sum_{i: m_i>0} A[i,j] * (log2(m_i) + 1/ln(2))

        def entropy_grad(m, A):
            """Gradient of Shannon entropy w.r.t. p, given marginal m and pooling matrix A."""
            mask = m > 1e-15
            if not np.any(mask):
                return np.zeros(size)
            # dH/dm_i = -log2(m_i) - 1/ln(2)
            dH_dm = np.zeros_like(m)
            dH_dm[mask] = -np.log2(m[mask]) - 1.0 / np.log(2.0)
            # dH/dp = A^T @ dH_dm
            return A.T @ dH_dm

        grad_D = entropy_grad(m_minus1, A_minus1)

        # For M = max(H0, H1, H2), the gradient is the gradient of the active (max) entropy
        # If there are ties, we use a weighted average or just pick one (subgradient)
        H_vals = [H0, H1, H2]
        A_vals = [A_0, A_1, A_2]
        m_vals = [m_0, m_1, m_2]

        # Find which entropy is max (use subgradient: average of all max entropies)
        max_M = M
        max_indices = [i for i, h in enumerate(H_vals) if abs(h - max_M) < 1e-12]

        grad_M = np.zeros(size)
        for idx in max_indices:
            grad_M += entropy_grad(m_vals[idx], A_vals[idx])
        grad_M /= len(max_indices)

        # d(D/M)/dp = (grad_D * M - D * grad_M) / M^2
        grad_value = (grad_D * M - D * grad_M) / (M * M)

        return (value, grad_value)

    def objective_and_jac(p_vec):
        """For scipy minimize: return (-value, -grad_value)."""
        value, grad_value = compute_value_and_grad(p_vec)
        return (-value, -grad_value)

    # Multiple starting points for robustness
    best_p = None
    best_value = -1.0

    # Starting point 1: uniform
    starts = []
    starts.append(np.ones(size) / size)

    # Starting point 2: concentrate on diagonal (x=y) to maximize x-y spread
    p_diag = np.zeros(size)
    for i in range(n):
        p_diag[i * n + i] = 1.0 / n
    starts.append(p_diag)

    # Starting point 3: concentrate on anti-diagonal (x+y=9)
    p_anti = np.zeros(size)
    for y in range(n):
        x = n - 1 - y
        p_anti[y * n + x] = 1.0 / n
    starts.append(p_anti)

    # Starting point 4: random sparse
    np.random.seed(42)
    p_rand = np.random.dirichlet(np.ones(size))
    starts.append(p_rand)

    for p0 in starts:
        p0 = np.maximum(p0, 1e-10)
        p0 = p0 / np.sum(p0)

        try:
            result = minimize(
                lambda p: objective_and_jac(p)[0],
                p0,
                jac=lambda p: objective_and_jac(p)[1],
                method='L-BFGS-B',
                bounds=Bounds(1e-10, 1.0),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            p_opt = result.x
            val, _ = compute_value_and_grad(p_opt)
            if val > best_value:
                best_value = val
                best_p = p_opt.copy()
        except Exception:
            continue

    if best_p is None:
        best_p = np.ones(size) / size

    # Reshape and normalize
    P = best_p.reshape((n, n))
    P = np.maximum(P, 0.0)
    total = np.sum(P)
    if total > 1e-9:
        P = P / total
    else:
        P = np.ones((n, n)) / size

    return P


# EVOLVE-BLOCK-END
