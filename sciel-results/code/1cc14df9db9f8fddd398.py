import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin before 630s limit

    m = 12
    alpha = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # L_k^{(alpha)}(x) = sum_{j=0}^k A[k,j] x^j
    # For alpha=1: A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + 1, j + 1) / math.factorial(j)

    # c = A^{-1} p, where p is ascending polynomial coefficients
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)

    # Norm weights: <L_k, L_l> = delta_kl * (k+alpha+1) for alpha=1 -> (k+1)
    weights = np.arange(m + 1) + 1  # [1, 2, 3, ..., 13]

    def compute_norm_sq(roots):
        """
        Compute squared norm of P(x) = prod(x - r_i) in Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in ascending order
        # P(x) = prod(x - r_i)
        # Use iterative multiplication to avoid numerical issues with np.poly
        p = np.array([1.0])  # start with 1
        for r in roots:
            # p = p * (x - r)
            new_p = np.zeros(len(p) + 1)
            new_p[1:] = p  # x * p
            new_p[:-1] -= r * p  # -r * p
            p = new_p

        # p is now ascending: [constant, x, x^2, ..., x^12]
        c = A_inv @ p

        # Norm squared: sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * weights)

        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize 1/norm_sq (equivalent to maximizing norm_sq).
        log_roots: array of 12 log-roots.
        """
        roots = np.exp(log_roots)

        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10

        norm_sq = compute_norm_sq(roots)

        if norm_sq <= 1e-10:
            return 1e10

        return 1.0 / norm_sq

    def gradient_cost(log_roots):
        """
        Numerical gradient for L-BFGS-B.
        """
        eps = 1e-7
        grad = np.zeros(len(log_roots))
        for i in range(len(log_roots)):
            lp = log_roots.copy()
            lm = log_roots.copy()
            lp[i] += eps
            lm[i] -= eps
            grad[i] = (cost_func(lp) - cost_func(lm)) / (2 * eps)
        return grad

    # Bounds for log_roots
    # Roots should be positive. Allow wide range.
    bounds = [(-3.0, 12.0)] * 12

    best_val = 1e10
    best_log_roots = None
    best_roots = None

    # Generate diverse initial points
    def generate_initial_points(n_points):
        points = []
        rng = np.random.RandomState(42)

        for i in range(n_points):
            if i % 4 == 0:
                # Geometric sequence with random base and ratio
                base = rng.uniform(0.5, 10.0)
                ratio = rng.uniform(1.2, 3.0)
                roots = base * (ratio ** np.arange(12))
            elif i % 4 == 1:
                # Log-uniform random
                log_roots = rng.uniform(-2.0, 10.0, size=12)
                roots = np.exp(log_roots)
            elif i % 4 == 2:
                # Clustered: some close together, some far
                centers = rng.uniform(0.5, 5.0, size=3)
                roots = []
                for c in centers:
                    for j in range(4):
                        roots.append(c * (1.0 + 0.1 * j))
                roots = np.array(roots)
            else:
                # Mixed: geometric with perturbation
                base = rng.uniform(1.0, 5.0)
                ratio = rng.uniform(1.5, 2.5)
                roots = base * (ratio ** np.arange(12))
                roots = roots * (1.0 + 0.1 * rng.randn(12))

            # Ensure positive
            roots = np.maximum(roots, 1e-3)
            points.append(np.log(roots))

        return points

    # Run multiple L-BFGS-B optimizations from different starting points
    initial_points = generate_initial_points(60)

    for idx, init_log_roots in enumerate(initial_points):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:  # Leave 30s for final polish
            break

        try:
            result = minimize(
                cost_func,
                init_log_roots,
                jac=gradient_cost,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )

            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)

        except Exception:
            pass

    # Fallback: seed from geometric sequence
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_log_roots = np.log(seed_roots)
    seed_val = cost_func(seed_log_roots)

    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Final high-precision polish
    if time.time() - start_time < deadline - 10 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func,
                best_log_roots,
                jac=gradient_cost,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)

    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
