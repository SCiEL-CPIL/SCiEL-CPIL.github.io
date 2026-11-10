# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: start at import time for the whole module
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds to spare before 630s


def _remaining_time():
    return 630.0 - (time.time() - _START_TIME) - _DEADLINE_MARGIN


def _shannon_entropy(p):
    """Shannon entropy in bits, ignoring zero probabilities."""
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log2(p))


def _project_entropy(P, r):
    """
    Compute the Shannon entropy of the projection of P along slope r.
    P is (10, 10) with P[y, x]. The projected value is x + r * y.
    Cells with equal projected value are pooled.
    """
    n = P.shape[0]
    xs = np.arange(n)
    ys = np.arange(n)
    # P[y, x] -> value = x + r * y
    values = xs[None, :] + r * ys[:, None]
    # For each unique value, sum the probabilities
    unique_vals, inverse = np.unique(values, return_inverse=True)
    proj = np.zeros(len(unique_vals))
    np.add.at(proj, inverse.ravel(), P.ravel())
    return _shannon_entropy(proj)


def _objective_ratio(P):
    """
    Compute D / M where D = H_{-1} and M = max(H_0, H_1, H_2).
    Returns (ratio, D, M, valid).
    """
    D = _project_entropy(P, -1.0)
    H0 = _project_entropy(P, 0.0)
    H1 = _project_entropy(P, 1.0)
    H2 = _project_entropy(P, 2.0)
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0, D, M, False
    ratio = D / M
    return ratio, D, M, True


def _objective_neg(P):
    """Negative of the ratio for minimization."""
    ratio, D, M, valid = _objective_ratio(P)
    if not valid:
        return 1e6
    return -ratio


def _normalize(P):
    """Clip at 0 and normalize to sum 1."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return np.ones((10, 10)) / 100.0
    return P / s


def _make_initial_patterns():
    """Generate diverse initial patterns for optimization."""
    n = 10
    patterns = []

    # 1. Uniform
    patterns.append(np.ones((n, n)) / (n * n))

    # 2. Concentrated on anti-diagonal (x - y = const) to boost H_{-1}
    #    Anti-diagonals: x - y = c for c in -9..9
    #    To maximize H_{-1}, we want many anti-diagonals to have similar mass.
    #    Uniform already does this. Let's try other structures.

    # 3. Row-concentrated (high H_0, low H_{-1}) - bad for us, but as starting point
    row_conc = np.zeros((n, n))
    for y in range(n):
        row_conc[y, :] = 1.0 / n
    patterns.append(row_conc / n)

    # 4. Column-concentrated
    col_conc = np.zeros((n, n))
    for x in range(n):
        col_conc[:, x] = 1.0 / n
    patterns.append(col_conc / n)

    # 5. Diagonal (x + y = const) concentrated
    diag = np.zeros((n, n))
    for y in range(n):
        diag[y, y] = 1.0 / n
    patterns.append(diag)

    # 6. Anti-diagonal concentrated
    anti_diag = np.zeros((n, n))
    for y in range(n):
        anti_diag[y, n - 1 - y] = 1.0 / n
    patterns.append(anti_diag)

    # 7. Random uniform-ish patterns
    np.random.seed(42)
    for _ in range(10):
        P = np.random.dirichlet(np.ones(n * n))
        patterns.append(P.reshape(n, n))

    # 8. Block patterns
    for block_size in [2, 3, 5]:
        P = np.zeros((n, n))
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                P[i:i+block_size, j:j+block_size] = 1.0
        P /= P.sum()
        patterns.append(P)

    # 9. Sparse patterns on specific structures
    #    Put mass on cells where x - y takes few values but x, x+y, x+2y spread out
    for c in range(-4, 5):
        P = np.zeros((n, n))
        for y in range(n):
            x = y + c
            if 0 <= x < n:
                P[y, x] = 1.0
        if P.sum() > 0:
            P /= P.sum()
            patterns.append(P)

    return patterns


def _coordinate_descent(P, max_iter=5000, tol=1e-10):
    """
    Coordinate descent: optimize one cell at a time.
    For each cell (y, x), adjust its weight to maximize the ratio.
    """
    n = 10
    P = _normalize(P).copy()
    best_P = P.copy()
    best_ratio, _, _, _ = _objective_ratio(P)

    for it in range(max_iter):
        if _remaining_time() < 5.0:
            break

        improved = False
        for y in range(n):
            for x in range(n):
                if _remaining_time() < 5.0:
                    return best_P

                # Try a range of new values for P[y, x]
                current = P[y, x]
                # We'll try scaling the cell weight
                # The remaining mass is 1 - current (after normalization)
                # Let's try new values: current * factor for various factors
                # Also try setting to 0
                candidates = []
                if current > 0:
                    for factor in [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]:
                        new_val = current * factor
                        if new_val > 0:
                            candidates.append(new_val)
                candidates.append(0.0)
                # Also try some absolute values
                for val in [0.001, 0.005, 0.01, 0.02, 0.05]:
                    candidates.append(val)

                best_candidate = current
                best_ratio_candidate = best_ratio

                for new_val in candidates:
                    if new_val == current:
                        continue
                    P[y, x] = new_val
                    P = _normalize(P)
                    ratio, _, _, valid = _objective_ratio(P)
                    if valid and ratio > best_ratio_candidate:
                        best_ratio_candidate = ratio
                        best_candidate = new_val

                if best_candidate != current:
                    P[y, x] = best_candidate
                    P = _normalize(P)
                    best_ratio = best_ratio_candidate
                    best_P = P.copy()
                    improved = True
                else:
                    P[y, x] = current
                    P = _normalize(P)

        if not improved:
            break

    return best_P


def _slsqp_optimize(P0, maxiter=500):
    """
    Optimize using SLSQP with constraints.
    Variables: P flattened (100 vars).
    Constraint: sum(P) = 1, P >= 0.
    """
    n = 10
    P0 = _normalize(P0).ravel()

    def neg_obj(p):
        P = np.clip(p, 0, None)
        s = P.sum()
        if s <= 1e-15:
            return 1e6
        P = P / s
        ratio, _, _, valid = _objective_ratio(P.reshape(n, n))
        if not valid:
            return 1e6
        return -ratio

    constraints = [
        {'type': 'eq', 'fun': lambda p: np.sum(np.clip(p, 0, None)) - 1.0},
    ]
    bounds = [(0, 1)] * (n * n)

    try:
        res = minimize(neg_obj, P0, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': maxiter, 'ftol': 1e-12})
        P = _normalize(res.x.reshape(n, n))
        return P
    except Exception:
        return P0


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    if _remaining_time() < 30.0:
        # Not enough time, return uniform
        return np.ones((10, 10), dtype=np.float64) / 100.0

    best_P = None
    best_ratio = 0.0

    # Generate initial patterns
    patterns = _make_initial_patterns()

    # Evaluate all patterns
    for P in patterns:
        if _remaining_time() < 10.0:
            break
        P = _normalize(P)
        ratio, D, M, valid = _objective_ratio(P)
        if valid and ratio > best_ratio:
            best_ratio = ratio
            best_P = P.copy()

    if best_P is None:
        best_P = np.ones((10, 10), dtype=np.float64) / 100.0
        best_ratio, _, _, _ = _objective_ratio(best_P)

    # Try SLSQP from the best pattern
    if _remaining_time() > 60.0:
        P_sl = _slsqp_optimize(best_P, maxiter=300)
        ratio_sl, _, _, valid_sl = _objective_ratio(P_sl)
        if valid_sl and ratio_sl > best_ratio:
            best_ratio = ratio_sl
            best_P = P_sl

    # Try coordinate descent from the best pattern
    if _remaining_time() > 30.0:
        P_cd = _coordinate_descent(best_P, max_iter=2000)
        ratio_cd, _, _, valid_cd = _objective_ratio(P_cd)
        if valid_cd and ratio_cd > best_ratio:
            best_ratio = ratio_cd
            best_P = P_cd

    # Try SLSQP from a few other good patterns
    if _remaining_time() > 60.0:
        for P in patterns[:5]:
            if _remaining_time() < 30.0:
                break
            P = _normalize(P)
            P_sl = _slsqp_optimize(P, maxiter=200)
            ratio_sl, _, _, valid_sl = _objective_ratio(P_sl)
            if valid_sl and ratio_sl > best_ratio:
                best_ratio = ratio_sl
                best_P = P_sl

    # Final coordinate descent polish
    if _remaining_time() > 20.0:
        P_final = _coordinate_descent(best_P, max_iter=3000)
        ratio_final, _, _, valid_final = _objective_ratio(P_final)
        if valid_final and ratio_final > best_ratio:
            best_ratio = ratio_final
            best_P = P_final

    # Final normalization and validation
    best_P = _normalize(best_P)
    ratio_final, _, _, valid_final = _objective_ratio(best_P)
    if not valid_final:
        # Fallback to uniform
        best_P = np.ones((10, 10), dtype=np.float64) / 100.0

    return best_P


# EVOLVE-BLOCK-END
