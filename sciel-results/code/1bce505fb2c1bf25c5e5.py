# EVOLVE-BLOCK-START
import numpy as np
import time


def _entropy_from_counts(counts):
    """Shannon entropy in bits from a count array (zeros ignored)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def _fast_ratio(P):
    """
    Fast computation of D/M using vectorized bincount-style approach.
    P[y, x] is the joint probability for x, y in 0..9.
    """
    if P.shape != (10, 10):
        return 0.0
    if not np.all(np.isfinite(P)):
        return 0.0
    if np.any(P < -1e-9):
        return 0.0
    total = P.sum()
    if total <= 1e-9:
        return 0.0

    P = P.copy()
    P[P < 0] = 0.0
    P = P / P.sum()

    y_idx, x_idx = np.indices((10, 10))
    flat_P = P.flatten()

    # x - y: range -9..9, offset by 9 to get 0..18
    v_minus1 = (x_idx - y_idx + 9).flatten()
    counts_minus1 = np.bincount(v_minus1, weights=flat_P, minlength=19)
    H_minus1 = _entropy_from_counts(counts_minus1)

    # x: range 0..9
    v_x = x_idx.flatten()
    counts_x = np.bincount(v_x, weights=flat_P, minlength=10)
    H_0 = _entropy_from_counts(counts_x)

    # x + y: range 0..18
    v_plus1 = (x_idx + y_idx).flatten()
    counts_plus1 = np.bincount(v_plus1, weights=flat_P, minlength=19)
    H_1 = _entropy_from_counts(counts_plus1)

    # x + 2y: range 0..27
    v_plus2 = (x_idx + 2 * y_idx).flatten()
    counts_plus2 = np.bincount(v_plus2, weights=flat_P, minlength=28)
    H_2 = _entropy_from_counts(counts_plus2)

    M = max(H_0, H_1, H_2)
    if M <= 1e-9:
        return 0.0

    return H_minus1 / M


def _softmax_ratio(P, beta=50.0):
    """
    Smooth surrogate for D/M using log-sum-exp for soft-max.
    Replaces max(H_0, H_1, H_2) with a soft-max to make the objective smoother.
    """
    if P.shape != (10, 10):
        return 0.0
    if not np.all(np.isfinite(P)):
        return 0.0
    if np.any(P < -1e-9):
        return 0.0
    total = P.sum()
    if total <= 1e-9:
        return 0.0

    P = P.copy()
    P[P < 0] = 0.0
    P = P / P.sum()

    y_idx, x_idx = np.indices((10, 10))
    flat_P = P.flatten()

    v_minus1 = (x_idx - y_idx + 9).flatten()
    counts_minus1 = np.bincount(v_minus1, weights=flat_P, minlength=19)
    H_minus1 = _entropy_from_counts(counts_minus1)

    v_x = x_idx.flatten()
    counts_x = np.bincount(v_x, weights=flat_P, minlength=10)
    H_0 = _entropy_from_counts(counts_x)

    v_plus1 = (x_idx + y_idx).flatten()
    counts_plus1 = np.bincount(v_plus1, weights=flat_P, minlength=19)
    H_1 = _entropy_from_counts(counts_plus1)

    v_plus2 = (x_idx + 2 * y_idx).flatten()
    counts_plus2 = np.bincount(v_plus2, weights=flat_P, minlength=28)
    H_2 = _entropy_from_counts(counts_plus2)

    # Soft-max of H_0, H_1, H_2
    hs = np.array([H_0, H_1, H_2])
    if np.max(hs) <= 1e-9:
        return 0.0
    m = np.max(hs)
    # log-sum-exp: m + log(sum(exp(h_i - m)))
    log_sum_exp = m + np.log(np.sum(np.exp(hs - m)))
    # Convert back from log to linear scale for the ratio
    soft_M = np.exp(log_sum_exp)  # This is sum(exp(h_i)), which is > max(h_i)
    # Actually, let's use a proper soft-max: (1/beta)*log(sum(exp(beta*h_i)))
    # But we want the soft-max to approximate max when beta is large
    beta = 50.0
    soft_M = (1.0 / beta) * np.log(np.sum(np.exp(beta * hs)))
    # For large beta, soft_M -> max(hs). For the ratio, we use soft_M in denominator.
    
    if soft_M <= 1e-9:
        return 0.0

    return H_minus1 / soft_M


def _nelder_mead_minimize(objective, x0, max_iter=500, tol=1e-10):
    """
    Nelder-Mead simplex method for minimization.
    Returns (x_best, f_best).
    """
    n = len(x0)
    # Initial simplex
    simplex = [np.array(x0, dtype=np.float64)]
    f_simplex = [objective(simplex[0])]
    
    # Create other vertices
    for i in range(n):
        xi = x0.copy()
        step = 0.1 if abs(x0[i]) < 1e-6 else 0.1 * abs(x0[i])
        xi[i] += step
        simplex.append(xi)
        f_simplex.append(objective(xi))
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    for iteration in range(max_iter):
        # Sort by function value
        order = np.argsort(f_simplex)
        simplex = [simplex[i] for i in order]
        f_simplex = [f_simplex[i] for i in order]
        
        # Check convergence
        if abs(f_simplex[-1] - f_simplex[0]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflect
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = objective(x_r)
        
        if f_r < f_simplex[0]:
            # Try expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_simplex[-1] = f_e
            else:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
        elif f_r < f_simplex[-2]:
            simplex[-1] = x_r
            f_simplex[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = objective(x_c)
            if f_c < f_simplex[-1]:
                simplex[-1] = x_c
                f_simplex[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_simplex[i] = objective(simplex[i])
    
    # Return best point
    best_idx = np.argmin(f_simplex)
    return simplex[best_idx], f_simplex[best_idx]


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    Optimizes the ratio D/M where D = H_{-1} and M = max(H_0, H_1, H_2).
    Uses Nelder-Mead on log-probability parameters with random restarts.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 630s limit

    n = 10
    rng = np.random.RandomState(42)

    # Precompute index arrays
    y_idx, x_idx = np.indices((n, n))

    # Helper: convert log-prob parameters to valid probability array
    def logparams_to_P(logp_flat):
        """Convert log-probability parameters to a valid 10x10 probability array."""
        # Use softmax to ensure positivity and normalization
        logp = logp_flat - np.max(logp_flat)  # numerical stability
        P_flat = np.exp(logp)
        P_flat = P_flat / P_flat.sum()
        return P_flat.reshape((n, n))

    # Objective: minimize negative ratio (we maximize ratio)
    def neg_ratio_from_logparams(logp_flat):
        P = logparams_to_P(logp_flat)
        return -_fast_ratio(P)

    # Objective with soft-max surrogate for smoother optimization
    def neg_soft_ratio_from_logparams(logp_flat):
        P = logparams_to_P(logp_flat)
        return -_softmax_ratio(P, beta=50.0)

    # Start with uniform distribution
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_score = _fast_ratio(best_P)

    # Initial log-prob parameters for uniform
    uniform_logp = np.full(n * n, np.log(1.0 / (n * n)))

    # Try structured initializations
    initializations = [uniform_logp.copy()]

    # Equal weight per diagonal
    P_diag = np.zeros((n, n), dtype=np.float64)
    diff = x_idx - y_idx
    for d in range(-9, 10):
        mask = (diff == d)
        num_cells = mask.sum()
        if num_cells > 0:
            P_diag[mask] = 1.0 / (19.0 * num_cells)
    P_diag = P_diag / P_diag.sum()
    score_diag = _fast_ratio(P_diag)
    if score_diag > best_score:
        best_P = P_diag.copy()
        best_score = score_diag
    # Convert to log params
    P_diag_clipped = np.clip(P_diag, 1e-15, None)
    initializations.append(np.log(P_diag_clipped.flatten()))

    # Concentrate on x=0 column
    P_col0 = np.zeros((n, n), dtype=np.float64)
    P_col0[:, 0] = 1.0 / n
    score_col0 = _fast_ratio(P_col0)
    if score_col0 > best_score:
        best_P = P_col0.copy()
        best_score = score_col0
    P_col0_clipped = np.clip(P_col0, 1e-15, None)
    initializations.append(np.log(P_col0_clipped.flatten()))

    # Random initializations
    for _ in range(5):
        random_P = rng.dirichlet(np.ones(n * n))
        random_P = random_P.reshape((n, n))
        score_rand = _fast_ratio(random_P)
        if score_rand > best_score:
            best_P = random_P.copy()
            best_score = score_rand
        random_P_clipped = np.clip(random_P, 1e-15, None)
        initializations.append(np.log(random_P_clipped.flatten()))

    # Nelder-Mead optimization with multiple restarts
    num_restarts = 12
    max_iter_per_restart = 300

    for restart in range(num_restarts):
        if time.time() > deadline - 120:
            break

        # Choose starting point
        if restart < len(initializations):
            x0 = initializations[restart].copy()
        else:
            # Perturb the best solution found so far
            best_logp = np.log(np.clip(best_P, 1e-15, None)).flatten()
            x0 = best_logp + rng.normal(0, 0.3, size=n * n)

        # Use soft-max surrogate for the optimization (smoother)
        x_best, f_best = _nelder_mead_minimize(
            neg_soft_ratio_from_logparams, x0, max_iter=max_iter_per_restart, tol=1e-10
        )

        # Evaluate the actual (non-smooth) ratio at the found point
        P_candidate = logparams_to_P(x_best)
        score_candidate = _fast_ratio(P_candidate)

        if score_candidate > best_score:
            best_P = P_candidate.copy()
            best_score = score_candidate

    # Final polish: run Nelder-Mead on the exact (non-smooth) objective from best point
    if time.time() < deadline - 30:
        best_logp = np.log(np.clip(best_P, 1e-15, None)).flatten()
        x_polished, f_polished = _nelder_mead_minimize(
            neg_ratio_from_logparams, best_logp, max_iter=200, tol=1e-12
        )
        P_polished = logparams_to_P(x_polished)
        score_polished = _fast_ratio(P_polished)
        if score_polished > best_score:
            best_P = P_polished.copy()
            best_score = score_polished

    # Final cleanup: ensure validity
    best_P[best_P < 0] = 0
    total = best_P.sum()
    if total > 1e-9:
        best_P = best_P / total
    else:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)

    return best_P


# EVOLVE-BLOCK-END
